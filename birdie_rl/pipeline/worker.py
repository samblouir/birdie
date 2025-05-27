# worker.py
"""
Worker module. Each worker now accumulates samples to form a full batch
before sending it to the sample_queue.
Profiling re-enabled for the worker's full run duration.
Enhanced _print_profile_stats to check profiler.getstats().
Logging is cleaned up.
Updated to use Batcher from packer_batcher2.py.
"""

import os
import sys
import time
import queue
import numpy as np
import traceback
import multiprocessing as mp
from typing import Any, Callable, Union, Iterable
from functools import partial
# Updated import to use packer_batcher2
from birdie_rl.pipeline.packer_batcher2 import Batcher
from birdie_rl.load_objective import load_objective
from datasets import load_dataset

import cProfile
import pstats
import io


def _default_text_source(worker_id: int, total_workers: int, split: str = "train", rng_seed: int = 0):
	pid = os.getpid()
	# print(f"[_default_text_source Worker {worker_id} PID {pid}] Initializing for TinyStories, split: {split}.", flush=True)

	try:
		ds_full = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
		ds_shard = ds_full.shard(num_shards=total_workers, index=worker_id, contiguous=True)
		data_list = list(ds_shard)
		local_rng = np.random.default_rng(rng_seed + worker_id + pid)
		local_rng.shuffle(data_list)
	except Exception as e:
		print(f"[_default_text_source Worker {worker_id} PID {pid}] CRITICAL: Failed to load/process TinyStories: {e}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
		while True: yield {"text":"ERROR: DATASET LOADING FAILED IN WORKER"}; time.sleep(1)

	if not data_list:
		print(f"[_default_text_source Worker {worker_id} PID {pid}] Warning: Data list is empty. Worker will yield no data.", flush=True)
		while True: yield {"text":""}; time.sleep(1)

	list_idx = 0
	try:
		while True:
			yield data_list[list_idx]
			list_idx = (list_idx + 1) % len(data_list)
	except Exception as e_yield:
		print(f"[Worker {worker_id} PID {pid} _default_text_source] EXCEPTION during yield: {e_yield}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
	# finally:
		# print(f"[_default_text_source Worker {worker_id} PID {pid}] Exiting.", flush=True)


class Worker:
	def __init__(
		self,
		worker_id: int,
		total_workers: int,
		tasks_queue: mp.Queue,
		results_queue: mp.Queue, # This argument is kept for signature consistency but not used by Worker directly
		sample_queue: mp.Queue,
		data_generator: Union[Callable, Iterable] = None,
		sequence_length: int = 1024,
		min_seq_len_for_packing: int = 32,
		tokenizer=None,
		split: str = None,
		text_grabber_fn: Callable[[Any], str] = None,
		infinite_loop: bool = True,
		start_generating_id: int = 2,
		latent_token_id: int = 1,
		max_samples_per_packer: float = float('inf'),
		rng_seed: int = 0,
		config: dict = None, # Main config dictionary
	):
		self.worker_id = worker_id
		self.total_workers = total_workers
		self.tasks_queue = tasks_queue
		self.sample_queue = sample_queue
		self.sequence_length = sequence_length
		self.min_seq_len_for_packing = min_seq_len_for_packing
		self.infinite_loop = infinite_loop
		self.split = split
		self.should_stop = False
		self.base_rng_seed = rng_seed
		self.main_config = config if config is not None else {}

		self.target_batch_size = self.main_config.get('batch_size', 1) # Get target batch size from main config
		self.collected_samples_for_batch = [] # Initialize list to store samples for a full batch

		if text_grabber_fn is None:
			def default_text_grabber(x_item):
				if isinstance(x_item, dict): return x_item.get("text", "")
				elif isinstance(x_item, str): return x_item
				return ""
			self.text_grabber_fn = default_text_grabber
		else:
			self.text_grabber_fn = text_grabber_fn

		if not callable(data_generator):
			self.data_generator_fn_callable = partial(_default_text_source,
												worker_id=self.worker_id, total_workers=self.total_workers,
												split=self.split, rng_seed=self.base_rng_seed + self.worker_id)
		else:
			self.data_generator_fn_callable = partial(data_generator,
													 split=self.split, worker_id=self.worker_id,
													 num_workers=self.total_workers,
													 rng_seed=self.base_rng_seed + self.worker_id)

		self.data_iter = None
		self.dataset_reset_counter = 0
		self.rng = np.random.default_rng(self.base_rng_seed + self.worker_id + os.getpid())

		self.tokenizer = tokenizer
		if self.tokenizer is None: raise ValueError(f"[Worker {self.worker_id}] ERROR: A tokenizer is required.")

		# Internal packer still has batch_size: 1
		config_for_internal_packer = {
			"batch_size": 1,
			"sequence_length": self.sequence_length,
			"minimum_sequence_length": self.min_seq_len_for_packing,
			"start_generating_id": start_generating_id,
			"latent_token_id": latent_token_id,
			"max_samples_per_packer": max_samples_per_packer, # This applies to the internal packer
			"seed": self.base_rng_seed + self.worker_id + os.getpid() + int(time.time()*1000) % 100000,
		}
		self.internal_packer = Batcher(config=config_for_internal_packer)

		self.leftover_text = ""
		self.objectives_info = []
		self.og_probs = np.array([], dtype=np.float32)
		self.objective_cache = {}

	def _log_print(self, *args, verbosity_level=2, **kwargs):
		min_worker_verbosity = 2
		if verbosity_level <= min_worker_verbosity:
			current_pid = os.getpid()
			print(f"[Worker {self.worker_id} (Split: {self.split}, PID: {current_pid})]", *args, **kwargs, flush=True)

	def _send_batch(self):
		"""Stacks collected samples and sends them to the sample_queue."""
		if not self.collected_samples_for_batch:
			return

		try:
			# Ensure all collected samples have the same keys for stacking
			if not all(isinstance(s, dict) for s in self.collected_samples_for_batch):
				self._log_print("Error: Not all collected samples are dictionaries.", verbosity_level=0)
				self.collected_samples_for_batch = []
				return

			first_sample_keys = self.collected_samples_for_batch[0].keys()
			if not all(s.keys() == first_sample_keys for s in self.collected_samples_for_batch):
				self._log_print("Error: Collected samples have inconsistent keys.", verbosity_level=0)
				# Potentially log the differing keys for debugging
				self.collected_samples_for_batch = []
				return

			stacked_dict = {
				key: np.stack([s[key] for s in self.collected_samples_for_batch])
				for key in first_sample_keys
			}
			
			item_to_send = {
				"worker_id": self.worker_id,
				"stacked_batch_data": stacked_dict # This is the fully formed batch
			}

			while not self.should_stop: # Retry putting to queue if full, respecting stop_event
				try:
					self.sample_queue.put(item_to_send, timeout=0.1)
					# self._log_print(f"Sent a batch of {len(self.collected_samples_for_batch)} samples.", verbosity_level=1)
					break 
				except queue.Full:
					time.sleep(0.005) 
				except Exception as e_put_batch:
					self._log_print(f"ERROR putting full batch to sample_queue: {e_put_batch}", verbosity_level=0)
					self.should_stop = True # Critical error, attempt to stop
					break
		except Exception as e_stack:
			self._log_print(f"Error stacking batch: {e_stack}", verbosity_level=0)
			traceback.print_exc(file=sys.stdout)
		finally:
			self.collected_samples_for_batch = [] # Clear after attempting to send


	def close(self):
		# self._log_print(f"close() called. Flushing any remaining collected samples.", verbosity_level=1)
		if self.collected_samples_for_batch: # Flush any partial batch
			self._send_batch()

		# Close the internal packer (which itself might have a partially filled sequence)
		if self.internal_packer and self.internal_packer.is_ready() == "ready":
			try:
				_ = self.internal_packer.pop(peek=False)
			except Exception as e_pop:
				self._log_print(f"Exception during internal_packer.pop() in close(): {e_pop}", verbosity_level=0)
		elif self.internal_packer:
			self.internal_packer.reset()
		# self._log_print(f"close() finished.", verbosity_level=1)

	def initialize_data_iterator(self):
		self.dataset_reset_counter += 1
		if (not self.infinite_loop) and (self.dataset_reset_counter > 1):
			self.should_stop = True; return
		try:
			self.data_iter = iter(self.data_generator_fn_callable())
		except Exception as e:
			self._log_print(f"EXCEPTION during data_generator_fn_callable() call or iter(): {e}", verbosity_level=0)
			traceback.print_exc(file=sys.stdout); sys.stdout.flush(); self.should_stop = True

	def _get_objective_instance(self, objective_name: str, config_overrides: dict):
		if 'tokenizer' not in config_overrides:
			config_overrides['tokenizer'] = self.tokenizer
		obj_config_tuple = (objective_name, frozenset(config_overrides.items()))
		if obj_config_tuple in self.objective_cache:
			return self.objective_cache[obj_config_tuple]
		instance = load_objective(objective_name, config_overrides)
		self.objective_cache[obj_config_tuple] = instance
		return instance

	def _try_get_instructions(self):
		try: data = self.tasks_queue.get(timeout=0.01)
		except queue.Empty: return None
		except Exception as e: self._log_print(f"Exception getting from tasks_queue: {e}", verbosity_level=0); return None
		if data is None: self.should_stop = True; return False # Corrected: set should_stop and return False

		new_objectives_info = data.get("objectives", [])
		if self.objectives_info == new_objectives_info: return True
		self.objectives_info = new_objectives_info
		self.objective_cache.clear()
		if not self.objectives_info: self.og_probs = np.array([], dtype=np.float32); return True
		arr = np.float32([obj.get("prob", 1.0) for obj in self.objectives_info])
		s = arr.sum()
		if s > 0: arr /= s
		elif len(self.objectives_info) > 0 : arr = np.float32([1.0 / len(self.objectives_info)] * len(self.objectives_info))
		else: arr = np.array([], dtype=np.float32)
		self.og_probs = arr
		return True

	def _produce_one_sample(self): # This method now aims to produce one *packed sequence*
		if not self.objectives_info or len(self.og_probs) == 0:
			time.sleep(0.05); return

		raw_data_item = None
		text_to_process = None

		if self.leftover_text:
			text_to_process = self.leftover_text; self.leftover_text = ""
		else:
			if self.data_iter is None: self.initialize_data_iterator()
			if self.should_stop or self.data_iter is None: return
			try:
				raw_data_item = next(self.data_iter)
				text_to_process = self.text_grabber_fn(raw_data_item)
			except StopIteration: self.initialize_data_iterator(); return
			except Exception as e_next:
				self._log_print(f"Error getting data: {e_next}", verbosity_level=0); self.should_stop = True; return

		if not (isinstance(text_to_process, str) and text_to_process.strip()):
			return

		try:
			obj_idx = self.rng.choice(len(self.objectives_info), p=self.og_probs)
			obj_info = self.objectives_info[obj_idx]
		except ValueError as e_choice:
			self._log_print(f"Error choosing objective: {e_choice}", verbosity_level=0); return

		objective_name = obj_info["name"]
		base_config_overrides = obj_info.get("config_overrides", {})
		cfg_overrides = base_config_overrides.copy() if isinstance(base_config_overrides, dict) else {}
		
		# The internal_packer's remaining space for a single sequence
		cfg_overrides["remaining_space"] = self.internal_packer.get_remaining_space(max_or_min="max") 
		cfg_overrides["tokenizer"] = self.tokenizer
		cfg_overrides["rng_seed"] = self.rng.integers(0, 2**32 -1)

		try: objective_instance = self._get_objective_instance(objective_name, cfg_overrides)
		except Exception as e_get_obj:
			self._log_print(f"Error getting objective '{objective_name}': {e_get_obj}", verbosity_level=0)
			self.leftover_text = text_to_process; return

		original_text_for_this_attempt = text_to_process
		result = objective_instance(text_to_process)

		if result.get("status") == "not_enough_space":
			current_internal_packer_instance = self.internal_packer.packers[0]
			if current_internal_packer_instance and current_internal_packer_instance.data_index > 0:
				packed_data_from_internal_packer = self.internal_packer.pop(peek=False)
				if packed_data_from_internal_packer:
					single_packed_sequence = {k: v[0] for k, v in packed_data_from_internal_packer.items()}
					if single_packed_sequence.get("input_ids", np.array([])).any():
						self.collected_samples_for_batch.append(single_packed_sequence)
						if len(self.collected_samples_for_batch) >= self.target_batch_size:
							self._send_batch()
			self.leftover_text = original_text_for_this_attempt; return

		self.leftover_text = result.get("unused_input_string", "")
		if result.get("status") != "ok": return

		input_ids = result.get("input_ids"); label_ids = result.get("label_ids")
		if not (input_ids is not None and hasattr(input_ids, '__len__') and len(input_ids) > 0): return
		if label_ids is None: label_ids = []
		
		input_ids_np = np.array(input_ids, dtype=np.int32)
		label_ids_np = np.array(label_ids, dtype=np.int32)

		if input_ids_np.size > 0 and label_ids_np.size == 0: return

		try:
			if not self.internal_packer.can_accept(input_ids_np, label_ids_np):
				if self.internal_packer.packers[0].data_index > 0:
					packed_data_from_internal_packer = self.internal_packer.pop(peek=False)
					if packed_data_from_internal_packer:
						single_packed_sequence = {k: v[0] for k, v in packed_data_from_internal_packer.items()}
						if single_packed_sequence.get("input_ids", np.array([])).any():
							self.collected_samples_for_batch.append(single_packed_sequence)
							if len(self.collected_samples_for_batch) >= self.target_batch_size:
								self._send_batch()
				if not self.internal_packer.can_accept(input_ids_np, label_ids_np):
					self.leftover_text = original_text_for_this_attempt; return
			
			packer_status = self.internal_packer.add(input_ids_np, label_ids_np)
		except ValueError as e_packer_add:
			self._log_print(f"ValueError from internal_packer.add for '{objective_name}': {e_packer_add}", verbosity_level=0)
			self.leftover_text = original_text_for_this_attempt; return

		if packer_status == "ready": # Internal packer (for one sequence) is ready
			packed_data_from_internal_packer = self.internal_packer.pop(peek=False)
			if packed_data_from_internal_packer:
				single_packed_sequence = {k: v[0] for k, v in packed_data_from_internal_packer.items()}
				if single_packed_sequence.get("input_ids", np.array([])).any():
					self.collected_samples_for_batch.append(single_packed_sequence)
					if len(self.collected_samples_for_batch) >= self.target_batch_size:
						self._send_batch()

	def run(self, profile: bool = False):
		self.rng = np.random.default_rng(self.base_rng_seed + self.worker_id + os.getpid() + int(time.time()*1000) % 100000)
		profiler = cProfile.Profile() if profile else None
		if profiler: profiler.enable()

		self.initialize_data_iterator()
		if self.should_stop:
			if profiler: profiler.disable()
			self.close() # Ensure remaining samples are flushed
			try: self.sample_queue.put(None, timeout=0.1)
			except Exception: pass
			if profiler: self._print_profile_stats(profiler)
			return

		last_instruction_check_time = time.time()
		try:
			while not self.should_stop:
				if time.time() - last_instruction_check_time > 0.05:
					if self._try_get_instructions() is False: break # Stop if sentinel received
					last_instruction_check_time = time.time()
				if not self.objectives_info: time.sleep(0.01); continue
				self._produce_one_sample()
				if hasattr(self.sample_queue, 'maxsize') and self.sample_queue.qsize() >= self.sample_queue.maxsize * 0.9: # Check if queue is getting too full
					time.sleep(0.005)
		except KeyboardInterrupt: self.should_stop = True
		except Exception as e:
			self._log_print(f"Unhandled EXCEPTION: {e}", verbosity_level=0); traceback.print_exc(); self.should_stop = True
		finally:
			if profiler: profiler.disable()
			self.close() # Flush any remaining collected samples
			try: self.sample_queue.put(None, timeout=1.0)
			except Exception: pass
			if profiler: self._print_profile_stats(profiler)

	def _print_profile_stats(self, profiler):
		self._log_print(f"\n--- Profiling Results for Worker {self.worker_id} ---", verbosity_level=0)
		s = io.StringIO()
		raw_stats = profiler.getstats()
		if raw_stats:
			try:
				ps = pstats.Stats(profiler, stream=s); ps.sort_stats('cumtime'); ps.print_stats(20)
				output = s.getvalue()
				if output.strip(): self._log_print(output, verbosity_level=0)
				else: self._log_print("pstats.print_stats() produced no output.", verbosity_level=0)
			except Exception as e_pstats: self._log_print(f"Error printing pstats: {e_pstats}", verbosity_level=0)
		else: self._log_print("No profiling stats collected.", verbosity_level=0)

if __name__ == "__main__":
	print("--- Testing _default_text_source (from worker.py) ---", flush=True)
	try:
		gen = _default_text_source(worker_id=0, total_workers=1, split="train", rng_seed=42)
		for i in range(5):
			item = next(gen)
			if isinstance(item, dict) and "text" in item and item["text"]: pass
			else: print("  No valid text found in item.", flush=True)
	except Exception as e_test_default:
		print(f"Error testing _default_text_source: {e_test_default}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
