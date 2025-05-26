# worker.py
"""
Worker module. Profiling re-enabled for the worker's full run duration.
Enhanced _print_profile_stats to check profiler.getstats().
Logging is cleaned up.
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
from birdie_rl.pipeline.packer_batcher import Batcher 
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
		results_queue: mp.Queue, 
		sample_queue: mp.Queue,  
		data_generator: Union[Callable, Iterable] = None,
		sequence_length: int = 1024,
		min_seq_len_for_packing: int = 32,
		tokenizer=None,
		split: str = None,
		text_grabber_fn: Callable[[Any], str] = None,
		infinite_loop: bool = True, 
		start_generating_paradigm: str = "\n<|assistant|>\n",
		rng_seed: int = 0, 
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
		
		# self._log_print(f"__init__ starting. Tokenizer type: {type(tokenizer)}", verbosity_level=2)

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

		self.packer = Batcher(
			batch_size=1, 
			tokenizer=self.tokenizer, 
			sequence_length=self.sequence_length,
			minimum_sequence_length=self.min_seq_len_for_packing,
			start_generating_paradigm=start_generating_paradigm 
		)

		self.leftover_text = ""
		self.objectives_info = []
		self.og_probs = np.array([], dtype=np.float32)
		self.objective_cache = {} 
		# self._log_print(f"__init__ finished.", verbosity_level=2)

	def _log_print(self, *args, verbosity_level=2, **kwargs): 
		min_worker_verbosity = 2 
		if verbosity_level <= min_worker_verbosity:
			current_pid = os.getpid() 
			print(f"[Worker {self.worker_id} (Split: {self.split}, PID: {current_pid})]", *args, **kwargs, flush=True)

	def close(self):
		# self._log_print(f"close() called.", verbosity_level=1)
		if self.packer and self.packer.is_ready() == "ready": 
			try:
				_ = self.packer.pop(peek=False) 
			except Exception as e_pop:
				self._log_print(f"Exception during packer.pop() in close(): {e_pop}", verbosity_level=0)
		elif self.packer: 
			self.packer.reset()
		# self._log_print(f"close() finished.", verbosity_level=1)

	def initialize_data_iterator(self):
		self.dataset_reset_counter += 1
		# self._log_print(f"initialize_data_iterator call #{self.dataset_reset_counter}. Infinite loop: {self.infinite_loop}", verbosity_level=1)
		if (not self.infinite_loop) and (self.dataset_reset_counter > 1):
			self._log_print(f"Not infinite_loop and dataset reset before => signaling stop.", verbosity_level=0)
			self.should_stop = True; return
		try:
			self.data_iter = iter(self.data_generator_fn_callable())
			# self._log_print(f"Data iterator successfully initialized.", verbosity_level=1)
		except Exception as e:
			self._log_print(f"EXCEPTION during data_generator_fn_callable() call or iter(): {e}", verbosity_level=0)
			traceback.print_exc(file=sys.stdout); sys.stdout.flush(); self.should_stop = True 

	def _get_objective_instance(self, objective_name: str, config_overrides: dict):
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
		
		if data is None: 
			self._log_print(f"Received None sentinel from tasks_queue. Will stop.", verbosity_level=1) 
			return False 
		
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

	def _produce_one_sample(self):
		if not self.objectives_info or len(self.og_probs) == 0: 
			time.sleep(0.05); return

		raw_data_item = None 
		text_to_process = None 
		
		if self.leftover_text: 
			text_to_process = self.leftover_text; self.leftover_text = "" 
		else: 
			if self.data_iter is None: 
				self.initialize_data_iterator()
			if self.should_stop or self.data_iter is None: 
				return 
			try:
				raw_data_item = next(self.data_iter)
				text_to_process = self.text_grabber_fn(raw_data_item)
			except StopIteration: 
				self.initialize_data_iterator(); return 
			except Exception as e_next: 
				self._log_print(f"_produce_one_sample: Error getting data from iterator: {e_next}", verbosity_level=0)
				traceback.print_exc(file=sys.stdout); sys.stdout.flush()
				self.should_stop = True; return
		
		is_valid_text = isinstance(text_to_process, str) and text_to_process.strip()
		
		if not is_valid_text:
			self._log_print(f"_produce_one_sample: text_grabber_fn returned invalid/empty/whitespace. "
						  f"Raw item type: {type(raw_data_item)}, content: '{str(raw_data_item)[:100]}'. "
						  f"Text processed (type: {type(text_to_process)}, len {len(text_to_process if text_to_process else '')}): '{str(text_to_process)[:50]}'. Skipping.", verbosity_level=0)
			return 
		
		if not text_to_process: 
			self._log_print(f"_produce_one_sample: No text to process after attempting to get data. Raw item was: {str(raw_data_item)[:200]}. Returning.", verbosity_level=0)
			return 
		
		try:
			if not self.objectives_info: return 
			obj_idx = self.rng.choice(len(self.objectives_info), p=self.og_probs)
			obj_info = self.objectives_info[obj_idx]
		except ValueError as e_choice: 
			self._log_print(f"_produce_one_sample: Error choosing objective (probs: {self.og_probs}): {e_choice}", verbosity_level=0)
			if self.objectives_info: 
				self.og_probs = np.float32([o.get("prob", 1.0) for o in self.objectives_info])
				s = self.og_probs.sum()
				if s > 0: self.og_probs /= s
				else: self.og_probs = np.float32([1.0/len(self.og_probs)]*len(self.og_probs)) if len(self.og_probs) > 0 else np.array([])
			return 

		objective_name = obj_info["name"]
		cfg_overrides = obj_info.get("config_overrides", {}).copy() 
		cfg_overrides["remaining_space"] = self.packer.get_remaining_space(max_or_min="max") 
		cfg_overrides["tokenizer"] = self.tokenizer 
		cfg_overrides["rng_seed"] = self.rng.integers(0, 2**32 -1) 
		
		try: objective_instance = self._get_objective_instance(objective_name, cfg_overrides)
		except Exception as e_get_obj: 
			self._log_print(f"_produce_one_sample: Error getting objective instance '{objective_name}': {e_get_obj}", verbosity_level=0)
			self.leftover_text = text_to_process; return

		original_text_for_this_attempt = text_to_process
		result = objective_instance(text_to_process) 

		if result.get("status") == "not_enough_space":
			current_internal_packer = self.packer.packers[0] if self.packer.packers else None
			if current_internal_packer and current_internal_packer.data_index > 0:
				packed_data_batch = self.packer.pop(peek=False) 
				if packed_data_batch:
					single_sample_data = {key: value[0] for key, value in packed_data_batch.items() if hasattr(value, 'ndim') and value.ndim > 0 and hasattr(value, 'shape') and value.shape[0] == 1}
					if single_sample_data.get("input_ids", np.array([])).any():
						item_to_send = {"worker_id": self.worker_id, "packed_data": single_sample_data, "objective_name": "flushed_due_to_no_space"}
						try:
							self.sample_queue.put(item_to_send, timeout=0.1)
						except queue.Full: self._log_print(f"WARNING: sample_queue full when flushing batch.", verbosity_level=1)
						except Exception as e_put: self._log_print(f"ERROR putting flushed batch: {e_put}", verbosity_level=0); self.should_stop = True; return
			self.leftover_text = original_text_for_this_attempt 
			return 
		
		self.leftover_text = result.get("unused_input_string", "") 
		if result.get("status") != "ok": return

		input_ids = result.get("input_ids"); label_ids = result.get("label_ids")
		input_ids_len = len(input_ids) if input_ids is not None and hasattr(input_ids, '__len__') else 0
		
		if not input_ids_len: return
		if label_ids is None : label_ids = [] 
		
		try: 
			input_ids_np = np.array(input_ids, dtype=np.int32) if not isinstance(input_ids, np.ndarray) else input_ids.astype(np.int32)
			label_ids_np = np.array(label_ids, dtype=np.int32) if not isinstance(label_ids, np.ndarray) else label_ids.astype(np.int32)
			if label_ids_np.size == 0 and objective_name != "Infilling (fallback, unmasked)": return
			packer_status = self.packer.add(input_ids_np, label_ids_np) 
		except ValueError as e_packer_add: 
			self._log_print(f"ValueError from packer.add for '{objective_name}': {e_packer_add}", verbosity_level=0) 
			return 

		if packer_status == "ready": 
			packed_data_batch = self.packer.pop(peek=False) 
			if packed_data_batch:
				single_sample_data = {key: value[0] for key, value in packed_data_batch.items() if hasattr(value, 'ndim') and value.ndim > 0 and hasattr(value, 'shape') and value.shape[0] == 1}
				if single_sample_data.get("input_ids", np.array([])).any(): 
					item_to_send = {"worker_id": self.worker_id, "packed_data": single_sample_data, "objective_name": objective_name}
					while True:
						try: 
							self.sample_queue.put(item_to_send, timeout=0.1) 
							break
						# except queue.Full: 
						# 	self._log_print(f"WARNING: sample_queue full for '{objective_name}'. Item might be dropped.", verbosity_level=1)
						except Exception as e_put: 
							self._log_print(f"ERROR putting to sample_queue: {e_put}", verbosity_level=0); self.should_stop = True 

	def run(self, profile: bool = False):
		# self._log_print(f"run() method STARTED.", verbosity_level=1) 
		self.rng = np.random.default_rng(self.base_rng_seed + self.worker_id + os.getpid() + int(time.time()*1000) % 100000)

		profiler = None
		if profile:
			profiler = cProfile.Profile()
			profiler.enable() # Profile the entire run method

		self.initialize_data_iterator()
		if self.should_stop: 
			self._log_print(f"Stopping after initialize_data_iterator due to self.should_stop=True.", verbosity_level=0)
			if profiler:
				profiler.disable() # Disable before early exit
			try: self.close() 
			except Exception as e_cl: self._log_print(f"Exception in self.close() during early exit: {e_cl}", verbosity_level=0)
			try: self.sample_queue.put(None, timeout=0.1) 
			except Exception as e: self._log_print(f"Error putting None (from init fail): {e}", verbosity_level=0)
			if profiler:
				self._print_profile_stats(profiler)
			return

		last_instruction_check_time = time.time()
		samples_processed_in_this_run = 0 
		# MAX_SAMPLES_FOR_PROFILE logic removed; profiler runs for the whole duration.

		try: 
			while not self.should_stop:
				if time.time() - last_instruction_check_time > 0.05: 
					instruction_status = self._try_get_instructions()
					if instruction_status is False: 
						self.should_stop = True; break 
					last_instruction_check_time = time.time()
				
				if not self.objectives_info: 
					time.sleep(0.01); continue
				
				self._produce_one_sample()
				samples_processed_in_this_run +=1 

				if hasattr(self.sample_queue, 'maxsize') and self.sample_queue.qsize() > (self.sample_queue.maxsize * 0.9):
					time.sleep(0.005) 
		
		except KeyboardInterrupt: 
			self._log_print(f"KeyboardInterrupt caught in Worker.run() loop. Setting should_stop=True.", verbosity_level=0)
			self.should_stop = True 
		except Exception as e: 
			self._log_print(f"Unhandled EXCEPTION in Worker.run() loop: {e}", verbosity_level=0)
			traceback.print_exc(file=sys.stdout); sys.stdout.flush(); self.should_stop = True
		finally: 
			# self._log_print(f"Entering FINALLY block of run(). should_stop: {self.should_stop}", verbosity_level=1)
			if profiler:
				profiler.disable() # Disable profiler at the start of finally
			# self._log_print(f"Calling self.close() from finally block.", verbosity_level=1)
			try:
				self.close() 
				# self._log_print(f"self.close() completed in finally.", verbosity_level=1)
			except Exception as e_close:
				self._log_print(f"Exception during self.close() in finally: {e_close}", verbosity_level=0)
				traceback.print_exc(file=sys.stdout); sys.stdout.flush()
			# self._log_print(f"Attempting to put None sentinel on sample_queue in finally.", verbosity_level=1)
			try:
				self.sample_queue.put(None, timeout=1.0) 
				# self._log_print(f"Successfully put None sentinel on sample_queue in finally.", verbosity_level=1)
			except queue.Full:
				self._log_print(f"sample_queue full when putting None sentinel in finally. This is a problem if consumer is stuck.", verbosity_level=0)
			except Exception as e_sq:
				self._log_print(f"Error putting None sentinel in finally: {e_sq}", verbosity_level=0)
				traceback.print_exc(file=sys.stdout); sys.stdout.flush()
			# self._log_print(f"Printing final profile stats. Processed ~{samples_processed_in_this_run} samples during this run.", verbosity_level=1)
			if profiler:
				self._print_profile_stats(profiler) # Print stats once at the very end

			# self._log_print(f"run() finished.", verbosity_level=1)

	def _print_profile_stats(self, profiler):
		self._log_print(f"\n--- Profiling Results for Worker {self.worker_id} ---", verbosity_level=0) 
		s = io.StringIO()
		# Check if profiler has stats before trying to use pstats
		raw_stats = profiler.getstats()
		if raw_stats:
			# self._log_print(f"Raw stats collected: {len(raw_stats)} entries.", verbosity_level=0)
			try:
				ps = pstats.Stats(profiler, stream=s) # Pass the profiler object
				ps.sort_stats('cumtime')
				ps.print_stats(20) # Print top 20
				output = s.getvalue()
				if output.strip():
					self._log_print(output, verbosity_level=0)
				else:
					self._log_print("pstats.print_stats() produced no output, though raw stats exist.", verbosity_level=0)
			except Exception as e_pstats:
				self._log_print(f"Error creating/printing pstats: {e_pstats}", verbosity_level=0)
				# self._log_print(f"First few raw stats entries: {raw_stats[:5]}", verbosity_level=0) 
		else:
			self._log_print("No profiling stats collected by profiler.getstats().", verbosity_level=0)


if __name__ == "__main__":
	print("--- Testing _default_text_source (from worker.py) ---", flush=True)
	try:
		gen = _default_text_source(worker_id=0, total_workers=1, split="train", rng_seed=42)
		for i in range(5): 
			item = next(gen)
			# print(f"Item {i}: {str(item)[:100]}...", flush=True) 
			if isinstance(item, dict) and "text" in item and item["text"]:
				# print(f"  Text: {item['text'][:50]}...", flush=True)  
				pass
			else:
				print("  No valid text found in item.", flush=True)
	except Exception as e_test_default:
		print(f"Error testing _default_text_source: {e_test_default}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()