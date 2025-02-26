# worker.py
"""
Worker module with extra debug prints and a fix for validation workers.
If the worker is for validation (split=="validation") then we force infinite_loop=True
so that the worker does not close after one pass over the dataset.
"""

import os
import time
import queue
import numpy as np
import traceback
import multiprocessing as mp
from typing import Any, Callable, Union, Iterable
from datasets import load_dataset
from functools import partial
from birdie_rl.pipeline.packer_batcher import Batcher
from birdie_rl.load_objective import load_objective


def _default_text_source(worker_id: int, total_workers: int):
	"""
	Fallback text source that shreds the TinyStories dataset by worker shard.
	"""
	print(f"[Worker _default_text_source] Loading TinyStories (worker {worker_id}/{total_workers}).")
	ds = load_dataset("roneneldan/TinyStories", split="train")
	ds = ds.shard(num_shards=total_workers, index=worker_id)
	while True:
		for record in ds:
			yield record["text"]


class Worker:
	"""
	A Worker process that:
	  1) Reads objective instructions from tasks_queue.
	  2) Uses a data source (generator/iterable) to produce text.
	  3) Transforms text with load_objective(...) => (input_ids, label_ids).
	  4) Adds them to a Batcher (from packer_batcher.py).
	  5) If batch is ready/full, pops and sends to results_queue.
	  6) On sentinel (None in tasks_queue) or finished epoch (for non-infinite loops), the worker's close() function is called. The worker will attempt to flush any partial batch it may have (timeout=0.5), then it will exit.
	"""

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
		batch_size: int = 8,
		tokenizer=None,
		split: str = None,
		text_grabber_fn: Callable[[Any], str] = None,
		infinite_loop: bool = True,
		start_generating_paradigm: str = "\n<|assistant|>\n",
	):
		self.worker_id = worker_id
		self.total_workers = total_workers
		self.tasks_queue = tasks_queue
		self.results_queue = results_queue
		self.sample_queue = sample_queue

		self.sequence_length = sequence_length
		self.min_seq_len_for_packing = min_seq_len_for_packing
		self.batch_size = batch_size
		self.infinite_loop = infinite_loop
		self.split = split

		if text_grabber_fn is None:
			# Default text grabber expects a dataset entry dict with a "text" key to be tokenized.
			def text_grabber_fn(x):
				try:
					return x["text"]
				except Exception as e:
					print(f"  FAILED:  Could not grab the key 'text' from the dataset entry dict: {x}")

		self.text_grabber_fn = text_grabber_fn

		# If no data_generator is provided, use our fallback.
		if data_generator is None:
			data_generator = partial(_default_text_source, worker_id=worker_id, total_workers=total_workers)
		self.data_generator_fn = data_generator

		self.data_iter = None
		self.dataset_reset_counter = 0

		# Create a random number generator seeded by worker info.
		self.rng = np.random.default_rng(self.worker_id * 100 + total_workers * 10_000)

		self.tokenizer = tokenizer
		if self.tokenizer is None:
			raise ValueError(f"[Worker {self.worker_id}] ERROR: A tokenizer is required for objective transforms.")

		# Create the Batcher from packer_batcher.py
		self.batcher = Batcher(config={
			"tokenizer": self.tokenizer,
			"batch_size": self.batch_size,
			"minimum_sequence_length": self.min_seq_len_for_packing,
			"sequence_length": self.sequence_length,
			"start_generating_paradigm": start_generating_paradigm,
		})

		self.leftover_text = ""
		self.leftover_ids = np.array([], dtype=np.int32)

		self.objectives_info = []
		self.og_probs = np.array([], dtype=np.float32)

		self.print(f"[Worker {self.worker_id} (split: {self.split})] Created with batch_size={batch_size}, seq_length={sequence_length}")

	def print(self, *args, verbosity_level=0, min_verbosity_level=1, **kwargs):
		"""
		This method helps us easily specifically silence worker debug info.
		"""
		# print(*args, **kwargs); return; ## Uncomment this to enable all worker debug printing
		if min_verbosity_level <= verbosity_level:
			print(*args, **kwargs) ## Uncomment this to enable worker debug printing
		pass

	def close(self):
		"""
			Flushes leftover partial batches (if any), and then exits the process.
		"""
		pass
		self.print(f"[Worker {self.worker_id} (split: {self.split})] close() called => Attempting to flush partial batch if any.")
		current_status = self.batcher.is_ready()
		self.print(f"[Worker {self.worker_id} (split: {self.split})] batcher.is_ready() => {current_status}")

		if current_status == "ready":
			final_data = self.batcher.pop(peek=False)
			if final_data is not None:
				self.print(f"[Worker {self.worker_id} (split: {self.split})] Popped leftover partial batch. Sending to results_queue.", verbosity_level=0)
				item = {
					"worker_id": self.worker_id,
					"batch_items": final_data,
					"objective_name": "partial_leftover",
				}
				try:
					if isinstance(final_data, str):
						raise ValueError(f"[Worker {self.worker_id} (split: {self.split})] ERROR: final_data is a string: {final_data}")
					self.results_queue.put(item, timeout=5,)
				except Exception as e:
					self.print(f"  worker.close() => results_queue.put() failed. Exiting via os._exit(1)! e: {e}")

		else:
			self.print(f"[Worker {self.worker_id} (split: {self.split})] No leftover partial batch to flush (or not ready).")
			pass

		self.print(f"[Worker {self.worker_id} (split: {self.split})] Exiting now via os._exit(1).")
		os._exit(1)

	def initialize_data_iterator(self):
		"""
		(Re)initialize the data iterator.
		For non-infinite workers (e.g. training when set so), after one epoch the worker would normally close.
		But for validation (or if infinite_loop is forced), the iterator will be reinitialized continuously.
		"""
		self.dataset_reset_counter += 1
		self.print(f"[Worker {self.worker_id} (split: {self.split})] (Re)initializing data iterator, dataset_reset={self.dataset_reset_counter}, infinite_loop={self.infinite_loop}")

		# For non-infinite workers (e.g. training) we may close after one pass.
		if (not self.infinite_loop) and (self.dataset_reset_counter > 1):
			self.print(f"[Worker {self.worker_id} (split: {self.split})] Not infinite_loop => finishing. Will call close().")
			self.close()
			return

		def _inf_loop(source):
			while True:
				if callable(source):
					gen = source()
					if not hasattr(gen, "__iter__"):
						gen = iter([gen])
				else:
					gen = iter(source)
				for y in gen:
					yield y
				if not self.infinite_loop:
					self.print(f"[Worker {self.worker_id} (split: {self.split})] _inf_loop => single pass done => break.")
					break

		self.data_iter = _inf_loop(self.data_generator_fn)

	def _try_get_instructions(self):
		"""
		Non-blocking attempt to get objective instructions from tasks_queue.
		Returns:
		  - False if a sentinel (None) is received (worker should exit),
		  - None if no new instructions,
		  - True if instructions were updated.
		"""
		try:
			data = self.tasks_queue.get(timeout=0.1)
			self.print(f"[Worker {self.worker_id} (split: {self.split})] _try_get_instructions => got data: {data}")
		except queue.Empty:
			return None
		try:
			self.tasks_queue.put(data)
		except:
			pass
		if data is None:
			return False

		objs = data.get("objectives", [])
		if not objs:
			self.print(f"[Worker {self.worker_id} (split: {self.split})] _try_get_instructions => No objectives in dict => continuing.")
			self.objectives_info = []
			return True

		self.objectives_info = objs
		arr = np.float32([obj.get("prob", 1.0) for obj in objs])
		s = arr.sum()
		if s > 0:
			arr /= s
		else:
			arr = np.float32([1.0 / len(objs)] * len(objs))
		self.og_probs = arr
		self.print(f"[Worker {self.worker_id} (split: {self.split})] => parsed objectives: {objs} => normalized probs: {arr.tolist()}")
		return True

	def _produce_one_sample(self):
		"""
		Produce one sample: get text (using leftover or by concatenating chunks),
		choose an objective, run transformation, add result to batcher, and if batch is full then pop and send.
		"""
		chunk_size = 256
		# Use leftover if long enough
		if len(self.leftover_text) >= chunk_size:
			text_sample = self.leftover_text
			self.leftover_text = ""
			self.leftover_ids = np.array([], dtype=np.int32)
			self.print(f"[Worker {self.worker_id} (split: {self.split})] Using leftover_text since len(leftover_text) >= {chunk_size}")
		else:
			text_sample = ""
			while len(text_sample) < chunk_size:
				try:
					next_item = next(self.data_iter)
					txt = self.text_grabber_fn(next_item)
					text_sample = str(txt)
					self.print(f"[Worker {self.worker_id} (split: {self.split})] got next text chunk of len={len(text_sample)}")
				except StopIteration:
					self.print(f"[Worker {self.worker_id} (split: {self.split})] data_iter exhausted => re-init or close if not infinite.")
					self.initialize_data_iterator()
					return
				except Exception as e:
					self.print(f"[Worker {self.worker_id} (split: {self.split})] EXCEPTION in _produce_one_sample: {e}")
					traceback.print_exc()
					return

		if len(self.og_probs) == 0:
			self.print(f"[Worker {self.worker_id} (split: {self.split})] No objectives => skip producing sample.")
			return

		idx = self.rng.choice(len(self.objectives_info), p=self.og_probs)
		obj_info = self.objectives_info[idx]
		objective_name = obj_info["name"]
		cfg_overrides = obj_info.get("config_overrides", {})
		if "remaining_space" not in cfg_overrides:
			cfg_overrides["remaining_space"] = self.batcher.get_remaining_space()
		cfg_overrides["tokenizer"] = self.tokenizer

		self.print(f"[Worker {self.worker_id} (split: {self.split})] Using objective={objective_name} with overrides={cfg_overrides} for text len={len(text_sample)}")

		try:
			objective = load_objective(objective_name, cfg_overrides)
		except Exception as e:
			self.print(f"[Worker {self.worker_id} (split: {self.split})] load_objective failed: {e}")
			traceback.print_exc()
			return
		
		result = objective(text_sample)
		self.print(f"")
		self.print(f"*" * 60,)
		self.print(f"[Worker {self.worker_id} (split: {self.split})] objective('{objective_name}') => result.status={result.get('status')} => leftover len={len(result.get('unused_input_string',''))}")
		for result_idx, (key, value) in enumerate(result.items()):
			pass
			self.print(f"  result[{key}]: {value}")
			
		self.print(f"*" * 60,)

		if result.get("status") != "ok":
			leftover_str = result.get("unused_input_string", "")
			self.leftover_text = leftover_str
			self.leftover_ids = result.get("unused_input_ids", np.array([], dtype=np.int32))
			self.print(f"[Worker {self.worker_id} (split: {self.split})] transform not OK => leftover_text len={len(leftover_str)} => skipping sample.")
			return

		input_ids = result["input_ids"]
		label_ids = result["label_ids"]
		self.leftover_text = result.get("unused_input_string", "")
		self.leftover_ids = result.get("unused_input_ids", np.array([], dtype=np.int32))
		self.print(f"[Worker {self.worker_id} (split: {self.split})] objective done => input_ids.len={len(input_ids)}, label_ids.len={len(label_ids)}, leftover_text len={len(self.leftover_text)}")

		status = self.batcher.add(input_ids, label_ids)
		self.print(f"[Worker {self.worker_id} (split: {self.split})] Batcher add => status='{status}'")

		if status in ["ready", "full"]:
			self.print(f"[Worker {self.worker_id} (split: {self.split})] Batcher full/ready => pop sub-batch and send to results_queue")
			batch_data = self.batcher.pop(peek=False)
			if batch_data is not None:
				item = {
					"worker_id": self.worker_id,
					"batch_items": batch_data,
					"objective_name": objective_name,
				}

				if isinstance(batch_data, str):
					raise ValueError(f"[Worker {self.worker_id} (split: {self.split})] ERROR: batch_data is a string: {batch_data}")
				
				while True:
					try:
						self.results_queue.put(item, timeout=0.5)
						break
					except queue.Full:
						self.print(f"[Worker {self.worker_id} (split: {self.split})] results_queue.put() failed => retrying...")
						continue

				self.print(f"[Worker {self.worker_id} (split: {self.split})] Sub-batch sent to results_queue. results_queue.qsize()={self.results_queue.qsize()}, batcher.get_remaining_space()={self.batcher.get_remaining_space()}")

			else:
				self.print(f"[Worker {self.worker_id} (split: {self.split})] Batcher.pop returned None (unexpected).")
				pass

	def run(self):
		"""
		Main worker loop:
		  1. (Re)initialize data iterator.
		  2. Check for instructions from tasks_queue.
		  3. Produce a sample.
		  4. Re-check instructions and either update objectives or exit on sentinel.
		"""
		self.print(f"[Worker {self.worker_id} (split: {self.split})] run() starting, calling initialize_data_iterator() now.")
		self.initialize_data_iterator()

		while True:
			if not self.objectives_info:
				self.print(f"[Worker {self.worker_id} (split: {self.split})] No objectives => _try_get_instructions()")
				got_instructions = self._try_get_instructions()
				if got_instructions is False:
					self.print(f"[Worker {self.worker_id} (split: {self.split})] tasks_queue gave None (sentinel) => close()")
					self.close()
					return
				elif got_instructions is None:
					self.print(f"[Worker {self.worker_id} (split: {self.split})] tasks_queue empty, sleeping 0.2 sec.")
					time.sleep(0.2)
					continue
				else:
					pass
					self.print(f"[Worker {self.worker_id} (split: {self.split})] New objectives: {self.objectives_info}")

			self._produce_one_sample()

			new_inst = self._try_get_instructions()
			if new_inst is False:
				self.print(f"[Worker {self.worker_id} (split: {self.split})] tasks_queue gave None (sentinel) => close()")
				self.close()
				return
			elif new_inst is True:
				pass
				self.print(f"[Worker {self.worker_id} (split: {self.split})] Objectives updated: {self.objectives_info}")
			# Else: no new instructions; continue.
