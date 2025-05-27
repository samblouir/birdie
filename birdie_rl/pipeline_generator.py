# pipeline_generator.py
import os
import sys
import threading
import queue as local_queue
import multiprocessing as mp
import queue # For mp.Queue.Empty exception
import time
from typing import List, Callable, Any
import traceback

from birdie_rl.pipeline.main_controller import MainController
from birdie_rl.pipeline.worker import Worker # Worker is already updated
import torch
import numpy as np
from functools import partial


def datagen_thread_fn(
	max_batches: int,
	results_q: mp.Queue, # This queue now receives fully formed batches from samples_to_batch_fn
	output_q: local_queue.Queue, # This is the final output queue for the generator
	datagen_stop_event: threading.Event,
	num_batcher_processes: int, # Number of samples_to_batch_fn processes
	accelerator=None,
	move_to_gpu_fn=None,
):
	print_fn = print
	pid = os.getpid()
	thread_id = threading.get_ident()
	# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Started. Expecting sentinels from {num_batcher_processes} batchers.", flush=True)

	batches_received_from_results_q = 0
	sentinels_received_from_batchers = 0
	try:
		while not datagen_stop_event.is_set():
			if max_batches != -1 and batches_received_from_results_q >= max_batches:
				if not datagen_stop_event.is_set(): datagen_stop_event.set()
				break

			try:
				# Item from results_q is now expected to be a fully formed batch
				# e.g., {"batch_items": stacked_batch_dict}
				batch_item_from_results_q = results_q.get(timeout=0.1)
			except queue.Empty:
				if datagen_stop_event.is_set(): break
				if num_batcher_processes > 0 and sentinels_received_from_batchers >= num_batcher_processes:
					if not datagen_stop_event.is_set(): datagen_stop_event.set()
				continue

			if datagen_stop_event.is_set(): break

			if batch_item_from_results_q is None:
				sentinels_received_from_batchers += 1
				if num_batcher_processes > 0 and sentinels_received_from_batchers >= num_batcher_processes:
					if not datagen_stop_event.is_set(): datagen_stop_event.set()
					break
				continue

			# Extract the actual batch data (the stacked dictionary)
			final_batch_to_yield = batch_item_from_results_q.get("batch_items")
			if final_batch_to_yield is None:
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Received item with no 'batch_items'. Skipping.", flush=True)
				continue

			if move_to_gpu_fn is not None and isinstance(final_batch_to_yield, dict):
				final_batch_to_yield = move_to_gpu_fn(final_batch_to_yield)

			try:
				output_q.put(final_batch_to_yield, timeout=0.5)
				batches_received_from_results_q += 1
			except local_queue.Full:
				if datagen_stop_event.is_set(): break
				time.sleep(0.01)

	except Exception as e:
		print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] EXCEPTION: {e}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
	finally:
		if not datagen_stop_event.is_set(): datagen_stop_event.set()
		try:
			output_q.put(None, timeout=0.2)
		except local_queue.Full:
			print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] output_q full putting sentinel.", flush=True)
		except Exception as e_f:
			print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Error putting None: {e_f}", flush=True)


def samples_to_batch_fn_wrapper(*args):
	pid = os.getpid()
	try:
		samples_to_batch_fn(*args)
	except Exception as e_wrapper:
		print(f"!!!!!!!!!! [samples_to_batch_fn_wrapper PID {pid}] UNCAUGHT EXCEPTION: {e_wrapper} !!!!!!!!!!", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()


def samples_to_batch_fn(
	sample_queue: mp.Queue, # Receives items from Workers: {"worker_id": ..., "stacked_batch_data": ...}
	results_queue: mp.Queue, # Sends to datagen_thread_fn: {"batch_items": stacked_batch_dict}
	# batch_size: int, # This parameter is no longer directly used for batching here, as workers send full batches
	stop_event: mp.Event,
	num_worker_processes_for_this_controller: int
):
	print_fn = print
	pid = os.getpid()
	# print_fn(f"!!!!!!!!!! [samples_to_batch_fn (actual) PID {pid}] ALIVE. Expecting sentinels from {num_worker_processes_for_this_controller} workers.", flush=True)

	sentinels_received_from_workers = 0
	try:
		while True:
			if stop_event.is_set(): break

			all_workers_confirmed_done = (num_worker_processes_for_this_controller > 0 and \
										  sentinels_received_from_workers >= num_worker_processes_for_this_controller)
			if all_workers_confirmed_done:
				if not stop_event.is_set(): stop_event.set()
				break

			try:
				# Item from worker is already a batch
				# e.g. {"worker_id": ..., "stacked_batch_data": ...}
				worker_output_item = sample_queue.get(timeout=0.1)
			except queue.Empty:
				if stop_event.is_set(): break
				continue

			if stop_event.is_set(): break

			if worker_output_item is None:
				sentinels_received_from_workers +=1
				continue

			if not isinstance(worker_output_item, dict) or "stacked_batch_data" not in worker_output_item:
				# print_fn(f"[samples_to_batch_fn PID {pid}] Invalid item from worker, skipping: {str(worker_output_item)[:100]}", flush=True)
				continue

			# The item from the worker already contains the stacked batch data
			stacked_batch_dict = worker_output_item["stacked_batch_data"]

			# Prepare item for results_queue (which datagen_thread_fn consumes)
			item_for_results_q = {"batch_items": stacked_batch_dict}
			try:
				results_queue.put(item_for_results_q, timeout=0.5)
			except queue.Full:
				if stop_event.is_set():
					break
				time.sleep(0.01)

	except KeyboardInterrupt:
		print_fn(f"[samples_to_batch_fn PID {pid}] KeyboardInterrupt. Setting stop_event.", flush=True)
		if not stop_event.is_set(): stop_event.set()
	except Exception as e:
		print_fn(f"[samples_to_batch_fn PID {pid}] EXCEPTION: {e}", flush=True)
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
		if not stop_event.is_set(): stop_event.set()
	finally:
		if not stop_event.is_set(): stop_event.set()
		try:
			results_queue.put(None, timeout=0.2) # Sentinel for datagen_thread_fn
		except queue.Full:
			print_fn(f"[samples_to_batch_fn PID {pid}] results_queue full putting sentinel.", flush=True)
		except Exception as e_rs:
			print_fn(f"[samples_to_batch_fn PID {pid}] Error putting None to results_queue: {e_rs}", flush=True)


def pipeline_data_generator(
	max_batches=-1,
	batch_size=8, # This batch_size is now passed to the Worker to form batches
	sequence_length=4096,
	num_workers = 16,
	objectives_config=None,
	accelerator=None,
	move_to_gpu_fn=None,
	data_generator: Callable = None,
	data_generator_fn_kwarg_overrides={},
	infinite_loop=True,
	split=None,
	config={}, # Main configuration dictionary
):
	print_fn = print
	parent_pid = os.getpid()

	if data_generator is None: raise ValueError("`data_generator` function must be provided.")
	if not callable(data_generator): raise ValueError("`data_generator` must be a callable function.")

	num_accelerator_processes = accelerator.num_processes if accelerator else 1
	num_workers_total_across_accelerators = num_workers * num_accelerator_processes
	num_local_workers = num_workers

	tasks_q = mp.Queue(maxsize=num_local_workers * 2)
	sample_q = mp.Queue(maxsize=num_local_workers * 2 + 5) # Workers put full batches here
	final_batches_q = mp.Queue(maxsize=32 + num_accelerator_processes + 5) # samples_to_batch_fn puts final items here
	output_q_for_generator = local_queue.Queue(maxsize=16)

	if objectives_config is None:
		objectives_config = [{"name": "next_token_prediction", "prob": 1.0}]

	worker_processes = []
	current_accelerator_worker_offset = (num_local_workers * accelerator.process_index) if accelerator else 0

	for local_idx in range(num_local_workers):
		global_worker_id = current_accelerator_worker_offset + local_idx
		data_gen_kwargs_for_worker = dict(split=split, worker_id=global_worker_id,
										  num_workers=num_workers_total_across_accelerators,
										  rng_seed=config.get("seed", int(time.time())) + global_worker_id)
		data_gen_kwargs_for_worker.update(data_generator_fn_kwarg_overrides)
		this_worker_data_gen_fn = partial(data_generator, **data_gen_kwargs_for_worker)

		# Ensure the main 'config' passed to Worker includes 'batch_size' for target_batch_size
		worker_config = config.copy() # Start with a copy of the main config
		worker_config['batch_size'] = batch_size # Explicitly set the target batch size for this worker

		worker_instance_for_process = Worker(
			worker_id=global_worker_id,
			total_workers=num_workers_total_across_accelerators,
			tasks_queue=tasks_q,
			results_queue=None,
			sample_queue=sample_q,
			sequence_length=sequence_length,
			min_seq_len_for_packing=config.get("min_seq_len_for_packing", 64),
			data_generator=this_worker_data_gen_fn,
			infinite_loop=infinite_loop,
			split=split,
			tokenizer=config['tokenizer'],
			text_grabber_fn=config.get("text_grabber_fn", None),
			start_generating_id=config.get("start_generating_id", 2),
			latent_token_id=config.get("latent_token_id", 1),
			max_samples_per_packer=config.get("max_samples_per_packer", float('inf')),
			rng_seed=config.get("seed", 0),
			config=worker_config # Pass the potentially modified config with target batch_size
		)
		p = mp.Process(target=worker_instance_for_process.run, args=(config.get("profile_workers", False),), daemon=True)
		worker_processes.append(p)

	num_batcher_processes = 1
	batcher_processes = []
	batcher_stop_event = mp.Event()

	for _ in range(num_batcher_processes):
		p = mp.Process(
			target=samples_to_batch_fn_wrapper,
			args=(sample_q, final_batches_q, # batch_size argument removed from samples_to_batch_fn
				  batcher_stop_event, len(worker_processes)),
			daemon=True
		)
		batcher_processes.append(p)

	main_ctrl = MainController(tasks_queue=tasks_q, sample_queue=sample_q,
							   objectives_config=objectives_config, worker_processes=worker_processes,
							   batcher_processes=batcher_processes, batcher_stop_event=batcher_stop_event,
							   num_workers_total=len(worker_processes))
	main_ctrl.run()

	datagen_stop_event = threading.Event()
	datagen_thread = threading.Thread(
		target=datagen_thread_fn,
		args=(max_batches, final_batches_q, output_q_for_generator,
			  datagen_stop_event, num_batcher_processes, accelerator, move_to_gpu_fn),
		daemon=True
	)

	for p in worker_processes: p.start()
	for p in batcher_processes: p.start()
	datagen_thread.start()

	def _final_generator():
		# print_gen_final = print
		shutdown_initiated_by_generator = False
		try:
			while True:
				try: batch = output_q_for_generator.get(timeout=1.0)
				except local_queue.Empty:
					if datagen_stop_event.is_set(): shutdown_initiated_by_generator = True; break
					if datagen_thread and not datagen_thread.is_alive(): shutdown_initiated_by_generator = True; break
					continue
				if datagen_stop_event.is_set() and batch is None: shutdown_initiated_by_generator = True; break
				if batch is None: shutdown_initiated_by_generator = True; break
				yield batch
				output_q_for_generator.task_done()
		except Exception as e:
			# print_gen_final(f"Exception in _final_generator: {e}", flush=True)
			traceback.print_exc(file=sys.stdout); sys.stdout.flush()
			shutdown_initiated_by_generator = True
		finally:
			if shutdown_initiated_by_generator:
				if datagen_stop_event and not datagen_stop_event.is_set(): datagen_stop_event.set()
				if batcher_stop_event and not batcher_stop_event.is_set(): batcher_stop_event.set()

	return main_ctrl, _final_generator(), datagen_thread, batcher_stop_event, datagen_stop_event
