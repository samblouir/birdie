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
from birdie_rl.pipeline.worker import Worker
# from datasets import load_dataset 
import torch 
import numpy as np 
from functools import partial


def datagen_thread_fn(
	max_batches: int,
	results_q: mp.Queue, 
	output_q: local_queue.Queue, 
	datagen_stop_event: threading.Event, 
	num_batcher_processes: int, 
	accelerator=None, 
	move_to_gpu_fn=None,
):
	print_fn = print 
	pid = os.getpid()
	thread_id = threading.get_ident()
	# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Started. Expecting sentinels from {num_batcher_processes} batchers.", flush=True)

	batches_received = 0
	sentinels_received_from_batchers = 0
	try:
		while not datagen_stop_event.is_set():
			if max_batches != -1 and batches_received >= max_batches:
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Reached max_batches ({max_batches}). Setting stop event.", flush=True)
				if not datagen_stop_event.is_set(): datagen_stop_event.set() 
				break

			try:
				batch_dict_from_batcher_proc = results_q.get(timeout=0.1) 
			except queue.Empty: 
				if datagen_stop_event.is_set(): 
					# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Stop event IS SET during results_q.get timeout. Breaking.", flush=True)
					break 
				if num_batcher_processes > 0 and sentinels_received_from_batchers >= num_batcher_processes: 
					# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] All batcher sentinels ({sentinels_received_from_batchers}/{num_batcher_processes}) received & results_q timed out. Setting stop event.", flush=True)
					if not datagen_stop_event.is_set(): datagen_stop_event.set()
				continue 
			
			if datagen_stop_event.is_set(): 
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Stop event set after results_q.get(). Breaking.", flush=True)
				break 

			if batch_dict_from_batcher_proc is None: 
				sentinels_received_from_batchers += 1
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Received None sentinel from results_q ({sentinels_received_from_batchers}/{num_batcher_processes}).", flush=True)
				if num_batcher_processes > 0 and sentinels_received_from_batchers >= num_batcher_processes:
					# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] All batcher sentinels received. Setting stop event and breaking.", flush=True)
					if not datagen_stop_event.is_set(): datagen_stop_event.set() 
					break 
				continue 

			final_batch_data = batch_dict_from_batcher_proc.get("batch_items") 
			if final_batch_data is None:
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Received item with no 'batch_items'. Skipping.", flush=True)
				continue

			if move_to_gpu_fn is not None and isinstance(final_batch_data, dict):
				final_batch_data = move_to_gpu_fn(final_batch_data)
			
			try:
				output_q.put(final_batch_data, timeout=0.5) 
				batches_received += 1
			except local_queue.Full: 
				# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] output_q full. Checking stop event.", flush=True)
				if datagen_stop_event.is_set(): break 
				time.sleep(0.01) 

	except Exception as e:
		print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] EXCEPTION: {e}", flush=True) # Keep error logs
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
	finally:
		# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Entering finally. stop_event: {datagen_stop_event.is_set()}. Sentinels received: {sentinels_received_from_batchers}/{num_batcher_processes}", flush=True)
		if not datagen_stop_event.is_set(): 
			# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Setting stop_event in finally.", flush=True)
			datagen_stop_event.set() 
		
		# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Attempting to put None sentinel on output_q.", flush=True)
		try:
			output_q.put(None, timeout=0.2) 
			# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Successfully put None on output_q.", flush=True)
		except local_queue.Full: 
			print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] output_q full while trying to put sentinel in finally.", flush=True) # Keep important warnings
		except Exception as e_f:
			print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Error putting None in finally: {e_f}", flush=True)
		# print_fn(f"[datagen_thread_fn PID {pid} TID {thread_id}] Thread finished.", flush=True)


def samples_to_batch_fn_wrapper(*args):
	pid = os.getpid()
	# print(f"!!!!!!!!!! [samples_to_batch_fn_wrapper PID {pid}] PROCESS TARGET ENTERED !!!!!!!!!!", flush=True) # Keep this one for now
	try:
		samples_to_batch_fn(*args)
	except Exception as e_wrapper:
		print(f"!!!!!!!!!! [samples_to_batch_fn_wrapper PID {pid}] UNCAUGHT EXCEPTION IN TARGET FUNCTION: {e_wrapper} !!!!!!!!!!", flush=True) # Keep error logs
		traceback.print_exc(file=sys.stdout) 
		sys.stdout.flush() 
	# finally:
		# print(f"!!!!!!!!!! [samples_to_batch_fn_wrapper PID {pid}] PROCESS TARGET EXITED FINALLY (samples_to_batch_fn completed or errored) !!!!!!!!!!", flush=True)


def samples_to_batch_fn(
	sample_queue: mp.Queue, 
	results_queue: mp.Queue, 
	batch_size: int, 
	stop_event: mp.Event, 
	num_worker_processes_for_this_controller: int 
):
	print_fn = print 
	pid = os.getpid()
	# print_fn(f"!!!!!!!!!! [samples_to_batch_fn (actual) PID {pid}] ALIVE AND STARTED !!!!!!!!!! Target batch: {batch_size}. Expecting sentinels from {num_worker_processes_for_this_controller} workers.", flush=True)
	
	current_batch_list = []
	sentinels_received_from_workers = 0
	
	try: 
		while True: 
			if stop_event.is_set():
				# print_fn(f"[samples_to_batch_fn PID {pid}] stop_event IS SET at top of loop. Breaking.", flush=True)
				break

			all_workers_confirmed_done = (num_worker_processes_for_this_controller > 0 and \
			                             sentinels_received_from_workers >= num_worker_processes_for_this_controller)

			if all_workers_confirmed_done:
				# print_fn(f"[samples_to_batch_fn PID {pid}] All {num_worker_processes_for_this_controller} workers confirmed done. Setting stop_event and breaking from top of loop.", flush=True)
				if not stop_event.is_set(): stop_event.set() 
				break 

			try:
				packed_sample_item = sample_queue.get(timeout=0.1) 
			except queue.Empty: 
				if stop_event.is_set(): 
					# print_fn(f"[samples_to_batch_fn PID {pid}] stop_event set during sample_queue.get timeout. Breaking.", flush=True)
					break 
				continue 
			
			if stop_event.is_set(): 
				# print_fn(f"[samples_to_batch_fn PID {pid}] stop_event found set after sample_queue.get(). Breaking.", flush=True)
				break

			if packed_sample_item is None: 
				sentinels_received_from_workers +=1
				# print_fn(f"[samples_to_batch_fn PID {pid}] Received None sentinel from sample_queue ({sentinels_received_from_workers}/{num_worker_processes_for_this_controller}).", flush=True)
				continue 

			if not isinstance(packed_sample_item, dict) or "packed_data" not in packed_sample_item:
				# print_fn(f"[samples_to_batch_fn PID {pid}] Received invalid item, skipping: {str(packed_sample_item)[:100]}", flush=True)
				continue
				
			current_batch_list.append(packed_sample_item["packed_data"])

			if len(current_batch_list) >= batch_size:
				keys = current_batch_list[0].keys() 
				stacked_batch_dict = {}
				try:
					for key in keys:
						stacked_batch_dict[key] = np.stack([sample[key] for sample in current_batch_list])
				except ValueError as e:
					print(f"[samples_to_batch_fn PID {pid}] Error stacking batch: {e}.", flush=True) # Keep error log
					current_batch_list = [] 
					continue 
				
				item_for_results_q = {"batch_items": stacked_batch_dict} 
				try:
					results_queue.put(item_for_results_q, timeout=0.5) 
				except queue.Full: 
					# print_fn(f"[samples_to_batch_fn PID {pid}] results_queue full. Checking stop_event.", flush=True)
					if stop_event.is_set(): break
					time.sleep(0.01) 
				current_batch_list = []
	
	except KeyboardInterrupt: 
		print_fn(f"[samples_to_batch_fn PID {pid}] Caught KeyboardInterrupt. Ensuring stop_event is set.", flush=True) # Keep this
		if not stop_event.is_set(): stop_event.set()
	except Exception as e:
		print_fn(f"[samples_to_batch_fn PID {pid}] EXCEPTION in main loop: {e}", flush=True) # Keep error logs
		traceback.print_exc(file=sys.stdout); sys.stdout.flush()
		if not stop_event.is_set(): stop_event.set() 
	finally:
		# print_fn(f"[samples_to_batch_fn PID {pid}] Entering finally. stop_event: {stop_event.is_set()}, sentinels_from_workers: {sentinels_received_from_workers}/{num_worker_processes_for_this_controller}", flush=True)
		if not stop_event.is_set(): 
			# print_fn(f"[samples_to_batch_fn PID {pid}] Setting stop_event in finally because it wasn't set.", flush=True)
			stop_event.set() 
		
		if current_batch_list and len(current_batch_list) >= batch_size : 
			# print_fn(f"[samples_to_batch_fn PID {pid}] Flushing a final batch of {len(current_batch_list)} items in finally.", flush=True)
			try:
				keys = current_batch_list[0].keys()
				samples_to_stack = current_batch_list[:batch_size] 
				stacked_batch_dict = {key: np.stack([sample[key] for sample in samples_to_stack]) for key in keys}
				item_for_results_q = {"batch_items": stacked_batch_dict}
				results_queue.put(item_for_results_q, timeout=0.2) 
				# print_fn(f"[samples_to_batch_fn PID {pid}] Flushed one final batch to results_queue in finally.", flush=True)
			except Exception as e_f: 
				print_fn(f"[samples_to_batch_fn PID {pid}] Error during final batch flush: {e_f}", flush=True) # Keep error log
		
		# print_fn(f"[samples_to_batch_fn PID {pid}] Attempting to put None sentinel on results_queue.", flush=True)
		try:
			results_queue.put(None, timeout=0.2) 
			# print_fn(f"[samples_to_batch_fn PID {pid}] Successfully put None on results_queue.", flush=True)
		except queue.Full: 
			print_fn(f"[samples_to_batch_fn PID {pid}] results_queue full while trying to put sentinel in finally.", flush=True) # Keep important warning
		except Exception as e_rs:
			print_fn(f"[samples_to_batch_fn PID {pid}] Error putting None to results_queue in finally: {e_rs}", flush=True) # Keep error log
		# print_fn(f"[samples_to_batch_fn PID {pid}] Process finished.", flush=True)


def pipeline_data_generator(
	max_batches=-1,      
	batch_size=8,        
	sequence_length=4096,
	num_workers = 16, 
	objectives_config=None,
	accelerator=None,
	move_to_gpu_fn=None, 
	data_generator: Callable = None, 
	data_generator_fn_kwarg_overrides={},
	infinite_loop=True,  
	split=None,          
	config={},           
):
	print_fn = print 
	parent_pid = os.getpid()
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] INITIALIZING...", flush=True)

	if data_generator is None:
		raise ValueError("`data_generator` function must be provided.")
	if not callable(data_generator):
		raise ValueError("`data_generator` must be a callable function.")

	num_accelerator_processes = accelerator.num_processes if accelerator else 1
	num_workers_total_across_accelerators = num_workers * num_accelerator_processes 
	num_local_workers = num_workers 

	
	tasks_q = mp.Queue(maxsize=num_local_workers * 2)  
	sample_q = mp.Queue(maxsize=num_local_workers * 16 + num_accelerator_processes + 5) 
	final_batches_q = mp.Queue(maxsize=32 + num_accelerator_processes + 5) 
	output_q_for_generator = local_queue.Queue(maxsize=16) 


	if objectives_config is None: 
		objectives_config = [{"name": "next_token_prediction", "prob": 1.0}]

	worker_processes = []
	current_accelerator_worker_offset = (num_local_workers * accelerator.process_index) if accelerator else 0
	
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Creating {num_local_workers} Worker process objects...", flush=True)
	for local_idx in range(num_local_workers): 
		global_worker_id = current_accelerator_worker_offset + local_idx
		
		data_gen_kwargs_for_worker = dict(
			split=split,
			worker_id=global_worker_id, 
			num_workers=num_workers_total_across_accelerators, 
			rng_seed=config.get("seed", int(time.time())) + global_worker_id, 
		)
		data_gen_kwargs_for_worker.update(data_generator_fn_kwarg_overrides)
		
		this_worker_data_gen_fn = partial(data_generator, **data_gen_kwargs_for_worker)

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
			start_generating_paradigm=config.get("start_generating_paradigm", "\n<|assistant|>\n"),
			rng_seed=config.get("seed", 0) 
		)
		p = mp.Process(target=worker_instance_for_process.run, daemon=True)
		worker_processes.append(p)
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Worker process objects created.", flush=True)


	num_batcher_processes = 1 
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Creating {num_batcher_processes} samples_to_batch_fn process objects...", flush=True)
	batcher_processes = []
	batcher_stop_event = mp.Event() 

	for _ in range(num_batcher_processes):
		p = mp.Process(
			target=samples_to_batch_fn_wrapper, 
			args=(sample_q, final_batches_q, batch_size, batcher_stop_event, len(worker_processes)), 
			daemon=True
		)
		batcher_processes.append(p)
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] samples_to_batch_fn process objects created.", flush=True)


	main_ctrl = MainController(
		tasks_queue=tasks_q,
		sample_queue=sample_q, 
		objectives_config=objectives_config,
		worker_processes=worker_processes, 
		batcher_processes=batcher_processes, 
		batcher_stop_event=batcher_stop_event, 
		num_workers_total=len(worker_processes), 
	)
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Calling main_ctrl.run()", flush=True)
	main_ctrl.run() 

	datagen_stop_event = threading.Event() 
	datagen_thread = threading.Thread(
		target=datagen_thread_fn,
		args=(
			max_batches,
			final_batches_q,
			output_q_for_generator,
			datagen_stop_event, 
			num_batcher_processes, 
			accelerator,
			move_to_gpu_fn,
		),
		daemon=True
	)
	
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Starting {len(worker_processes)} worker processes...", flush=True)
	for p_idx, p in enumerate(worker_processes): 
		p.start()
		# print_fn(f"[pipeline_data_generator PID {parent_pid}] Started worker process {p_idx} (PID: {p.pid if p.pid else 'N/A - not started?'})", flush=True)

	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Starting {len(batcher_processes)} batcher processes...", flush=True)
	for p_idx, p in enumerate(batcher_processes): 
		p.start()
		# print_fn(f"[pipeline_data_generator PID {parent_pid}] Started batcher process {p_idx} (PID: {p.pid if p.pid else 'N/A - not started?'})", flush=True)

	# print_fn(f"[pipeline_data_generator PID {parent_pid}] Starting datagen_thread...", flush=True)
	datagen_thread.start()
	# print_fn(f"[pipeline_data_generator PID {parent_pid}] All processes/threads started.", flush=True)


	def _final_generator():
		print_gen_final = print 
		shutdown_initiated_by_generator = False
		# print_gen_final(f"[_final_generator PID {os.getpid()}] Generator created. Waiting for data...", flush=True)
		try:
			while True: 
				try:
					batch = output_q_for_generator.get(timeout=1.0) 
				except local_queue.Empty: 
					if datagen_stop_event.is_set(): 
						# print_gen_final(f"[_final_generator PID {os.getpid()}] datagen_stop_event IS SET during timeout. Breaking.", flush=True)
						shutdown_initiated_by_generator = True; break 
					if datagen_thread and not datagen_thread.is_alive():
						# print_gen_final(f"[_final_generator PID {os.getpid()}] datagen_thread NOT ALIVE during timeout. Breaking.", flush=True)
						shutdown_initiated_by_generator = True; break
					continue 

				if datagen_stop_event.is_set() and batch is None: 
					# print_gen_final(f"[_final_generator PID {os.getpid()}] Stop event set and batch is None or queue was empty. Breaking.", flush=True)
					shutdown_initiated_by_generator = True; break

				if batch is None: 
					# print_gen_final(f"[_final_generator PID {os.getpid()}] Received None sentinel from output_q. Ending.", flush=True)
					shutdown_initiated_by_generator = True; break 
				yield batch
				output_q_for_generator.task_done() 
		except Exception as e: 
			print_gen_final(f"[pipeline_data_generator PID {os.getpid()}] Exception in _final_generator: {e}", flush=True) # Keep error log
			traceback.print_exc(file=sys.stdout); sys.stdout.flush()
			shutdown_initiated_by_generator = True 
		finally:
			# print_gen_final(f"[_final_generator PID {os.getpid()}] Entering finally. Shutdown by gen: {shutdown_initiated_by_generator}, datagen_stop: {datagen_stop_event.is_set()}, batcher_stop: {batcher_stop_event.is_set()}", flush=True)
			if shutdown_initiated_by_generator: 
				if datagen_stop_event and not datagen_stop_event.is_set(): 
					# print_gen_final(f"[_final_generator PID {os.getpid()}] Setting datagen_stop_event in finally.", flush=True)
					datagen_stop_event.set() 
				if batcher_stop_event and not batcher_stop_event.is_set(): 
					# print_gen_final(f"[_final_generator PID {os.getpid()}] Setting batcher_stop_event in finally.", flush=True)
					batcher_stop_event.set() 
			# print_gen_final(f"[_final_generator PID {os.getpid()}] Exiting.", flush=True)
	
	return main_ctrl, _final_generator(), datagen_thread, batcher_stop_event, datagen_stop_event
