# pipeline_generator.py
import os
import threading
import queue
from typing import List
import torch
from datasets import load_dataset
import multiprocessing as mp
import queue as local_queue
import time
from birdie_rl.pipeline.main_controller import MainController
from birdie_rl.pipeline.worker import Worker
# from birdie_rl.load_objective import load_objective
# from birdie_rl.packer import SequencePacker
import dill
import numpy as np

def datagen(
	max_batches: int,
	results_q: mp.Queue,
	tasks_q: mp.Queue,
	output_q: queue.Queue,
	sample_q: mp.Queue,

	worker_threads: List[threading.Thread],
	accelerator=None,
	move_to_gpu_fn=None,
	split=None,
):
	print_fn = print if accelerator is None else accelerator.print

	batches_received = 0
	while (max_batches == -1) or (batches_received < max_batches):
		try:
			# 5) Wait for a batch from the worker => 
			#    each batch is a dict: {"batch_items": [...], "worker_id": ...}
			batch_dict = results_q.get(timeout=0.1)
			# batch_dict = dill.loads(batch_dict)
		except queue.Empty:

			
			# if split and split == "train":
			# 	tp = {
			# 		"results_q.qsize()": results_q.qsize(),
			# 		"output_q.qsize()": output_q.qsize(),
			# 		"tasks_q.qsize()": tasks_q.qsize(),
			# 		"sample_q.qsize()": sample_q.qsize(),
			# 	}
			# 	strs = []
			# 	for tp_idx, (key, value) in enumerate(tp.items()):
			# 		strs.append(f"{key}: {value:,}")
			# 	print_fn(f"  pipeline_generator.py  batches_received: {batches_received:,},  ", '  '.join(strs), flush=True, end='\n')
				

				
			# print_fn(f"[pipeline_data_generator] Timed out waiting for a batch. results_q.qsize(): {results_q.qsize()} Retrying...")
			continue

		if batch_dict is None:
			# If we ever got the "None" sentinel that stems from birdie.close(), we should exit our while look, send out "None" on our queues to cause those threads or processes to exit also, and then try to join all of our threads.
			# TODO: FIX THIS. PATCH IMPLEMENTED BELOW USING os._exit(1)
			break

		# tp = {
		# 	"results_q.qsize()": results_q.qsize(),
		# 	"output_q.qsize()": output_q.qsize(),
		# 	"tasks_q.qsize()": tasks_q.qsize(),
		# 	"sample_q.qsize()": sample_q.qsize(),
		# }
		# strs = []
		# for tp_idx, (key, value) in enumerate(tp.items()):
		# 	strs.append(f"{key}: {value:,}")
		# print_fn(f"  pipeline_generator.py  batches_received: {batches_received:,},  ", '  '.join(strs), flush=True, end='\n')
			

		batches_received += 1
		# print(f"  batch_dict: {batch_dict}")
		
			
		batch = batch_dict["batch_items"]
		# print(f"  received batch: {batch.keys()}, results_q.qsize(): {results_q.qsize()}, output_q.qsize(): {output_q.qsize()}, tasks_q.qsize(): {tasks_q.qsize()}, sample_q.qsize(): {sample_q.qsize()}", flush=True,)


		if move_to_gpu_fn is not None:
			batch = move_to_gpu_fn(batch)

		output_q.put(batch)
		# yield batch
	
	# 8) Once we've yielded all we want, we can stop the worker
	#    by sending the sentinel None. The MainController also does this,
	#    but in case you want a direct stop:
	tasks_q.put(None)
	output_q.put(None)

	# Make sure threads close properly (you can optionally join them here)
	# for idx, worker_thread in enumerate(worker_threads):
	# 	print_fn(f"  Joining all pipeline worker threads. ({idx+1} of {len(worker_threads)})", flush=True,)
	# 	worker_thread.join()
	# print_fn(f"  All pipeline worker threads joined.", flush=True,)

	# Patch to try and solve non-exiting threads or processes
	os._exit(1)

def samples_to_batch(
		sample_queue: mp.Queue,
		results_queue: mp.Queue,
		batch_size:int,
	):
	batch = []

	start_time = time.time()

	time_waiting_for_samples = []
	ctr = 0
	while True:
		try:
			start_waiting = time.time()
			packed_sample = sample_queue.get(timeout=1.0)
			time_waiting = time.time() - start_waiting
			time_waiting_for_samples.append(time_waiting)
		except Exception as e:
			# print(f"  Exception in samples_to_batch():  sample_queue.qsize(): {sample_queue.qsize():,},  sample_queue.get() e: {e}")
			continue


		# worker_id = packed_sample["worker_id"]
		# packed_arr = packed_sample["packed_data"]
		# objective_name = packed_sample["objective_name"]

		batch.append(packed_sample)

		if len(batch) == batch_size:
			start_time_concatenating = time.time()
			keys = [
				"input_ids",
				"label_ids",
				"segment_ids",
				"attention_mask",
			]
			stacked_batch = {}
			for key in keys:
				stacked_batch[key] = np.stack([x["packed_data"][key] for x in batch])
			stacked_batch = {k:torch.tensor(v, dtype=torch.long) for k,v in stacked_batch.items()}
			batch = []

			end_time_concatenating = time.time()
			time_concatenating = (end_time_concatenating - start_time_concatenating)

			# start_time = new_time


			time_sending = time.time()
			results_queue.put({
				# "worker_id": worker_id,
				"batch_items": stacked_batch,
				# "objective_name": objective_name,
			})


			new_time = time.time()
			elapsed = (new_time - start_time)

			time_sending = (new_time - time_sending)

			ctr += 1
			time_waiting_for_samples = np.sum(time_waiting_for_samples)
			throughput = (ctr) / elapsed



			# print(f"  Took {elapsed:.2f} seconds to pack and send a batch of {batch_size} samples.  Throughput: {throughput:.2f} batches/sec,  time_waiting_for_samples: {time_waiting_for_samples:0.2f},  time_concatenating: {time_concatenating:0.2f},  time_sending: {time_sending:0.2f}", flush=True, )
			if ctr == 5:
				start_time = time.time()
				ctr = 0
			time_waiting_for_samples = []




def pipeline_data_generator(
	max_batches=-1,
	batch_size=8,
	sequence_length=4096,
	num_workers = 16,
	# sequence_length=128
	objectives_config=None,
	accelerator=None,
	move_to_gpu_fn=None,
	data_generator=None,
	data_generator_fn_kwarg_overrides={},
	infinite_loop=True,
	split=None,
	# accelerator=None,
	config={},
):
	assert(data_generator is not None), f"  {__file__}.pipeline_data_generator(): data_generator is None. Please provide a data generator. pipeline_data_generator() currently only supports a list for the data_generator input that supports len() and data_generator[worker_idx::num_workers]."
	# try:
	# 	assert(0 < len(data_generator))
	# except Exception as e:
	# 	raise NotImplementedError(f'  {__file__}.pipeline_data_generator(): You passed in ([...], data_generator={data_generator}), and pipeline_data_generator() currently only supports a list for the data_generator input that supports len() and data_generator[worker_idx::num_workers].')

	"""
	Spawns the pipeline in threads and yields (input_ids, label_ids, segment_ids, attention_mask)
	for each batch. This can directly replace your dummy data generator in minimal_trainer.
	"""
	# 0) Calculate
	if accelerator is None:
		total_workers = (num_workers)
	else:
		total_workers = (num_workers * accelerator.num_processes)


	# 1) Create queues
	tasks_q = mp.Queue()
	results_q = mp.Queue(8)
	sample_queue = mp.Queue(8)
	output_q = local_queue.Queue(8)

	if objectives_config is None:
		next_token_prediction = {
				"name": "next_token_prediction",
				"prob": 0.5,
			}
		objectives_config = [next_token_prediction]

	# if split is not None and split == 'train':
	# 	worker_idx = accelerator.process_index
	# 	num_processes = accelerator.num_processes
	# 	total_num_workers = num_processes * num_workers
	
	
	worker_threads = []
	our_worker_id_offset = (num_workers*accelerator.process_index) if accelerator is not None else 0
	for worker_id in range(our_worker_id_offset, our_worker_id_offset + num_workers):

		# if split and split in ['train']:
		# 	our_data_generator = data_generator[worker_id::total_workers]
		# 	print(f"  pipeline_generator.py  worker_id: {worker_id},  len(our_data_generator): {len(our_data_generator)}")
		# else:
		# 	our_data_generator = data_generator
		# 	print(f"  pipeline_generator.py  worker_id: {worker_id},  split: {split}")
		# print(f"  data_generator: {data_generator}")
		data_generator_kwargs = dict(
			split=split,
			worker_id=worker_id,
			num_workers=total_workers,
			rng_seed=0,
		)
		data_generator_kwargs.update(data_generator_fn_kwarg_overrides)
		our_data_generator = data_generator(**data_generator_kwargs)

		
		worker = Worker(
			worker_id=worker_id,
			total_workers=total_workers,
			tasks_queue=tasks_q,
			results_queue=results_q,
			sample_queue=sample_queue,
			sequence_length=sequence_length,
			batch_size=batch_size,
			# batch_size=1,
			min_seq_len_for_packing=config.get("min_seq_len_for_packing", 64),
			data_generator=our_data_generator,
			infinite_loop=infinite_loop,
			split=split,
			tokenizer=config['tokenizer'],
			text_grabber_fn=config.get("text_grabber_fn", None),
			start_generating_paradigm=config.get("start_generating_paradigm", "\n<|assistant|>\n"),
		)
		worker_thread = mp.Process(target=worker.run, )
		worker_threads.append(worker_thread)

	num_bp = 8 if split == 'train' else 1
	for batcher in range(num_bp):
		batch_proc = mp.Process(target=samples_to_batch, args=(sample_queue, results_q, batch_size))
		worker_threads.append(batch_proc)

	# This is the generator that the workers output to.
	datagen_kwargs = dict(
		max_batches=max_batches,
		results_q=results_q,
		tasks_q=tasks_q,
		output_q=output_q,
		sample_q=sample_queue,
		worker_threads=worker_threads,
		accelerator=accelerator,
		move_to_gpu_fn=move_to_gpu_fn,
		split=split,
	)
	# generator = datagen(**datagen_kwargs)
	threading.Thread(target=datagen, kwargs=datagen_kwargs).start()

	def _generator():
		ctr = 0
		while True:
			try:
				yield output_q.get(timeout=1)
			except queue.Empty:
				continue
			ctr += 1

	generator = _generator()

	
	# 3) Create MainController & Worker
	main_ctrl = MainController(
		tasks_queue=tasks_q,
		results_queue=results_q,
		objectives_config=objectives_config,
		num_workers=num_workers,
		max_batches=max_batches,
		# move_to_gpu_fn=move_to_gpu_fn,
	)

	main_ctrl.run()


	
	for worker_id, (worker_thread) in enumerate(worker_threads):
		worker_thread.start()
		# print(f"  Started worker_id: {worker_id}")

	return (main_ctrl, generator)



#
# Provide your text data to the Worker
#
def _my_text_source():
	"""
	Example text generator. Replace with real data or Hugging Face splits, etc.
	This yields lines or documents repeatedly.
	"""
	# texts = [
	# 	"Hello world. This is sample #1.",
	# 	"Pipeline demonstration line #2.",
	# 	"Third line with some example text.",
	# 	"Another line for autoencoding or copying objective."
	# ]
	# while True:
	# 	for t in texts:
	# 		yield t
			
	"""
	A generator of text for demonstration.
	This uses "roneneldan/TinyStories" on HuggingFace for example data.
	Replace with your own source if desired.
	"""
	ds = load_dataset("roneneldan/TinyStories", split="train")
	while True:
		for x in ds:
			yield x["text"]
	# while True:
	# 	# x = "abcdefghijklmnopqrstuvwxyz"
	# 	yield '''ant bird cat dog eel fox gat hen iggy jay koi lion mole new owl PIG quail rat sow tan uwu vole wolf xxx yam zebu '''
	# 	# x = "0123456789"
	# 	# yield ' '.join(x)
	# 	# yield "0 1 2 3 " * 10
	# 	# yield "0 1 2 3 4 5 6 7 8 9 " * 10
