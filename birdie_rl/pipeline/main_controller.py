"""
MainController module:
- Sends a set of objective probabilities to the worker.
- Manages the lifecycle of worker processes and batcher processes.
- Corrected close() to use stop_event for batchers, not put None on sample_queue.
"""

import time
import queue # For queue.Full and queue.Empty
import multiprocessing as mp 

class MainController:
	"""
	The main controller that instructs the worker on which objectives to run
	and manages the lifecycle of associated processes.
	"""

	def __init__(
		self,
		tasks_queue: mp.Queue,
		sample_queue: mp.Queue, # sample_queue is INPUT to batchers, MainController should not put None here.
		objectives_config,
		worker_processes: list, 
		batcher_processes: list, 
		batcher_stop_event: mp.Event, # Added: event to stop batcher processes
		num_workers_total: int, 
	):
		self.tasks_queue = tasks_queue
		# self.sample_queue = sample_queue # MainController does not need direct access to sample_queue for putting sentinels
		self.objectives_config = objectives_config
		self.worker_processes = worker_processes
		self.batcher_processes = batcher_processes
		self.batcher_stop_event = batcher_stop_event # Store the event
		self.num_workers_managed_by_this_controller = len(worker_processes) # num_workers_total might be across all accelerators
		self.is_running = True
		self.print_fn = print # Simple print for controller logs

	def close(self, timeout_join=5.0):
		self.print_fn(f"[MainController] close() called. Shutting down {len(self.worker_processes)} workers and {len(self.batcher_processes)} batchers.")
		self.is_running = False

		# 1. Signal worker processes to stop by putting None on tasks_queue
		# self.print_fn(f"[MainController] Sending {self.num_workers_managed_by_this_controller} sentinels to tasks_queue for workers.")
		for _ in range(self.num_workers_managed_by_this_controller): # Send one per worker managed by this controller
			try:
				self.tasks_queue.put(None, timeout=1.0)
			except queue.Full:
				self.print_fn(f"[MainController Warning] tasks_queue full while sending sentinel to worker.")
			except Exception as e:
				self.print_fn(f"[MainController Error] putting sentinel to tasks_queue: {e}")

		# 2. Signal batcher processes to stop using their shared event
		if self.batcher_stop_event and not self.batcher_stop_event.is_set():
			self.print_fn(f"[MainController] Setting batcher_stop_event for {len(self.batcher_processes)} batcher processes.")
			try:
				self.batcher_stop_event.set()
			except Exception as e:
				self.print_fn(f"[MainController Error] setting batcher_stop_event: {e}")
		# DO NOT put None on sample_queue from MainController. Workers do that.

		# 3. Join worker processes
		# self.print_fn(f"[MainController] Joining {len(self.worker_processes)} worker processes...")
		for i, p in enumerate(self.worker_processes):
			if p.is_alive():
				p.join(timeout=timeout_join)
				if p.is_alive():
					self.print_fn(f"[MainController Warning] Worker process {i} (PID: {p.pid if hasattr(p, 'pid') else 'N/A'}) did not terminate after {timeout_join}s. Forcing termination.")
					p.terminate(); p.join(timeout=1.0)
		# self.print_fn(f"[MainController] All worker processes joined or terminated.")

		# 4. Join batcher processes
		# self.print_fn(f"[MainController] Joining {len(self.batcher_processes)} batcher processes...")
		for i, p in enumerate(self.batcher_processes):
			if p.is_alive():
				p.join(timeout=timeout_join)
				if p.is_alive():
					self.print_fn(f"[MainController Warning] Batcher process {i} (PID: {p.pid if hasattr(p, 'pid') else 'N/A'}) did not terminate after {timeout_join}s. Forcing termination.")
					p.terminate(); p.join(timeout=1.0)
		# self.print_fn(f"[MainController] All batcher processes joined or terminated.")
		
		self.print_fn(f"[MainController] Shutdown sequence complete.")


	def update(self, objectives_config, clear_prefetched=False):
		if not self.is_running:
			# self.print_fn("[MainController] update() called, but controller is not running. Ignoring.")
			return

		self.objectives_config = objectives_config
		instructions = {"objectives": self.objectives_config}
		
		for _ in range(self.num_workers_managed_by_this_controller): 
			try:
				self.tasks_queue.put(instructions, timeout=0.5)
			except queue.Full:
				self.print_fn(f"[MainController Warning] tasks_queue full while sending objective update.")
				break 
			except Exception as e:
				self.print_fn(f"[MainController Error] putting objective update to tasks_queue: {e}")
				break


	def run(self):
		if not self.is_running:
			# self.print_fn("[MainController] run() called, but controller is not running. Ignoring.")
			return
		# self.print_fn(f"[MainController] run() called. Sending initial objectives.")
		self.update(self.objectives_config)
		# self.print_fn(f"[MainController] Initial objectives sent.")


	def stop(self): 
		# self.print_fn(f"[MainController] stop() called.")
		self.close()
