"""
MainController module:
- Sends a set of objective probabilities to the worker.
- Consumes a certain number of batches from the results queue (max_batches).
- After enough batches, sends a sentinel (None) to stop the worker.
"""

import time
import queue


class MainController:
	"""
	The main controller that instructs the worker on which objectives to run.
	"""

	def __init__(
		self,
		tasks_queue: queue.Queue,
		results_queue: queue.Queue,
		objectives_config,
		num_workers=1,
		max_batches=32,
		# generator=None,
	):
		"""
		Args:
			tasks_queue: Queue for sending instructions (dict or None) to the worker.
			results_queue: Queue from which to receive batch results from the worker.
			objectives_config: A list of dicts describing objectives:
			  {
				"name": str,
				"prob": float,
				"config_overrides": {...}
			  }
			max_batches: Number of total batches we want to collect.
		"""
		self.tasks_queue = tasks_queue
		self.results_queue = results_queue
		self.objectives_config = objectives_config
		self.max_batches = max_batches
		self.num_workers = num_workers
		self.is_running = True
		# self.generator = generator

	def close(self):
		"""
		Close the controller by draining queues and sending sentinel values.
		This version avoids using qsize() and uses get_nowait() instead.
		"""
		self.is_running = False

		# Drain tasks_queue non-blockingly
		try:
			while True:
				self.tasks_queue.get_nowait()
		except Exception:
			pass

		# Drain results_queue non-blockingly
		try:
			while True:
				self.results_queue.get_nowait()
		except Exception:
			pass

                # Send a sentinel (None) for each worker so they can exit.
                for _ in range(self.num_workers):
                        try:
                                self.tasks_queue.put_nowait(None)
                        except Exception:
                                pass


	def update(self, objectives_config, clear_prefetched=False,):
		"""
		Update the objectives configuration.
		"""
		self.objectives_config = objectives_config
                instructions = {"objectives": self.objectives_config}

                # Drain any existing instructions so workers read the latest.
                try:
                        while True:
                                self.tasks_queue.get_nowait()
                except Exception:
                        pass

                if clear_prefetched:
                        try:
                                while True:
                                        self.results_queue.get_nowait()
                        except Exception:
                                pass

                # Send the updated instructions once for each worker.
                for _ in range(self.num_workers):
                        self.tasks_queue.put(instructions)



	def run(self):
		"""
		Main routine:
		1) Send the objective distribution to the worker.
		"""
		self.update(self.objectives_config)
		


	def stop(self):
		"""
		Manually stop the controller (optional usage).
		"""
		self.is_running = False
		self.tasks_queue.put(None)
		self.close()
