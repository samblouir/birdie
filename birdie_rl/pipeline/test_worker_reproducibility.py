import queue as thread_queue
import unittest
from unittest import mock

import numpy as np

from birdie_rl.pipeline.worker import Worker


class MockTokenizer:
	def encode(self, text: str):
		if not isinstance(text, str):
			raise TypeError(f"MockTokenizer.encode expected str, got {type(text)}")
		return [ord(c) for c in text]

	def decode(self, ids):
		if ids is None:
			return ""
		if isinstance(ids, np.ndarray):
			ids = ids.tolist()
		return "".join(chr(i) for i in ids if isinstance(i, int) and i >= 0)


def infinite_text_generator(split, worker_id, num_workers, rng_seed=0):
	_ = (split, worker_id, num_workers, rng_seed)
	text = (
		"Birdie worker reproducibility test. "
		+ ("abcdefghijklmnopqrstuvwxyz " * 2000)
	)
	while True:
		yield {"text": text}


def _collect_batches(worker: Worker, sample_q: thread_queue.Queue, target_batches: int) -> list[dict]:
	worker.initialize_data_iterator()

	batches: list[dict] = []
	max_calls = 2000
	for _ in range(max_calls):
		worker._produce_one_sample()
		while True:
			try:
				item = sample_q.get_nowait()
			except thread_queue.Empty:
				break
			else:
				batches.append(item["stacked_batch_data"])
		if len(batches) >= target_batches:
			break
	return batches[:target_batches]


class TestWorkerReproducibility(unittest.TestCase):
	def test_deterministic_worker_rng_is_pid_independent(self):
		"""
		When `config['deterministic_worker_rng']=True`, objective selection and per-sample
		rng_seeds should be reproducible across different PIDs (e.g. multiprocess workers).
		"""

		def make_worker_with_pid(pid: int):
			with mock.patch("os.getpid", return_value=pid):
				tasks_q = thread_queue.Queue()
				sample_q = thread_queue.Queue(maxsize=2048)
				worker = Worker(
					worker_id=0,
					total_workers=1,
					tasks_queue=tasks_q,
					results_queue=None,
					sample_queue=sample_q,
					data_generator=infinite_text_generator,
					sequence_length=512,
					min_seq_len_for_packing=64,
					tokenizer=MockTokenizer(),
					split="train",
					infinite_loop=True,
					start_generating_id=2,
					latent_token_id=1,
					max_samples_per_packer=float("inf"),
					rng_seed=123,
					config={"batch_size": 1, "deterministic_worker_rng": True},
				)
				worker.objectives_info = [
					{
						"name": "infilling",
						"prob": 1.0,
						"config_overrides": {"max_attempts": 8, "max_mask_spans": 8, "use_fast_backend": False},
					}
				]
				worker.og_probs = np.array([1.0], dtype=np.float32)
				return worker, sample_q

		worker_a, q_a = make_worker_with_pid(111)
		worker_b, q_b = make_worker_with_pid(222)
		try:
			batches_a = _collect_batches(worker_a, q_a, target_batches=2)
			batches_b = _collect_batches(worker_b, q_b, target_batches=2)
		finally:
			worker_a.close()
			worker_b.close()

		self.assertEqual(len(batches_a), 2)
		self.assertEqual(len(batches_b), 2)

		for batch_idx in range(2):
			a = batches_a[batch_idx]
			b = batches_b[batch_idx]
			self.assertEqual(set(a.keys()), set(b.keys()))
			for k in a.keys():
				self.assertTrue(np.array_equal(a[k], b[k]), f"Mismatch for key={k} in batch_idx={batch_idx}")

