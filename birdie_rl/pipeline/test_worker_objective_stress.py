import contextlib
import itertools
import queue as thread_queue
import signal
import time
import unittest

import numpy as np

from birdie_rl.pipeline.worker import Worker


class MockTokenizer:
	"""
	Simple, deterministic tokenizer for tests:
	- encode: 1 token per character
	- decode: inverse of encode for ASCII-ish ints
	"""

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
		"Birdie worker stress test. "
		"This string is intentionally long enough to satisfy minimum lengths. "
		+ ("0123456789 " * 3000)
	)
	while True:
		yield {"text": text}


class CyclingRNG:
	"""
	Deterministic objective selection for mixed-objective stress tests.

	- `choice(...)` cycles through a fixed index sequence (ignores `p=...`).
	- `integers(...)` delegates to a real NumPy RNG for per-sample seeds.
	"""

	def __init__(self, indices, seed: int = 0):
		self._cycle = itertools.cycle(indices)
		self._rng = np.random.default_rng(seed)

	def choice(self, a, p=None):  # noqa: ARG002
		if not isinstance(a, int):
			raise TypeError(f"CyclingRNG.choice expects int 'a', got {type(a)}")
		return int(next(self._cycle) % a)

	def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
		return self._rng.integers(low, high=high, size=size, dtype=dtype, endpoint=endpoint)


@contextlib.contextmanager
def alarm_timeout(seconds: int, message: str):
	def _handler(signum, frame):  # noqa: ARG001
		raise TimeoutError(message)

	old_handler = signal.signal(signal.SIGALRM, _handler)
	try:
		signal.setitimer(signal.ITIMER_REAL, float(seconds))
		yield
	finally:
		signal.setitimer(signal.ITIMER_REAL, 0.0)
		signal.signal(signal.SIGALRM, old_handler)


OBJECTIVES = [
	"autoencoding_with_deshuffling",
	"autoencoding",
	"copying",
	"deshuffling",
	"infilling",
	"next_token_prediction",
	"prefix_language_modeling",
	"selective_copying",
]

OBJECTIVE_OVERRIDES = {
	# Keep stress tests fast and deterministic.
	"autoencoding_with_deshuffling": {"max_attempts": 8},
	"autoencoding": {"max_attempts": 8},
	"infilling": {"max_attempts": 8, "max_mask_spans": 8},
	"selective_copying": {
		"max_attempts": 8,
		"tokens_per_mask": 4,
		"min_delimiter_prefix_length": 2,
		"max_delimiter_prefix_length": 8,
		"min_delimiter_suffix_length": 2,
		"max_delimiter_suffix_length": 8,
		"objective_verbosity": 0,
	},
}


class TestWorkerObjectiveStress(unittest.TestCase):
	def test_objective_cache_does_not_grow_per_sample(self):
		"""
		Stress each objective individually and ensure the worker doesn't leak
		objective instances per-sample (which can look like a hang due to memory
		blow-up / GC pressure).
		"""
		for objective_name in OBJECTIVES:
			with self.subTest(objective=objective_name):
				tasks_q = thread_queue.Queue()
				sample_q = thread_queue.Queue(maxsize=128)

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
						"name": objective_name,
						"prob": 1.0,
						"config_overrides": OBJECTIVE_OVERRIDES.get(objective_name, {}),
					}
				]
				worker.og_probs = np.array([1.0], dtype=np.float32)
				worker.initialize_data_iterator()

				with alarm_timeout(10, f"Worker hung while sampling objective '{objective_name}'"):
					batches_produced = 0
					for _ in range(200):
						worker._produce_one_sample()
						self.assertLessEqual(
							len(worker.objective_cache),
							4,
							f"objective_cache grew unexpectedly for '{objective_name}' (size={len(worker.objective_cache)})",
						)
						while True:
							try:
								_ = sample_q.get_nowait()
							except thread_queue.Empty:
								break
							else:
								batches_produced += 1

				# With a single objective, the cache should be O(1), not O(samples).
				self.assertGreater(
					batches_produced,
					0,
					f"Worker produced no batches for '{objective_name}'",
				)

				worker.close()

	def test_objectives_do_not_hang_at_large_sequence_lengths(self):
		"""
		Stress each objective at the sequence lengths you care about in training:
		- 2048
		- 16384
		"""

		for sequence_length in (2048, 16384):
			for objective_name in OBJECTIVES:
				with self.subTest(objective=objective_name, sequence_length=sequence_length):
					tasks_q = thread_queue.Queue()
					sample_q = thread_queue.Queue(maxsize=256)

					worker = Worker(
						worker_id=0,
						total_workers=1,
						tasks_queue=tasks_q,
						results_queue=None,
						sample_queue=sample_q,
						data_generator=infinite_text_generator,
						sequence_length=sequence_length,
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
							"name": objective_name,
							"prob": 1.0,
							"config_overrides": OBJECTIVE_OVERRIDES.get(objective_name, {}),
						}
					]
					worker.og_probs = np.array([1.0], dtype=np.float32)
					worker.initialize_data_iterator()

					timeout_s = 20 if sequence_length >= 16384 else 10
					with alarm_timeout(
						timeout_s,
						f"Worker hung while sampling objective '{objective_name}' (sequence_length={sequence_length})",
					):
						batches_produced = 0
						calls = 0
						max_calls = 500 if sequence_length >= 16384 else 200
						target_batches = 5
						start_t = time.perf_counter()

						for _ in range(max_calls):
							calls += 1
							worker._produce_one_sample()
							self.assertLessEqual(
								len(worker.objective_cache),
								4,
								f"objective_cache grew unexpectedly for '{objective_name}' (size={len(worker.objective_cache)})",
							)

							while True:
								try:
									_ = sample_q.get_nowait()
								except thread_queue.Empty:
									break
								else:
									batches_produced += 1

							if batches_produced >= target_batches:
								break
						elapsed_s = max(1e-9, time.perf_counter() - start_t)

					print(
						f"[throughput] objective={objective_name} "
						f"seq_len={sequence_length} "
						f"calls={calls} batches={batches_produced} "
						f"elapsed_s={elapsed_s:.3f} "
						f"calls_per_s={calls/elapsed_s:.1f} "
						f"batches_per_s={batches_produced/elapsed_s:.2f} "
						f"tokens_per_s~={(batches_produced * sequence_length)/elapsed_s:.0f}",
						flush=True,
					)

					self.assertGreater(
						batches_produced,
						target_batches - 1,
						f"Worker produced no batches for '{objective_name}' (sequence_length={sequence_length})",
					)

					worker.close()

	def test_objectives_mixed_together_do_not_hang(self):
		"""
		Stress the same mixed objective distribution Birdie uses in practice.

		This catches deadlocks caused by interactions between:
		- leftover text reuse
		- per-objective min_remaining_space checks
		- packing readiness / flushing logic
		"""

		for sequence_length in (2048, 16384):
			with self.subTest(sequence_length=sequence_length):
				tasks_q = thread_queue.Queue()
				sample_q = thread_queue.Queue(maxsize=512)

				worker = Worker(
					worker_id=0,
					total_workers=1,
					tasks_queue=tasks_q,
					results_queue=None,
					sample_queue=sample_q,
					data_generator=infinite_text_generator,
					sequence_length=sequence_length,
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
						"name": name,
						"prob": 1.0,
						"config_overrides": OBJECTIVE_OVERRIDES.get(name, {}),
					}
					for name in OBJECTIVES
				]
				worker.og_probs = np.array([1.0 / len(OBJECTIVES)] * len(OBJECTIVES), dtype=np.float32)
				worker.rng = CyclingRNG(indices=list(range(len(OBJECTIVES))), seed=0)
				worker.initialize_data_iterator()

				timeout_s = 30 if sequence_length >= 16384 else 20
				with alarm_timeout(
					timeout_s,
					f"Worker hung while sampling mixed objectives (sequence_length={sequence_length})",
				):
					batches_produced = 0
					calls = 0
					max_calls = 600 if sequence_length >= 16384 else 400
					target_batches = 10
					start_t = time.perf_counter()

					for _ in range(max_calls):
						calls += 1
						worker._produce_one_sample()

						self.assertLessEqual(
							len(worker.objective_cache),
							len(OBJECTIVES) + 2,
							f"objective_cache grew unexpectedly under mixed sampling (size={len(worker.objective_cache)})",
						)

						while True:
							try:
								_ = sample_q.get_nowait()
							except thread_queue.Empty:
								break
							else:
								batches_produced += 1

						if batches_produced >= target_batches and len(worker.objective_cache) == len(OBJECTIVES):
							break
					elapsed_s = max(1e-9, time.perf_counter() - start_t)

				print(
					f"[throughput] objective=mixed "
					f"seq_len={sequence_length} "
					f"calls={calls} batches={batches_produced} "
					f"elapsed_s={elapsed_s:.3f} "
					f"calls_per_s={calls/elapsed_s:.1f} "
					f"batches_per_s={batches_produced/elapsed_s:.2f} "
					f"tokens_per_s~={(batches_produced * sequence_length)/elapsed_s:.0f}",
					flush=True,
				)

				self.assertEqual(
					len(worker.objective_cache),
					len(OBJECTIVES),
					f"Not all objectives were exercised under mixed sampling (cache_size={len(worker.objective_cache)})",
				)
				self.assertGreaterEqual(
					batches_produced,
					target_batches,
					f"Worker produced no batches under mixed sampling (sequence_length={sequence_length})",
				)

				worker.close()
