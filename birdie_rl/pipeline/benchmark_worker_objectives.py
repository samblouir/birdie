import argparse
import contextlib
import itertools
import queue as thread_queue
import signal
import time

import numpy as np

from birdie_rl.pipeline.worker import Worker


class MockTokenizer:
	"""
	Simple, deterministic tokenizer for benchmarking:
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

def _is_fast_infilling_available() -> bool:
	try:
		from birdie_rl.objectives import infilling as infilling_mod
	except Exception:
		return False
	return getattr(infilling_mod, "_infilling_fast", None) is not None


def infinite_text_generator(split, worker_id, num_workers, rng_seed=0):
	_ = (split, worker_id, num_workers, rng_seed)
	text = (
		"Birdie worker objective benchmark. "
		"This string is intentionally long enough to satisfy minimum lengths. "
		+ ("0123456789 " * 3000)
	)
	while True:
		yield {"text": text}


class CyclingRNG:
	"""
	Deterministic objective selection for mixed-objective benchmarks.

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


DEFAULT_OBJECTIVES = [
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
	# Keep the benchmark fast and deterministic.
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


def _make_worker(sequence_length: int, batch_size: int, deterministic_worker_rng: bool):
	tasks_q = thread_queue.Queue()
	sample_q = thread_queue.Queue(maxsize=1024)

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
		config={"batch_size": batch_size, "deterministic_worker_rng": deterministic_worker_rng},
	)
	worker.initialize_data_iterator()
	return worker, sample_q


def _drain_batches(sample_q: thread_queue.Queue) -> int:
	batches = 0
	while True:
		try:
			_ = sample_q.get_nowait()
		except thread_queue.Empty:
			break
		else:
			batches += 1
	return batches


def bench_single_objective(
	objective_name: str,
	sequence_length: int,
	batch_size: int,
	target_batches: int,
	max_calls: int,
	timeout_s: int,
	infilling_backend: str,
	infilling_candidates: int,
	deterministic_worker_rng: bool,
):
	worker, sample_q = _make_worker(sequence_length, batch_size, deterministic_worker_rng)
	config_overrides = OBJECTIVE_OVERRIDES.get(objective_name, {}).copy()
	if objective_name == "infilling":
		config_overrides["fast_backend_num_candidates"] = int(max(1, infilling_candidates))
		if infilling_backend == "python":
			config_overrides["use_fast_backend"] = False
		elif infilling_backend == "fast":
			config_overrides["use_fast_backend"] = True
	worker.objectives_info = [
		{
			"name": objective_name,
			"prob": 1.0,
			"config_overrides": config_overrides,
		}
	]
	worker.og_probs = np.array([1.0], dtype=np.float32)

	batches_produced = 0
	calls = 0
	start_t = time.perf_counter()

	with alarm_timeout(timeout_s, f"Hung while sampling '{objective_name}' (sequence_length={sequence_length})"):
		for _ in range(max_calls):
			calls += 1
			worker._produce_one_sample()
			batches_produced += _drain_batches(sample_q)
			if batches_produced >= target_batches:
				break

	elapsed_s = max(1e-9, time.perf_counter() - start_t)
	tokens_per_s = (batches_produced * batch_size * sequence_length) / elapsed_s
	print(
		f"[throughput] objective={objective_name} "
		f"seq_len={sequence_length} batch={batch_size} "
		f"calls={calls} batches={batches_produced} "
		f"elapsed_s={elapsed_s:.3f} "
		f"calls_per_s={calls/elapsed_s:.1f} "
		f"batches_per_s={batches_produced/elapsed_s:.2f} "
		f"tokens_per_s~={tokens_per_s:.0f}",
		flush=True,
	)

	worker.close()
	if batches_produced < target_batches:
		raise RuntimeError(
			f"Only produced {batches_produced}/{target_batches} batches for '{objective_name}' "
			f"(sequence_length={sequence_length}, max_calls={max_calls})"
		)
	return {
		"objective": objective_name,
		"sequence_length": int(sequence_length),
		"batch_size": int(batch_size),
		"calls": int(calls),
		"batches": int(batches_produced),
		"elapsed_s": float(elapsed_s),
		"tokens_per_s": float(tokens_per_s),
	}


def bench_mixed_objectives(
	objectives,
	sequence_length: int,
	batch_size: int,
	target_batches: int,
	max_calls: int,
	timeout_s: int,
	infilling_backend: str,
	infilling_candidates: int,
	deterministic_worker_rng: bool,
):
	worker, sample_q = _make_worker(sequence_length, batch_size, deterministic_worker_rng)
	worker.objectives_info = [
		{
			"name": name,
			"prob": 1.0,
			"config_overrides": (
				{
					**OBJECTIVE_OVERRIDES.get(name, {}),
					**(
						{"fast_backend_num_candidates": int(max(1, infilling_candidates))}
						if name == "infilling"
						else {}
					),
					**(
						{"use_fast_backend": False}
						if (name == "infilling" and infilling_backend == "python")
						else {"use_fast_backend": True}
						if (name == "infilling" and infilling_backend == "fast")
						else {}
					),
				}
			),
		}
		for name in objectives
	]
	worker.og_probs = np.array([1.0 / len(objectives)] * len(objectives), dtype=np.float32)
	worker.rng = CyclingRNG(indices=list(range(len(objectives))), seed=0)

	batches_produced = 0
	calls = 0
	start_t = time.perf_counter()

	with alarm_timeout(timeout_s, f"Hung while sampling mixed objectives (sequence_length={sequence_length})"):
		for _ in range(max_calls):
			calls += 1
			worker._produce_one_sample()
			batches_produced += _drain_batches(sample_q)
			if batches_produced >= target_batches and len(worker.objective_cache) == len(objectives):
				break

	elapsed_s = max(1e-9, time.perf_counter() - start_t)
	tokens_per_s = (batches_produced * batch_size * sequence_length) / elapsed_s
	print(
		f"[throughput] objective=mixed "
		f"seq_len={sequence_length} batch={batch_size} "
		f"calls={calls} batches={batches_produced} "
		f"elapsed_s={elapsed_s:.3f} "
		f"calls_per_s={calls/elapsed_s:.1f} "
		f"batches_per_s={batches_produced/elapsed_s:.2f} "
		f"tokens_per_s~={tokens_per_s:.0f}",
		flush=True,
	)

	worker.close()
	if batches_produced < target_batches:
		raise RuntimeError(
			f"Only produced {batches_produced}/{target_batches} batches for mixed objectives "
			f"(sequence_length={sequence_length}, max_calls={max_calls})"
		)
	return {
		"objective": "mixed",
		"sequence_length": int(sequence_length),
		"batch_size": int(batch_size),
		"calls": int(calls),
		"batches": int(batches_produced),
		"elapsed_s": float(elapsed_s),
		"tokens_per_s": float(tokens_per_s),
	}


def main(argv=None):
	parser = argparse.ArgumentParser(description="Benchmark Birdie objective sample generation (no PyTorch required).")
	parser.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 16384])
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--objectives", nargs="+", default=DEFAULT_OBJECTIVES)
	parser.add_argument("--per-objective-batches", type=int, default=5)
	parser.add_argument("--mixed-batches", type=int, default=10)
	parser.add_argument(
		"--infilling-backend",
		choices=["auto", "python", "fast"],
		default="auto",
		help="Select the infilling implementation: auto (default), python, or fast (C++ if available).",
	)
	parser.add_argument(
		"--compare-infilling-backends",
		action="store_true",
		help="Benchmark infilling twice (python vs fast) and print the speedup.",
	)
	parser.add_argument(
		"--infilling-candidates",
		type=int,
		default=1,
		help="When using the fast infilling backend, try N deterministic seed candidates in parallel (default: 1).",
	)
	parser.add_argument(
		"--deterministic-worker-rng",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Remove PID/time from Worker RNG seeding for fully reproducible benchmarking.",
	)
	args = parser.parse_args(argv)

	if args.infilling_backend == "fast" and not _is_fast_infilling_available():
		raise RuntimeError(
			"Requested --infilling-backend=fast but birdie_rl.objectives._infilling_fast is not importable. "
			"Build it with: `python setup.py build_ext --inplace` (or install with a compiler)."
		)

	if args.compare_infilling_backends:
		if not _is_fast_infilling_available():
			print(
				"[benchmark] Warning: fast infilling backend not available; skipping comparison. "
				"Build it with: `python setup.py build_ext --inplace`.",
				flush=True,
			)
			return

		for seq_len in args.seq_lens:
			timeout_s = 30 if seq_len >= 16384 else 20
			max_calls = 600 if seq_len >= 16384 else 400
			python_timeout_s = max(timeout_s, 60) if seq_len >= 16384 else timeout_s
			python_max_calls = max_calls * 3 if seq_len >= 16384 else max_calls
			print(f"[benchmark] infilling backend=python seq_len={seq_len}", flush=True)
			py = bench_single_objective(
				objective_name="infilling",
				sequence_length=seq_len,
				batch_size=args.batch_size,
				target_batches=args.per_objective_batches,
				max_calls=python_max_calls,
				timeout_s=python_timeout_s,
				infilling_backend="python",
				infilling_candidates=args.infilling_candidates,
				deterministic_worker_rng=args.deterministic_worker_rng,
			)
			print(f"[benchmark] infilling backend=fast seq_len={seq_len}", flush=True)
			fast = bench_single_objective(
				objective_name="infilling",
				sequence_length=seq_len,
				batch_size=args.batch_size,
				target_batches=args.per_objective_batches,
				max_calls=max_calls,
				timeout_s=timeout_s,
				infilling_backend="fast",
				infilling_candidates=args.infilling_candidates,
				deterministic_worker_rng=args.deterministic_worker_rng,
			)
			speedup = fast["tokens_per_s"] / max(1e-9, py["tokens_per_s"])
			print(
				f"[benchmark] infilling speedup seq_len={seq_len} "
				f"python={py['tokens_per_s']:.0f} tok/s fast={fast['tokens_per_s']:.0f} tok/s "
				f"speedup={speedup:.2f}x",
				flush=True,
			)
		return

	for seq_len in args.seq_lens:
		timeout_s = 30 if seq_len >= 16384 else 20
		max_calls = 600 if seq_len >= 16384 else 400
		for objective in args.objectives:
			obj_timeout_s = timeout_s
			obj_max_calls = max_calls
			if objective == "infilling" and args.infilling_backend == "python" and seq_len >= 16384:
				obj_timeout_s = max(timeout_s, 60)
				obj_max_calls = max_calls * 3
			bench_single_objective(
				objective_name=objective,
				sequence_length=seq_len,
				batch_size=args.batch_size,
				target_batches=args.per_objective_batches,
				max_calls=obj_max_calls,
				timeout_s=obj_timeout_s,
				infilling_backend=args.infilling_backend,
				infilling_candidates=args.infilling_candidates,
				deterministic_worker_rng=args.deterministic_worker_rng,
			)

	for seq_len in args.seq_lens:
		timeout_s = 30 if seq_len >= 16384 else 20
		max_calls = 600 if seq_len >= 16384 else 400
		bench_mixed_objectives(
			objectives=args.objectives,
			sequence_length=seq_len,
			batch_size=args.batch_size,
			target_batches=args.mixed_batches,
			max_calls=max_calls,
			timeout_s=timeout_s,
			infilling_backend=args.infilling_backend,
			infilling_candidates=args.infilling_candidates,
			deterministic_worker_rng=args.deterministic_worker_rng,
		)


if __name__ == "__main__":
	main()
