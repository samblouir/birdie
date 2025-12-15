import unittest

import numpy as np

from birdie_rl.objectives.infilling import InfillingConfig, InfillingObjective


class CharTokenizer:
	"""
	Tokenizer for deterministic tests:
	- 1 token per character (ord)
	- decode is only used for unused_input_string; keep it simple.
	"""

	def encode(self, text: str):
		if not isinstance(text, str):
			raise TypeError(f"encode expected str, got {type(text)}")
		return [ord(c) for c in text]

	def decode(self, ids):
		if ids is None:
			return ""
		if isinstance(ids, np.ndarray):
			ids = ids.tolist()
		return "".join(chr(int(i)) for i in ids if isinstance(i, (int, np.integer)) and int(i) >= 0)


def _is_fast_backend_available() -> bool:
	try:
		from birdie_rl.objectives import infilling as infilling_mod
	except Exception:
		return False
	return getattr(infilling_mod, "_infilling_fast", None) is not None


def _run_infilling_once(*, use_fast_backend: bool, rng_seed: int, remaining_space: int, deterministic: bool, fast_backend_num_candidates: int = 1):
	tok = CharTokenizer()
	text = "a" * 5000

	cfg = InfillingConfig(
		tokenizer=tok,
		remaining_space=remaining_space,
		rng_seed=rng_seed,
		corruption_rate=0.15,
		mean_tokens_per_span=3.0,
		max_mask_spans=8,
		max_attempts=8,
		mask_prefix="<m",
		mask_suffix=">",
		paradigm="",
		paradigm_end="",
		use_fast_backend=use_fast_backend,
		fast_backend_num_candidates=fast_backend_num_candidates,
	)

	if deterministic:
		# Make the algorithm deterministic regardless of RNG implementation:
		# - span_len_to_mask always clamps to 1
		# - prob_to_mask always becomes 1, so masking always happens
		cfg.minimum_corruption_rate = 1.0
		cfg.maximum_corruption_rate = 1.0
		cfg.minimum_mean_tokens_per_span = 1.0
		cfg.maximum_mean_tokens_per_span = 1.0
		cfg.max_attempts = 1
		cfg.max_mask_spans = 4

	obj = InfillingObjective(cfg)
	return obj(text)


class TestInfillingBackends(unittest.TestCase):
	def test_python_backend_is_reproducible_per_seed(self):
		res1 = _run_infilling_once(use_fast_backend=False, rng_seed=123, remaining_space=2048, deterministic=False)
		res2 = _run_infilling_once(use_fast_backend=False, rng_seed=123, remaining_space=2048, deterministic=False)

		self.assertEqual(res1["status"], "ok")
		self.assertEqual(res2["status"], "ok")
		self.assertTrue(np.array_equal(res1["input_ids"], res2["input_ids"]))
		self.assertTrue(np.array_equal(res1["label_ids"], res2["label_ids"]))
		self.assertTrue(np.array_equal(res1["unused_input_ids"], res2["unused_input_ids"]))
		self.assertEqual(int(res1["masked_count"]), int(res2["masked_count"]))
		self.assertEqual(int(res1["original_length"]), int(res2["original_length"]))

	def test_python_backend_reseeds_on_config_change(self):
		tok = CharTokenizer()
		text = "a" * 5000
		cfg = InfillingConfig(
			tokenizer=tok,
			remaining_space=2048,
			rng_seed=1,
			corruption_rate=0.15,
			mean_tokens_per_span=3.0,
			max_mask_spans=8,
			max_attempts=8,
			mask_prefix="<m",
			mask_suffix=">",
			paradigm="",
			paradigm_end="",
			use_fast_backend=False,
		)
		obj = InfillingObjective(cfg)

		cfg.rng_seed = 7
		res_a = obj(text)
		cfg.rng_seed = 8
		res_b = obj(text)
		cfg.rng_seed = 7
		res_c = obj(text)

		self.assertEqual(res_a["status"], "ok")
		self.assertEqual(res_b["status"], "ok")
		self.assertEqual(res_c["status"], "ok")
		self.assertTrue(np.array_equal(res_a["input_ids"], res_c["input_ids"]))
		self.assertTrue(np.array_equal(res_a["label_ids"], res_c["label_ids"]))
		all_equal = (
			np.array_equal(res_a["input_ids"], res_b["input_ids"])
			and np.array_equal(res_a["label_ids"], res_b["label_ids"])
			and np.array_equal(res_a["unused_input_ids"], res_b["unused_input_ids"])
			and int(res_a["masked_count"]) == int(res_b["masked_count"])
			and int(res_a["original_length"]) == int(res_b["original_length"])
		)
		self.assertFalse(all_equal, "Expected config rng_seed change to change the output")

	@unittest.skipUnless(_is_fast_backend_available(), "C++ infilling backend not available")
	def test_fast_backend_is_reproducible_per_seed(self):
		res1 = _run_infilling_once(use_fast_backend=True, rng_seed=123, remaining_space=2048, deterministic=False)
		res2 = _run_infilling_once(use_fast_backend=True, rng_seed=123, remaining_space=2048, deterministic=False)

		self.assertEqual(res1["status"], "ok")
		self.assertEqual(res2["status"], "ok")
		self.assertTrue(np.array_equal(res1["input_ids"], res2["input_ids"]))
		self.assertTrue(np.array_equal(res1["label_ids"], res2["label_ids"]))
		self.assertTrue(np.array_equal(res1["unused_input_ids"], res2["unused_input_ids"]))
		self.assertEqual(int(res1["masked_count"]), int(res2["masked_count"]))
		self.assertEqual(int(res1["original_length"]), int(res2["original_length"]))

	@unittest.skipUnless(_is_fast_backend_available(), "C++ infilling backend not available")
	def test_fast_backend_multi_candidate_is_reproducible_per_seed(self):
		res1 = _run_infilling_once(
			use_fast_backend=True,
			fast_backend_num_candidates=5,
			rng_seed=123,
			remaining_space=2048,
			deterministic=False,
		)
		res2 = _run_infilling_once(
			use_fast_backend=True,
			fast_backend_num_candidates=5,
			rng_seed=123,
			remaining_space=2048,
			deterministic=False,
		)

		self.assertEqual(res1["status"], "ok")
		self.assertEqual(res2["status"], "ok")
		self.assertTrue(np.array_equal(res1["input_ids"], res2["input_ids"]))
		self.assertTrue(np.array_equal(res1["label_ids"], res2["label_ids"]))
		self.assertTrue(np.array_equal(res1["unused_input_ids"], res2["unused_input_ids"]))
		self.assertEqual(int(res1["masked_count"]), int(res2["masked_count"]))
		self.assertEqual(int(res1["original_length"]), int(res2["original_length"]))

	@unittest.skipUnless(_is_fast_backend_available(), "C++ infilling backend not available")
	def test_fast_backend_matches_python_when_deterministic(self):
		py = _run_infilling_once(use_fast_backend=False, rng_seed=0, remaining_space=512, deterministic=True)
		fast = _run_infilling_once(use_fast_backend=True, rng_seed=0, remaining_space=512, deterministic=True)

		self.assertEqual(py["status"], "ok")
		self.assertEqual(fast["status"], "ok")

		self.assertTrue(np.array_equal(py["input_ids"], fast["input_ids"]))
		self.assertTrue(np.array_equal(py["label_ids"], fast["label_ids"]))
		self.assertTrue(np.array_equal(py["unused_input_ids"], fast["unused_input_ids"]))
		self.assertEqual(int(py["masked_count"]), int(fast["masked_count"]))
		self.assertEqual(int(py["original_length"]), int(fast["original_length"]))

		remaining_space = 512
		self.assertLessEqual(len(py["input_ids"]) + len(py["label_ids"]), remaining_space)
