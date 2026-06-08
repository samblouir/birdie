import queue
import threading
import time

import numpy as np
import pytest

from birdie_rl.pipeline_generator import pipeline_data_generator


class CharTokenizer:
	def encode(self, text):
		if isinstance(text, list):
			return [self.encode(item) for item in text]
		return [ord(char) + 3 for char in text]

	def decode(self, token_ids):
		if isinstance(token_ids, np.ndarray):
			token_ids = token_ids.tolist()
		if not token_ids:
			return ""
		if isinstance(token_ids[0], list):
			return [self.decode(item) for item in token_ids]
		return "".join(chr(int(token_id) - 3) for token_id in token_ids if int(token_id) >= 3)


def variable_length_data_generator(
	split=None,
	worker_id=0,
	num_workers=1,
	rng_seed=0,
	text_lengths=(32, 128, 512),
):
	base_text = "abcdefghijklmnopqrstuvwxyz "
	idx = worker_id
	while True:
		length = int(text_lengths[idx % len(text_lengths)])
		repeated = base_text * ((length // len(base_text)) + 1)
		yield {"text": repeated[:length]}
		idx += max(1, num_workers)


def _next_with_timeout(generator, timeout_s=5.0):
	result_q = queue.Queue(maxsize=1)

	def run_next():
		try:
			result_q.put(("ok", next(generator)))
		except BaseException as exc:
			result_q.put(("err", exc))

	thread = threading.Thread(target=run_next, daemon=True)
	thread.start()
	thread.join(timeout_s)
	if thread.is_alive():
		raise TimeoutError(f"pipeline did not yield within {timeout_s:.1f}s")

	status, payload = result_q.get_nowait()
	if status == "err":
		raise payload
	return payload


def _collect_batches(*, objectives_config, text_lengths, num_batches, sequence_length=128, batch_size=4, num_workers=1):
	controller = None
	generator = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = pipeline_data_generator(
			max_batches=num_batches,
			batch_size=batch_size,
			sequence_length=sequence_length,
			num_workers=num_workers,
			objectives_config=objectives_config,
			data_generator=variable_length_data_generator,
			data_generator_fn_kwarg_overrides={"text_lengths": tuple(text_lengths)},
			config={
				"tokenizer": CharTokenizer(),
				"min_seq_len_for_packing": max(4, sequence_length // 8),
				"seed": 12345,
				"max_consecutive_failed_samples": 200,
			},
		)
		start = time.perf_counter()
		batches = [_next_with_timeout(generator) for _ in range(num_batches)]
		elapsed = time.perf_counter() - start
		return batches, elapsed
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)


def _segment_start_tokens(batch):
	start_tokens = []
	input_ids = batch["input_ids"]
	segment_ids = batch["segment_ids"]

	for row_input_ids, row_segment_ids in zip(input_ids, segment_ids):
		active = row_segment_ids > 0
		start_mask = active.copy()
		start_mask[1:] &= row_segment_ids[1:] != row_segment_ids[:-1]
		start_positions = np.flatnonzero(start_mask)
		start_tokens.extend(int(row_input_ids[pos]) for pos in start_positions)

	return start_tokens


def test_pipeline_many_batches_tracks_commanded_mixture_rate():
	ntp_marker = ord("N") + 3
	copy_marker = ord("C") + 3
	num_batches = 80
	commanded_ntp_rate = 0.70

	batches, elapsed = _collect_batches(
		num_batches=num_batches,
		batch_size=4,
		sequence_length=128,
		text_lengths=(24, 48, 96, 256, 1024),
		objectives_config=[
			{
				"name": "next_token_prediction",
				"prob": commanded_ntp_rate,
				"config_overrides": {"paradigm": "N"},
			},
			{
				"name": "copying",
				"prob": 1.0 - commanded_ntp_rate,
				"config_overrides": {"paradigm": "C"},
			},
		],
	)

	start_tokens = [token for batch in batches for token in _segment_start_tokens(batch)]
	recognized = [token for token in start_tokens if token in {ntp_marker, copy_marker}]
	assert len(recognized) >= num_batches * 2

	observed_ntp_rate = recognized.count(ntp_marker) / len(recognized)
	assert abs(observed_ntp_rate - commanded_ntp_rate) <= 0.12
	assert num_batches / elapsed >= 20.0


def test_pipeline_keeps_throughput_with_varied_input_sizes():
	num_batches = 50
	batches, elapsed = _collect_batches(
		num_batches=num_batches,
		batch_size=4,
		sequence_length=160,
		text_lengths=(16, 32, 64, 512, 2048),
		objectives_config=[
			{
				"name": "next_token_prediction",
				"prob": 0.60,
				"config_overrides": {"paradigm": "N"},
			},
			{
				"name": "copying",
				"prob": 0.40,
				"config_overrides": {"paradigm": "C"},
			},
		],
	)

	assert all(batch["input_ids"].shape == (4, 160) for batch in batches)
	assert sum(int(np.count_nonzero(batch["label_ids"] != -100)) for batch in batches) > 0
	assert num_batches / elapsed >= 10.0


def test_unusable_objective_configuration_terminates_instead_of_hanging():
	controller = None
	generator = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = pipeline_data_generator(
			max_batches=1,
			batch_size=2,
			sequence_length=64,
			num_workers=1,
			objectives_config=[
				{
					"name": "selective_copying",
					"prob": 1.0,
					"config_overrides": {
						"min_delimiter_prefix_length": 64,
						"min_delimiter_suffix_length": 64,
						"min_context_content_length": 64,
						"max_attempts": 2,
					},
				}
			],
			data_generator=variable_length_data_generator,
			data_generator_fn_kwarg_overrides={"text_lengths": (8, 12, 16)},
			config={
				"tokenizer": CharTokenizer(),
				"min_seq_len_for_packing": 8,
				"seed": 222,
				"max_consecutive_failed_samples": 10,
			},
		)

		with pytest.raises(RuntimeError):
			_next_with_timeout(generator, timeout_s=5.0)
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)
