import itertools

import numpy as np
import pytest

from birdie_rl.pipeline_generator import pipeline_data_generator
from birdie_rl.tests.pipeline_benchmark_helpers import CharTokenizer, _next_with_timeout


def text_data_generator(split=None, worker_id=0, num_workers=1, rng_seed=0, texts=("sample text",)):
	idx = worker_id
	while True:
		yield {"text": texts[idx % len(texts)]}
		idx += max(1, num_workers)


def empty_text_data_generator(split=None, worker_id=0, num_workers=1, rng_seed=0):
	while True:
		yield {"text": ""}


def missing_text_data_generator(split=None, worker_id=0, num_workers=1, rng_seed=0):
	while True:
		yield {"not_text": "this should be rejected by the default text grabber"}


def finite_text_data_generator(split=None, worker_id=0, num_workers=1, rng_seed=0, texts=("finite sample",)):
	for text in itertools.islice(texts, worker_id, None, max(1, num_workers)):
		yield {"text": text}


def constant_bad_type_generator(split=None, worker_id=0, num_workers=1, rng_seed=0):
	while True:
		yield object()


def _open_pipeline(
	*,
	objectives_config,
	data_generator=text_data_generator,
	sequence_length=64,
	batch_size=2,
	num_workers=1,
	infinite_loop=True,
	config=None,
	data_generator_fn_kwarg_overrides=None,
):
	merged_config = {
		"tokenizer": CharTokenizer(),
		"min_seq_len_for_packing": max(4, sequence_length // 8),
		"seed": 999,
		"max_consecutive_failed_samples": 8,
	}
	if config:
		merged_config.update(config)

	return pipeline_data_generator(
		max_batches=4,
		batch_size=batch_size,
		sequence_length=sequence_length,
		num_workers=num_workers,
		objectives_config=objectives_config,
		data_generator=data_generator,
		data_generator_fn_kwarg_overrides=data_generator_fn_kwarg_overrides or {},
		infinite_loop=infinite_loop,
		config=merged_config,
	)


def _assert_pipeline_raises_runtime_error(**kwargs):
	controller = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = _open_pipeline(**kwargs)
		with pytest.raises(RuntimeError):
			_next_with_timeout(generator, timeout_s=6.0)
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)


def _assert_pipeline_stops_without_yielding(**kwargs):
	controller = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = _open_pipeline(**kwargs)
		with pytest.raises(StopIteration):
			_next_with_timeout(generator, timeout_s=6.0)
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)


def test_missing_data_generator_fails_immediately():
	with pytest.raises(ValueError, match="data_generator"):
		pipeline_data_generator(config={"tokenizer": CharTokenizer()})


def test_missing_tokenizer_fails_immediately():
	with pytest.raises(KeyError, match="tokenizer"):
		pipeline_data_generator(data_generator=text_data_generator, config={})


def test_unknown_objective_hard_fails_instead_of_spinning_forever():
	_assert_pipeline_raises_runtime_error(
		objectives_config=[{"name": "not_a_real_objective", "prob": 1.0}],
		config={"max_consecutive_failed_samples": 3},
	)


@pytest.mark.parametrize("data_generator", [empty_text_data_generator, missing_text_data_generator, constant_bad_type_generator])
def test_bad_or_empty_data_hard_fails_instead_of_spinning_forever(data_generator):
	_assert_pipeline_raises_runtime_error(
		objectives_config=[{"name": "next_token_prediction", "prob": 1.0}],
		data_generator=data_generator,
		config={"max_consecutive_failed_samples": 4},
	)


def test_all_unusable_objective_mixture_hard_fails_instead_of_spinning_forever():
	_assert_pipeline_raises_runtime_error(
		objectives_config=[
			{
				"name": "selective_copying",
				"prob": 0.5,
				"config_overrides": {
					"min_delimiter_prefix_length": 256,
					"min_delimiter_suffix_length": 256,
					"min_context_content_length": 256,
					"max_attempts": 1,
				},
			},
			{
				"name": "prefix_language_modeling",
				"prob": 0.5,
				"config_overrides": {
					"paradigm_str": "this prompt is deliberately too long for the sequence",
					"minimum_remaining_space": 1024,
				},
			},
		],
		data_generator_fn_kwarg_overrides={"texts": ("tiny", "short")},
		sequence_length=32,
		config={"max_consecutive_failed_samples": 6},
	)


def test_mostly_failing_mixture_hard_fails_instead_of_falling_back_to_viable_objective():
	controller = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = _open_pipeline(
			objectives_config=[
				{
					"name": "selective_copying",
					"prob": 0.9,
					"config_overrides": {
						"min_delimiter_prefix_length": 256,
						"min_delimiter_suffix_length": 256,
						"min_context_content_length": 256,
						"max_attempts": 1,
					},
				},
				{
					"name": "next_token_prediction",
					"prob": 0.1,
					"config_overrides": {"paradigm": "N"},
				},
			],
			data_generator_fn_kwarg_overrides={"texts": ("abcdefghijklmnopqrstuvwxyz " * 4,)},
			sequence_length=96,
			batch_size=2,
			config={"max_consecutive_failed_samples": 200},
		)
		with pytest.raises(RuntimeError):
			_next_with_timeout(generator, timeout_s=8.0)
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)


def test_finite_dataset_with_no_complete_packed_sequence_stops_without_yielding():
	_assert_pipeline_stops_without_yielding(
		objectives_config=[
			{
				"name": "next_token_prediction",
				"prob": 1.0,
				"config_overrides": {"paradigm": "N"},
			}
		],
		data_generator=finite_text_data_generator,
		data_generator_fn_kwarg_overrides={"texts": ("alpha beta gamma delta " * 4,)},
		sequence_length=96,
		batch_size=4,
		infinite_loop=False,
	)


def test_finite_dataset_with_enough_tokens_yields_batches_then_stops_cleanly():
	controller = None
	try:
		controller, generator, _thread, _batcher_stop, _datagen_stop = _open_pipeline(
			objectives_config=[
				{
					"name": "next_token_prediction",
					"prob": 1.0,
					"config_overrides": {"paradigm": "N"},
				}
			],
			data_generator=finite_text_data_generator,
			data_generator_fn_kwarg_overrides={"texts": ("alpha beta gamma delta " * 20,)},
			sequence_length=96,
			batch_size=4,
			infinite_loop=False,
		)
		batch = _next_with_timeout(generator, timeout_s=6.0)
		assert batch["input_ids"].shape == (4, 96)
		assert int(np.count_nonzero(batch["label_ids"] != -100)) > 0

		for _ in range(10):
			try:
				_next_with_timeout(generator, timeout_s=6.0)
			except StopIteration:
				break
		else:
			pytest.fail("finite pipeline kept yielding after the finite source should have been exhausted")
	finally:
		if controller is not None:
			controller.close(timeout_join=2.0)
