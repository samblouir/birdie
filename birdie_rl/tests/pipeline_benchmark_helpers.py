import argparse
import json
import queue
import threading
import time
from collections import OrderedDict

import numpy as np

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


OBJECTIVE_SPECS = OrderedDict(
	[
		(
			"next_token_prediction",
			{
				"marker": "A",
				"config_overrides": {"paradigm": "A"},
			},
		),
		(
			"copying",
			{
				"marker": "B",
				"config_overrides": {"paradigm": "B"},
			},
		),
		(
			"prefix_language_modeling",
			{
				"marker": "C",
				"config_overrides": {"paradigm_str": "C", "prefix_fraction": 0.5},
			},
		),
		(
			"autoencoding",
			{
				"marker": "D",
				"config_overrides": {
					"paradigm_prompt": "D",
					"corruption_rate": 0.85,
					"tokens_per_mask": 4,
					"max_attempts": 100,
				},
			},
		),
		(
			"infilling",
			{
				"marker": "E",
				"config_overrides": {
					"paradigm": "E",
					"corruption_rate": 0.85,
					"mean_tokens_per_span": 4.0,
					"max_attempts": 100,
				},
			},
		),
	]
)


def variable_length_data_generator(
	split=None,
	worker_id=0,
	num_workers=1,
	rng_seed=0,
	text_lengths=(128, 512, 2048),
):
	base_text = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 "
	idx = worker_id
	while True:
		length = int(text_lengths[idx % len(text_lengths)])
		repeated = base_text * ((length // len(base_text)) + 1)
		yield {"text": repeated[:length]}
		idx += max(1, num_workers)


def build_objectives(commanded_rates):
	total = float(sum(commanded_rates.values()))
	if total <= 0:
		raise ValueError("commanded_rates must sum to a positive value")

	objectives = []
	for objective_name, raw_rate in commanded_rates.items():
		spec = OBJECTIVE_SPECS[objective_name]
		objectives.append(
			{
				"name": objective_name,
				"prob": float(raw_rate) / total,
				"config_overrides": dict(spec["config_overrides"]),
			}
		)
	return objectives


def _objective_prefix_len(objective_name, tokenizer):
	config_overrides = OBJECTIVE_SPECS[objective_name]["config_overrides"]
	if objective_name == "next_token_prediction":
		prefix = config_overrides.get("paradigm", "")
	elif objective_name == "copying":
		prefix = config_overrides.get("paradigm", "")
	elif objective_name == "prefix_language_modeling":
		prefix = config_overrides.get("paradigm_str", "")
	elif objective_name == "autoencoding":
		prefix = config_overrides.get("paradigm_prompt", "")
	elif objective_name == "infilling":
		prefix = config_overrides.get("paradigm", "")
	else:
		prefix = ""
	return len(tokenizer.encode(prefix)) if prefix else 0


def _context_token_count(objective_name, input_tokens, tokenizer):
	if objective_name == "next_token_prediction":
		return min(input_tokens, _objective_prefix_len(objective_name, tokenizer))
	return input_tokens


def _next_with_timeout(generator, timeout_s=10.0):
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


def _ordered_segment_starts(segment_ids):
	segment_values = []
	for segment_id in segment_ids:
		segment_id = int(segment_id)
		if segment_id > 0 and (not segment_values or segment_values[-1] != segment_id):
			segment_values.append(segment_id)
	return segment_values


def collect_pipeline_benchmark(
	*,
	commanded_rates,
	sequence_length,
	batch_size,
	num_batches,
	text_lengths,
	num_workers=1,
	next_timeout_s=20.0,
):
	tokenizer = CharTokenizer()
	objectives_config = build_objectives(commanded_rates)
	marker_to_objective = {
		ord(spec["marker"]) + 3: objective_name
		for objective_name, spec in OBJECTIVE_SPECS.items()
		if objective_name in commanded_rates
	}
	commanded_total = float(sum(commanded_rates.values()))
	normalized_commanded_rates = {
		objective_name: float(rate) / commanded_total
		for objective_name, rate in commanded_rates.items()
	}

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
				"tokenizer": tokenizer,
				"min_seq_len_for_packing": max(4, sequence_length // 16),
				"seed": 12345,
				"max_consecutive_failed_samples": 1000,
			},
		)

		start_time = time.perf_counter()
		batches = [_next_with_timeout(generator, timeout_s=next_timeout_s) for _ in range(num_batches)]
		elapsed_s = time.perf_counter() - start_time
	finally:
		if controller is not None:
			controller.close(timeout_join=3.0)

	per_objective = {
		objective_name: {
			"commanded_rate": normalized_commanded_rates[objective_name],
			"sample_count": 0,
			"context_tokens": 0,
			"input_tokens": 0,
			"output_tokens": 0,
			"total_tokens": 0,
		}
		for objective_name in commanded_rates
	}
	unknown_segments = 0
	active_token_slots = 0

	for batch in batches:
		input_ids = batch["input_ids"]
		label_ids = batch["label_ids"]
		segment_ids = batch["segment_ids"]
		active_token_slots += int(np.count_nonzero(segment_ids > 0))

		for row_input_ids, row_label_ids, row_segment_ids in zip(input_ids, label_ids, segment_ids):
			for segment_value in _ordered_segment_starts(row_segment_ids):
				positions = np.flatnonzero(row_segment_ids == segment_value)
				if positions.size == 0:
					continue

				objective_name = marker_to_objective.get(int(row_input_ids[positions[0]]))
				if objective_name is None:
					unknown_segments += 1
					continue

				output_tokens = int(np.count_nonzero(row_label_ids[positions] != -100))
				total_tokens = int(positions.size)
				input_tokens = total_tokens - output_tokens
				context_tokens = _context_token_count(objective_name, input_tokens, tokenizer)
				stats = per_objective[objective_name]
				stats["sample_count"] += 1
				stats["context_tokens"] += context_tokens
				stats["input_tokens"] += input_tokens
				stats["output_tokens"] += output_tokens
				stats["total_tokens"] += total_tokens

	total_samples = sum(stats["sample_count"] for stats in per_objective.values())
	for stats in per_objective.values():
		sample_count = stats["sample_count"]
		stats["actual_rate"] = (sample_count / total_samples) if total_samples else 0.0
		stats["avg_context_tokens"] = (stats["context_tokens"] / sample_count) if sample_count else 0.0
		stats["avg_input_tokens"] = (stats["input_tokens"] / sample_count) if sample_count else 0.0
		stats["avg_output_tokens"] = (stats["output_tokens"] / sample_count) if sample_count else 0.0
		stats["avg_total_tokens"] = (stats["total_tokens"] / sample_count) if sample_count else 0.0

	total_token_slots = int(num_batches * batch_size * sequence_length)
	return {
		"sequence_length": int(sequence_length),
		"batch_size": int(batch_size),
		"num_batches": int(num_batches),
		"num_workers": int(num_workers),
		"text_lengths": list(text_lengths),
		"elapsed_s": elapsed_s,
		"batches_per_s": num_batches / elapsed_s if elapsed_s else 0.0,
		"samples_per_s": total_samples / elapsed_s if elapsed_s else 0.0,
		"active_tokens_per_s": active_token_slots / elapsed_s if elapsed_s else 0.0,
		"token_slots_per_s": total_token_slots / elapsed_s if elapsed_s else 0.0,
		"fill_rate": active_token_slots / total_token_slots if total_token_slots else 0.0,
		"total_samples": total_samples,
		"active_token_slots": active_token_slots,
		"total_token_slots": total_token_slots,
		"unknown_segments": unknown_segments,
		"per_objective": per_objective,
	}


def format_benchmark_markdown(result):
	lines = [
		f"sequence_length={result['sequence_length']} batch_size={result['batch_size']} "
		f"num_batches={result['num_batches']} elapsed_s={result['elapsed_s']:.3f}",
		f"batches/s={result['batches_per_s']:.2f} samples/s={result['samples_per_s']:.2f} "
		f"active_tokens/s={result['active_tokens_per_s']:.0f} token_slots/s={result['token_slots_per_s']:.0f} "
		f"fill_rate={result['fill_rate']:.2%}",
		"",
		"| objective | commanded | actual | samples | context tokens | input tokens | output tokens | total tokens | avg ctx | avg in | avg out | avg total |",
		"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for objective_name, stats in result["per_objective"].items():
		lines.append(
			"| {objective} | {commanded:.2%} | {actual:.2%} | {samples} | {context_tokens} | "
			"{input_tokens} | {output_tokens} | {total_tokens} | {avg_ctx:.1f} | "
			"{avg_in:.1f} | {avg_out:.1f} | {avg_total:.1f} |".format(
				objective=objective_name,
				commanded=stats["commanded_rate"],
				actual=stats["actual_rate"],
				samples=stats["sample_count"],
				context_tokens=stats["context_tokens"],
				input_tokens=stats["input_tokens"],
				output_tokens=stats["output_tokens"],
				total_tokens=stats["total_tokens"],
				avg_ctx=stats["avg_context_tokens"],
				avg_in=stats["avg_input_tokens"],
				avg_out=stats["avg_output_tokens"],
				avg_total=stats["avg_total_tokens"],
			)
		)
	return "\n".join(lines)


def main():
	parser = argparse.ArgumentParser(description="Run Birdie pipeline benchmark statistics.")
	parser.add_argument("--sequence-length", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--num-batches", type=int, default=20)
	parser.add_argument("--num-workers", type=int, default=1)
	parser.add_argument("--text-lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
	parser.add_argument("--json", action="store_true")
	args = parser.parse_args()

	commanded_rates = {
		"next_token_prediction": 0.30,
		"copying": 0.20,
		"prefix_language_modeling": 0.20,
		"autoencoding": 0.20,
		"infilling": 0.10,
	}
	result = collect_pipeline_benchmark(
		commanded_rates=commanded_rates,
		sequence_length=args.sequence_length,
		batch_size=args.batch_size,
		num_batches=args.num_batches,
		num_workers=args.num_workers,
		text_lengths=args.text_lengths,
	)

	if args.json:
		print(json.dumps(result, indent=2, sort_keys=True))
	else:
		print(format_benchmark_markdown(result))


if __name__ == "__main__":
	main()
