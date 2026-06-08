from birdie_rl.tests.pipeline_benchmark_helpers import collect_pipeline_benchmark


def test_pipeline_benchmark_reports_per_objective_statistics():
	commanded_rates = {
		"next_token_prediction": 0.30,
		"copying": 0.20,
		"prefix_language_modeling": 0.20,
		"autoencoding": 0.20,
		"infilling": 0.10,
	}

	result = collect_pipeline_benchmark(
		commanded_rates=commanded_rates,
		sequence_length=1024,
		batch_size=8,
		num_batches=24,
		text_lengths=(64, 256, 1024, 4096),
		next_timeout_s=10.0,
	)

	assert result["unknown_segments"] == 0
	assert result["total_samples"] >= 24 * 8
	assert result["batches_per_s"] > 2.0
	assert result["active_tokens_per_s"] > 10_000

	for objective_name, stats in result["per_objective"].items():
		assert stats["sample_count"] > 0, objective_name
		assert stats["context_tokens"] > 0, objective_name
		assert stats["input_tokens"] > 0, objective_name
		assert stats["output_tokens"] > 0, objective_name
		assert stats["context_tokens"] <= stats["input_tokens"], objective_name
		assert stats["total_tokens"] == stats["input_tokens"] + stats["output_tokens"]
		assert abs(stats["actual_rate"] - stats["commanded_rate"]) <= 0.20, objective_name

	ntp_stats = result["per_objective"]["next_token_prediction"]
	assert ntp_stats["context_tokens"] == ntp_stats["sample_count"]
	assert ntp_stats["avg_context_tokens"] == 1.0


def test_pipeline_benchmark_large_sequence_length_and_batch_size():
	result = collect_pipeline_benchmark(
		commanded_rates={
			"next_token_prediction": 0.40,
			"copying": 0.25,
			"prefix_language_modeling": 0.20,
			"autoencoding": 0.15,
		},
		sequence_length=16_384,
		batch_size=64,
		num_batches=2,
		text_lengths=(512, 4096, 16_384, 32_768),
		next_timeout_s=45.0,
	)

	assert result["sequence_length"] == 16_384
	assert result["batch_size"] == 64
	assert result["unknown_segments"] == 0
	assert result["total_samples"] >= 64
	assert result["fill_rate"] > 0.50
	assert result["token_slots_per_s"] > 50_000

	for objective_name, stats in result["per_objective"].items():
		assert stats["sample_count"] > 0, objective_name
		assert stats["context_tokens"] > 0, objective_name
		assert stats["context_tokens"] <= stats["input_tokens"], objective_name
		assert stats["total_tokens"] > 0, objective_name

	ntp_stats = result["per_objective"]["next_token_prediction"]
	assert ntp_stats["context_tokens"] == ntp_stats["sample_count"]
