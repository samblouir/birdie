import numpy as np
from functools import partial
import copy
from objective_configs import (
	NextTokenPredictionConfig,
	PrefixLanguageModelingConfig,
	InfillingConfig,
	AutoencodingConfig,
	CopyingConfig,
	DeshufflingConfig,
	SelectiveCopyingConfig,
)
from birdie_objectives._birdie_utils import is_valid_configuration



def get_objectives(config=None):

	'''
		Selective Copying
	'''
	selective_copying_config = partial(
		SelectiveCopyingConfig,
	)

	# Non-overlapping sequence length ranges due to bug in paper.
	sequence_lengths = [
		(384, 768),
		(768, 1536),
	]

	num_spans_choices = [
		1,
		2,
		3,
		4,
	]

	formatting_types = [
		"query_context",
		"context_query",
	]

	paradigm_function_strs = [
		"[COPY]",
	]

	paradigm_start_span_strs = [
		"[START]",
	]

	paradigm_end_span_strs = [
		"[END]",
	]

	paradigm_context_strs = [
		"[CONTEXT]",
	]

	paradigm_done_strs = [
		"[DONE]",
	]

	paradigm_sep_strs = [
		"[SEP]",
	]


	'''
		Prepare Selective Copying configurations
	'''
	configured_objectives = {}

	for (selective_copying_minimum_sequence_length, selective_copying_maximum_sequence_length) in sequence_lengths:
		for num_spans in num_spans_choices:
			for formatting_type in formatting_types:
				for paradigm_function_str in paradigm_function_strs:
					for paradigm_start_span_str in paradigm_start_span_strs:
						for paradigm_end_span_str in paradigm_end_span_strs:
							for paradigm_context_str in paradigm_context_strs:
								for paradigm_done_str in paradigm_done_strs:
									for paradigm_sep_str in paradigm_sep_strs:

								
								

										selective_copying_key = '_'.join([
											"selective_copying",
											f"{selective_copying_minimum_sequence_length}-{selective_copying_maximum_sequence_length}L",
											f"{num_spans}S",
											f"{formatting_type}F",

										])

										configured_selective_copying_objective = dict(
											config=selective_copying_config(
												objective="selective_copying",
												num_spans=num_spans,
												formatting_type=formatting_type,
												minimum_sequence_length=selective_copying_minimum_sequence_length,
												maximum_sequence_length=selective_copying_maximum_sequence_length,
												paradigm_function_str=paradigm_function_str,
												paradigm_start_span_str=paradigm_start_span_str,
												paradigm_end_span_str=paradigm_end_span_str,
												paradigm_context_str=paradigm_context_str,
												paradigm_done_str=paradigm_done_str,
												paradigm_sep_str=paradigm_sep_str,
											),
											sampling_probability=1,
										)

										(configured_selective_copying_objective, message) = selective_copying_post_process(configured_selective_copying_objective)
										if not is_valid_configuration(configured_selective_copying_objective, message):
											continue

										(configured_selective_copying_objective, message) = selective_copying_filter_fn(configured_selective_copying_objective)
										if not is_valid_configuration(configured_selective_copying_objective, message):
											continue


										configured_objectives[selective_copying_key] = configured_selective_copying_objective
						
	return configured_objectives


def selective_copying_filter_fn(x):

	return (x, "Valid Configuration")



def selective_copying_post_process(x):
	'''
		Post-processes the configuration.
	'''

	return (x, "Valid Configuration")