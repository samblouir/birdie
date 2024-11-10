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
)
from birdie_objectives._birdie_utils import is_valid_configuration



def get_objectives(config=None):

	'''
		Deshuffling
	'''
	objective_config = partial(
		DeshufflingConfig,
	)

	# Non-overlapping sequence length ranges due to bug in paper.
	sequence_lengths = [
		(128, 256),
		(512, 1024),
	]

	average_shuffled_percentages = [
		0.5,
		1.0,
	]


	'''
		Prepare Deshuffling configurations
	'''
	configured_objectives = {}
	for (deshuffling_minimum_sequence_length, deshuffling_maximum_sequence_length) in sequence_lengths:
			for average_shuffled_percentage in average_shuffled_percentages:
			

				deshuffling_key = '_'.join([
					"deshuffling",
					f"{deshuffling_minimum_sequence_length}-{deshuffling_maximum_sequence_length}l",
					f"{int(average_shuffled_percentage * 100)}p",
				])

				configured_deshuffling_objective = dict(
					config=objective_config(
						objective="deshuffling",
						minimum_sequence_length=deshuffling_minimum_sequence_length,
						maximum_sequence_length=deshuffling_maximum_sequence_length,
					),
					sampling_probability=1,
				)

				(configured_deshuffling_objective, message) = deshuffling_post_process(configured_deshuffling_objective)
				if not is_valid_configuration(configured_deshuffling_objective, message):
					continue

				(configured_deshuffling_objective, message) = deshuffling_filter_fn(configured_deshuffling_objective)
				if not is_valid_configuration(configured_deshuffling_objective, message):
					continue


				configured_objectives[deshuffling_key] = configured_deshuffling_objective
			
	return configured_objectives


def deshuffling_filter_fn(x):

	return (x, "Valid Configuration")



def deshuffling_post_process(x):
	'''
		Post-processes the configuration.
	'''

	return (x, "Valid Configuration")