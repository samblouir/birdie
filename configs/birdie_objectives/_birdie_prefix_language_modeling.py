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
		Prefix Language Modeling
	'''
	objective_config = partial(
		PrefixLanguageModelingConfig,
	)

	# Non-overlapping sequence length ranges due to bug in paper.
	sequence_lengths = [
		(-1, -1),
	]


	'''
		Prepare configurations
	'''
	configured_objectives = {}
	for (minimum_sequence_length, maximum_sequence_length) in sequence_lengths:
			
			objective_name = "Prefix Language Modeling"
			

			key = '_'.join([
				objective_name,
				# f"{minimum_sequence_length}-{maximum_sequence_length}l"
			])

			configured_objective = dict(
				config=objective_config(
					objective=objective_name,
					minimum_sequence_length=minimum_sequence_length,
					maximum_sequence_length=maximum_sequence_length,
				),
				sampling_probability=1,
			)

			(configured_objective, message) = post_process(configured_objective)
			if not is_valid_configuration(configured_objective, message):
				continue

			(configured_objective, message) = filter_fn(configured_objective)
			if not is_valid_configuration(configured_objective, message):
				continue


			configured_objectives[key] = configured_objective
			
	return configured_objectives


def filter_fn(x):

	return (x, "Valid Configuration")



def post_process(x):
	'''
		Post-processes the configuration.
	'''

	return (x, "Valid Configuration")