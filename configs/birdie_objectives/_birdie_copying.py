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
		Copying
	'''
	objective_config = partial(
		CopyingConfig,
	)

	# Non-overlapping sequence length ranges due to bug in paper.
	sequence_lengths = [
		(128, 256),
		(512, 1024),
	]


	'''
		Prepare Copying configurations
	'''
	configured_objectives = {}
	for (copying_minimum_sequence_length, copying_maximum_sequence_length) in sequence_lengths:
			

			copying_key = '_'.join([
				"copying",
				f"{copying_minimum_sequence_length}-{copying_maximum_sequence_length}l"
			])

			configured_copying_objective = dict(
				config=objective_config(
					objective="copying",
					minimum_sequence_length=copying_minimum_sequence_length,
					maximum_sequence_length=copying_maximum_sequence_length,
				),
				sampling_probability=1,
			)

			(configured_copying_objective, message) = copying_post_process(configured_copying_objective)
			if not is_valid_configuration(configured_copying_objective, message):
				continue

			(configured_copying_objective, message) = copying_filter_fn(configured_copying_objective)
			if not is_valid_configuration(configured_copying_objective, message):
				continue


			configured_objectives[copying_key] = configured_copying_objective
			
	return configured_objectives


def copying_filter_fn(x):

	return (x, "Valid Configuration")



def copying_post_process(x):
	'''
		Post-processes the configuration.
	'''

	return (x, "Valid Configuration")