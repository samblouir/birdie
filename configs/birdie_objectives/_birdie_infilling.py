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



def get_infilling_objectives(config):

	'''
	Infilling
	'''
	objective_config = partial(
		InfillingConfig,
		mask_token_ids = config['infilling_mask_token_ids'],
	)

	infilling_corruption_rates = [
		0.05,
		0.15,
		0.5,
	]

	infilling_span_widths = [
		3,
		8,
		32,
	]

	infilling_lengths = [
		(128, 256),
		(256, 512),
		(512, 1024),
		(1024, 2048),
	]


	'''
		Prepare Infilling configurations
	'''
	infilling_objectives = {}
	for infilling_corruption_rate in infilling_corruption_rates:
		for infilling_mean_span_width in infilling_span_widths:
			for (infilling_minimum_sequence_length, infilling_maximum_sequence_length) in infilling_lengths:
				

				infilling_key = '_'.join([
					"infilling"
					f"{infilling_mean_span_width}w",
					f"{int(infilling_corruption_rate * 100)}p",
					f"{infilling_minimum_sequence_length}-{infilling_maximum_sequence_length}l"
				])

				configured_infilling_objective = dict(
					config=objective_config(
						objective="Infilling",
						corruption_rate=infilling_corruption_rate,
						mean_span_width=infilling_mean_span_width,
						minimum_sequence_length=infilling_minimum_sequence_length,
						maximum_sequence_length=infilling_maximum_sequence_length,
					),
					sampling_probability=1,
				)

				(configured_infilling_objective, message) = infilling_post_process(configured_infilling_objective)
				if not is_valid_configuration(configured_infilling_objective, message):
					continue

				(configured_infilling_objective, message) = infilling_filter_fn(configured_infilling_objective)
				if not is_valid_configuration(configured_infilling_objective, message):
					continue


				infilling_objectives[infilling_key] = configured_infilling_objective
				
	return infilling_objectives




def infilling_filter_fn(x):
	'''
	'''
	config = x['config']

	if (config.maximum_sequence_length <= config.minimum_sequence_length):
		message = ' '.join([
			f"Invalid Infilling Configuration: ",
			f"sequence length range: (max <= min) ({config.maximum_sequence_length} <= {config.minimum_sequence_length})"
		])
		return (x, message)
	
	if (0.05 <= config.corruption_rate) and (config.mean_span_width <= 3) and (256 < config.maximum_sequence_length):
		message =  ' '.join([
			f"Invalid Infilling Configuration: ",
			f"(0.05 <= {config.corruption_rate} (corruption_rate))",
			f"and (mean_span_width <= {config.mean_span_width} (mean_span_width))",
			f"and (256 < {config.maximum_sequence_length} (maximum_sequence_length))",
		])
		return (x, message)
	
	return (x, "Valid Configuration")



def infilling_post_process(x):
	'''
		Post-processes the configuration.
	'''

	config = x['config']
	infilling_minimum_corruption_percentage = np.floor(config.corruption_rate * 0.25)
	infilling_maximum_corruption_percentage = np.ceil(config.corruption_rate * 2.0)

	infilling_calculated_minimum_sequence_length = np.floor(config.mean_span_width / config.corruption_rate / 2)

	# If the calculated minimum sequence length is already greater than the maximum sequence length, then we just ignore this configuration.
	if (config.maximum_sequence_length < infilling_calculated_minimum_sequence_length):
		message = ' '.join([
			f"Invalid Infilling Configuration: ",
			f"Calculated minimum sequence length ({infilling_calculated_minimum_sequence_length}) is greater than the maximum sequence length ({config.maximum_sequence_length}).",
		])
		return (x, message)


	# If there isn't enough space between our minimum and maximuze sequence lengths, then we ignore this configuration.
	allowed_minimum_sequence_length_hysteresis = (config.maximum_sequence_length - config.minimum_sequence_length)//4
	proposed_minimum_sequence_length_gap = (config.maximum_sequence_length - infilling_calculated_minimum_sequence_length)
	if (proposed_minimum_sequence_length_gap < allowed_minimum_sequence_length_hysteresis):
		message = ' '.join([
			f"Invalid Infilling Configuration: ",
			f"Proposed minimum sequence length gap is too small.",
			f"(proposed: {proposed_minimum_sequence_length_gap}, allowed: {allowed_minimum_sequence_length_hysteresis})",
		])
		return (x, message)

	# All checks passed: update the configuration.
	config.minimum_corruption_percentage = infilling_minimum_corruption_percentage
	config.maximum_corruption_percentage = infilling_maximum_corruption_percentage
	config.minimum_sequence_length = infilling_calculated_minimum_sequence_length

	return (x, "Valid Configuration")