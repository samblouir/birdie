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



def get_autoencoding_objectives(config):
	'''
	Autoencoding
	'''
	objective_config = partial(
		AutoencodingConfig,
		mask_token_ids = config['autoencoding_mask_token_ids'],
	)
	
	autoencoding_corruption_rates = [
		0.05,
		0.15,
		0.5,
	]

	autoencoding_span_widths = [
		3,
		8,
		32,
	]

	autoencoding_lengths = [
		(192, 384),
		(384, 768),
		(768, 1536),
	]

	autoencoding_shuffle_prefix_spans = [
		False,
		True,
	]


	'''
		Prepare Autoencoding configurations
	'''
	autoencoding_objectives = {}
	for autoencoding_corruption_rate in autoencoding_corruption_rates:
		for autoencoding_mean_span_width in autoencoding_span_widths:
			for (autoencoding_minimum_sequence_length, autoencoding_maximum_sequence_length) in autoencoding_lengths:
				for autoencoding_shuffle in autoencoding_shuffle_prefix_spans:


					autoencoding_key = '_'.join([
						"autoencoding",
						f"{int(autoencoding_shuffle)}shuffle",
						f"{autoencoding_mean_span_width}w",
						f"{int(autoencoding_corruption_rate * 100)}p",
						f"{autoencoding_minimum_sequence_length}-{autoencoding_maximum_sequence_length}l",
					])

					configured_autoencoding_objective = dict(
						config=objective_config(
							objective="Autoencoding",
							corruption_rate=autoencoding_corruption_rate,
							mean_span_width=autoencoding_mean_span_width,
							minimum_sequence_length=autoencoding_minimum_sequence_length,
							maximum_sequence_length=autoencoding_maximum_sequence_length,
							shuffle=autoencoding_shuffle,
						),
						sampling_probability=1,
					)

					(configured_autoencoding_objective, message) = autoencoding_post_process(configured_autoencoding_objective)
					if not is_valid_configuration(configured_autoencoding_objective, message):
						continue

					(configured_autoencoding_objective, message) = autoencoding_filter_fn(configured_autoencoding_objective)
					if not is_valid_configuration(configured_autoencoding_objective, message):
						continue


					autoencoding_objectives[autoencoding_key] = configured_autoencoding_objective

	return autoencoding_objectives




def autoencoding_filter_fn(x):
	'''
	'''
	config = x['config']

	if (config.maximum_sequence_length <= config.minimum_sequence_length):
		message = ' '.join([
			f"Invalid Autoencoding Configuration: ",
			f"sequence length range: (max <= min) ({config.maximum_sequence_length} <= {config.minimum_sequence_length})"
		])
		return (x, message)
	
	if (1.0 <= config.corruption_rate):
		message =  ' '.join([
			f"Invalid Autoencoding Configuration: ",
			f"(1.0 <= {config.corruption_rate} (corruption_rate))",
		])
		return (x, message)
	
	return (x, "Valid Configuration")



def autoencoding_post_process(x):
	'''
		Post-processes the configuration.
	'''

	config = x['config']
	autoencoding_minimum_corruption_percentage = np.floor(config.corruption_rate * 0.25)
	autoencoding_maximum_corruption_percentage = np.ceil(config.corruption_rate * 2.0)

	autoencoding_calculated_minimum_sequence_length = np.floor(config.mean_span_width / config.corruption_rate / 2)

	# If the calculated minimum sequence length is already greater than the maximum sequence length, then we just ignore this configuration.
	if (config.maximum_sequence_length < autoencoding_calculated_minimum_sequence_length):
		message = ' '.join([
			f"Invalid Autoencoding Configuration: ",
			f"Calculated minimum sequence length ({autoencoding_calculated_minimum_sequence_length}) is greater than the maximum sequence length ({config.maximum_sequence_length}).",
		])
		return (x, message)


	# If there isn't enough space between our minimum and maximuze sequence lengths, then we ignore this configuration.
	allowed_minimum_sequence_length_hysteresis = (config.maximum_sequence_length - config.minimum_sequence_length)//4
	proposed_minimum_sequence_length_gap = (config.maximum_sequence_length - autoencoding_calculated_minimum_sequence_length)
	if (proposed_minimum_sequence_length_gap < allowed_minimum_sequence_length_hysteresis):
		message = ' '.join([
			f"Invalid Autoencoding Configuration: ",
			f"Proposed minimum sequence length gap is too small.",
			f"(proposed: {proposed_minimum_sequence_length_gap}, allowed: {allowed_minimum_sequence_length_hysteresis})",
		])
		return (x, message)

	# All checks passed: update the configuration.
	config.minimum_corruption_percentage = autoencoding_minimum_corruption_percentage
	config.maximum_corruption_percentage = autoencoding_maximum_corruption_percentage
	config.minimum_sequence_length = autoencoding_calculated_minimum_sequence_length

	return (x, "Valid Configuration")