import numpy as np
from functools import partial
import copy
from objective_configs import (
	InfillingConfig,
	PrefixLanguageModelingConfig,
)

def get_objectives(config=None):

	infilling_config = partial(
		InfillingConfig,
		mask_token_ids = config['infilling_mask_token_ids'],
	)
	
	pretraining_objectives = dict(
		nlu_3w_15p = dict(
				config=infilling_config(
					corruption_rate=0.15,
					mean_span_width=3,
					paradigm="[[[NLU]]]",
				),
				sampling_probability=4,
		),

		nlu_8w_15p = dict(
				config=infilling_config(
					corruption_rate=0.15,
					mean_span_width=8,
					paradigm="[[[NLU]]]",
				),
				sampling_probability=4,
		),

		nlg_3w_50p = dict(
				config=infilling_config(
					corruption_rate=0.5,
					mean_span_width=3,
					paradigm="[[[NLG]]]",
				),
				sampling_probability=2,
		),

		nlg_8w_50p = dict(
				config=infilling_config(
					corruption_rate=0.5,
					mean_span_width=8,
					paradigm="[[[NLG]]]",
				),
				sampling_probability=2,
		),

		nlg_64w_15p = dict(
				config=infilling_config(
					corruption_rate=0.15,
					mean_span_width=64,
					paradigm="[[[NLG]]]",
				),
				sampling_probability=2,
		),

		nlg_64w_50p = dict(
				config=infilling_config(
					corruption_rate=0.5,
					mean_span_width=64,
						paradigm="[[[NLG]]]",
				),
				sampling_probability=2,
		),

		s2s = dict(
				config=PrefixLanguageModelingConfig(
						prefix_frac=0.75,
						minimum_prefix_length=8,
						paradigm="[[[Prefix Language Modeling]]]",
						),
			sampling_probability=4,
		),
	)


	## Let's add some sanity checks
	for (key, value) in pretraining_objectives.items():
		config = value['config']
		if isinstance(config, InfillingConfig):
			corruption_rate = config.corruption_rate
			mean_span_width = config.mean_span_width

			config.minimum_corruption_percentage = np.floor(corruption_rate * 0.25)
			config.maximum_corruption_percentage = np.ceil(corruption_rate * 2.0)

			config.minimum_sequence_length = np.floor(mean_span_width / corruption_rate)

			if (0.5 <= corruption_rate) or (32 <= mean_span_width):
				config.paradigm = '[[[Extreme Infilling]]]'
			else:
				config.paradigm = '[[[Infilling]]]'

		


	## Normalize the sampling probabilities
	total = sum([v['sampling_probability'] for v in pretraining_objectives.values()])
	for k in pretraining_objectives:
		pretraining_objectives[k]['sampling_probability'] /= total


	return pretraining_objectives




def apply(config, inplace=False,):

	if not inplace:
		config = copy.deepcopy(config)

	config['fully_causal'] = False
	config['bidirectional'] = True
	config['pretraining_objectives'] = get_objectives(config=config)

	return config


if __name__ == "__main__":
	dummy_cfg = {
		"infilling_mask_token_ids": [11, 3333, 77,],
	}

	apply(dummy_cfg)

	sampling_probabilities = [v['sampling_probability'] for v in dummy_cfg['pretraining_objectives'].values()]
	assert(np.isclose(np.sum(sampling_probabilities), 1.0))

	for dummy_cfg_idx, (key, value) in enumerate(dummy_cfg.items()):
		print(f"  dummy_cfg[{key}]: ")

		if isinstance(value, dict) and "config" in value:
			config = value['config']
			if isinstance(config, InfillingConfig):
				corruption_rate = config.corruption_rate
				mean_span_width = config.mean_span_width

				config.minimum_corruption_percentage = np.floor(corruption_rate * 0.25)
				config.maximum_corruption_percentage = np.ceil(corruption_rate * 2.0)

				config.minimum_sequence_length = np.floor(mean_span_width / corruption_rate)
				if (0.5 <= corruption_rate) or (32 <= mean_span_width):
					assert(config.paradigm == '[[[Extreme Infilling]]]')
				else:
					assert(config.paradigm == '[[[Infilling]]]')
			
		