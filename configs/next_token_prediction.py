
import numpy as np
from functools import partial
import copy
from objective_configs import (
	NextTokenPredictionConfig,
)
from birdie_objectives._birdie_utils import is_valid_configuration


def get_objectives(config=None):

	'''
		Next Token Prediction
	'''

	return{
		"Next Token Prediction": dict(
		config=NextTokenPredictionConfig(
			objective="Next Token Prediction",
		),
		sampling_probability=1,
	)
	}




def apply(config):

	config['fully_causal'] = True
	config['bidirectional'] = False

	config['pretraining_objectives'] = get_objectives(config=config)

	return config

	# "fully_causal": 2,
	# "head_dims": 3,
	# "num_heads": 3,
	# "pretraining_objective": 2,
	# "striped_encoder_decoder": 8,
	# "use_hawk": 3,
	# "vocab_size": 1,