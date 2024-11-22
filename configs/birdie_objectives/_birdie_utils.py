import numpy as np

## Verbose helper function
def is_valid_configuration(configured_objective, message):
	if message.strip().startswith('Invalid'):
		print(f"***\n  WARNING: Rejecting configuration!\n\t{message}")
		print(f"\tRejected configured_objective: {configured_objective}\n")
		return False
	assert(message.strip().startswith('Valid'))
	return True




def calculate_reward_scaling_vector(config):
	pretraining_objectives = config['pretraining_objectives']

	## Calculate the reward scaling vector
	unique_objectives = {}
	for value in pretraining_objectives.values():
		objective_config = value['config']
		unique_objectives[objective_config.objective] = unique_objectives.get(objective_config.objective, 0) + 1

	num_objective_configurations = len(pretraining_objectives)
	reward_scaling_vector = np.zeros(num_objective_configurations, dtype=np.float32)
	reward_index_offset = 0
	for number_of_objective_configurations in unique_objectives.values():
		cut_start = reward_index_offset
		cut_end = (reward_index_offset + number_of_objective_configurations)
		reward_scaling_vector[cut_start:cut_end] = (1.0 / number_of_objective_configurations)
		reward_index_offset += number_of_objective_configurations

	num_unique_objectives = len(unique_objectives)
	for pretraining_objectives_idx, (key, value) in enumerate(pretraining_objectives.items()):
		
		objective_config = value['config']
		number_of_objective_configurations = unique_objectives[objective_config.objective]
		pretraining_objectives[key]["sampling_probability"] = (1 / number_of_objective_configurations) * (1 / num_unique_objectives)
		pretraining_objectives[key]['reward_scale'] = reward_scaling_vector[pretraining_objectives_idx]

	reward_scaling_vector = np.float32(reward_scaling_vector)
	reward_scaling_vector /= np.sum(reward_scaling_vector)
	return reward_scaling_vector



def get_unique_objective_indices(config):
	pretraining_objectives = config['indexed_pretraining_objectives']

	## Calculate the unique objective indices by seeing who is in which objectives classes
	unique_objectives = {}
	output_unique_ids = []
	for value in pretraining_objectives:
		objective_config = value['fn']
		unique_objectives[objective_config.objective] = unique_objectives.get(objective_config.objective, len(unique_objectives)+1)
		output_unique_ids.append(unique_objectives[objective_config.objective])
		
	return np.int32(output_unique_ids)

def get_sampling_probabilities_vector(config):
	pretraining_objectives = config['pretraining_objectives']

	sampling_probabilities_vector = np.zeros(len(pretraining_objectives), dtype=np.float32)
	for idx, (key, value) in enumerate(pretraining_objectives.items()):
		sampling_probabilities_vector[idx] = value['sampling_probability']

	return sampling_probabilities_vector
	