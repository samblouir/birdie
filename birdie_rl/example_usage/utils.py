
from datasets import load_dataset
import numpy as np
import torch

def reward_fn(
	action_taken=None,
	old_loss=None,
	new_loss=None,
	old_step_idx=None,
	new_step_idx=None
):
	"""
	The default reward function from the original Birdie paper. 
	It computes a reward based on the change in loss (delta_loss).
	This was tuned to aggressively pursue rewards enough so that 1.4B parameter SSMs being trained on Chinchilla-many (~32B) tokens would do well with the `Selective Copying` objective, with enough reward left on other objectives.

	The unused arguments (action_taken, old_step_idx, and new_step_idx) are there to improve support for different reward functions and features, such as decaying rewards over time.
	"""
	# Compute the change in loss
	delta_loss = (new_loss - old_loss)

	# Compute the proportional value of the delta loss with respect to the old loss
	rv = (delta_loss / (old_loss + 1e-8))

	# Construct an intermediate term based on sqrt(new_loss * old_loss), 
	# the cube of rv, and e (the base of natural logs)
	n = ((new_loss * old_loss).sqrt() * rv.pow(3) * torch.e)

	# Re-scale and apply a hyperbolic tangent function to allow help reduce the influence of minor rewards - this increases the impact of larger rewards for the reward model and helps reduce rewards for minor loss reductions.
	reward = (-100 * torch.tanh(n) * torch.e)

	# Replace NaN values in the reward with 0.0 (You shouldn't see this in practice...)
	reward = torch.where(torch.isnan(reward), torch.tensor(0.0), reward)

	# Clamp (limit) the reward between -1.0 and 1.0
	reward = torch.clamp(reward, -1.0, 1.0)

	# Return the computed reward
	return reward



def data_generator(split, worker_id, num_workers, rng_seed=0):
	"""
	The data_generator function will be called by each dataloading worker.
	This currently only data parallel training, where each accelerator has its own copy of the model.

	This function should return a generator for a given
	  - split (e.g., "train", "validation", "test")
	  - shards it by worker_id and num_workers
	  - shuffles the data using rng_seed
	"""

	# Load the TinyStories dataset from Hugging Face
	ds = load_dataset("roneneldan/TinyStories", split=split)

	# Shard the dataset among multiple workers
	ds = ds.shard(num_shards=num_workers, index=worker_id)

	# Shuffle the dataset for randomness
	ds = ds.shuffle(rng_seed)

	# Return the prepared dataset
	return ds