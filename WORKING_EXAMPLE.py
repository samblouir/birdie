'''
	!!!!!!!!!!!!!!!!!!!!!!!!!!
	PLEASE SEE """birdie_rl/example_usage/example.py"""
	!!!!!!!!!!!!!!!!!!!!!!!!!!
	This minimal example loads in the original UL2 config, and simulates training (random losses are given for the validation losses).
	
'''
import os
from birdie_rl import Birdie
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from ul2_config import ul2_config
import accelerate
import numpy as np
import tiktoken
import torch



# Calculates the rewards
def reward_fn(action_taken=None, old_loss=None, new_loss=None, old_step_idx=None, new_step_idx=None,):
	## Equation from the original Birdie paper
	delta_loss = (new_loss - old_loss)
	rv = (delta_loss / (old_loss + 1e-8))
	n = ((new_loss * old_loss).sqrt() * rv.pow(3) * torch.e)
	reward = (-100 * torch.tanh(n) * torch.e)

	reward = torch.where(torch.isnan(reward), 0.0, reward)
	reward = torch.clamp(reward, -1.0, 1.0)
	return reward



config = {
	"batch_size": 8,
	"sequence_length": 2048,
	"num_workers": 16,
	"steps_between_evaluations": 256,
	"num_steps": 4096,
}


# This just needs to be a tokenizer that supports .encode() and .decode()
config['tokenizer'] = tiktoken.get_encoding("o200k_base")
config['accelerator'] = accelerate.Accelerator()

config['reward_fn'] = reward_fn



# Here we can add additional objectives to the validation set
config['objectives'] = ul2_config
# config['validation_objectives'] = ul2_config + [{"name": "next_token_prediction", "prob": 1.0}]


# We pass in a data generator function that can handle that returns the correct dataset for a given split, worker_id, num_workers, and rng_seed
def data_generator(split, worker_id, num_workers, rng_seed=0):
	ds = load_dataset("roneneldan/TinyStories", split=split,)
	ds = ds.shard(num_shards=num_workers, index=worker_id)
	ds = ds.shuffle(rng_seed)
	return ds
config['ds'] = data_generator

birdie = Birdie(config)

# get action
initial_action = birdie.get_current_action()
print(f"  initial_action: {initial_action}")


seeded_np_rng = np.random.default_rng(0)



results = []
progress_bar = tqdm(total=config['num_steps'], desc="Training")
for step_idx in range(config['num_steps']):
	progress_bar.update(1)
	train_batch = birdie.get_next_training_sample()

	## Calculate efficiency
	used_iids = torch.where(train_batch["input_ids"] == 0, 0, 1).sum().item()
	wasted_iids = torch.where(train_batch["input_ids"] == 0, 1, 0).sum().item()

	max_input_ids = (train_batch["input_ids"]).max().item()
	max_label_ids = (train_batch["label_ids"]).max().item()
	packer_efficiency = (1 - ((wasted_iids / (wasted_iids + used_iids))))
	
	# results.append(f"  max_input_ids: {max_input_ids:,}, max_label_ids: {max_label_ids:,},  wasted_iids: {wasted_iids:,}, used_iids: {used_iids:,}, packer_efficiency: {packer_efficiency:.2%}")

	if birdie.time_for_eval(step_idx):

		for (objective_name, batch) in birdie.measure_validation_losses():
			# loss = model(**batch)
			loss = seeded_np_rng.random()
			birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step_idx)

		# These are the new sampling probabilities that Birdie is using for the dataloader
		action = birdie.get_current_action()
		progress_bar.set_description(f"Current action: {action}")

print("\n".join(results))
print(f"\n" * 3, end='',)

print(f'  All done. Closing Birdie...')
birdie.close()
# Close stragglers
os._exit(0)




