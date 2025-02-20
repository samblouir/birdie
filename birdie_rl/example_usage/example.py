"""
This script demonstrates a minimal example of using Birdie with a custom reward function.
It sets up a dataset loader, defines a reward function, configures Birdie, and runs training
with periodic evaluations.
"""

# Import OS-level operations
import os

from birdie_rl import Birdie
from datasets import load_dataset
from tqdm import tqdm
import accelerate
import numpy as np
import tiktoken
import torch
from ul2_config import dummy_config, ul2_config, ul2_decoder_config
import utils


# Create a configuration dictionary for Birdie and training
config = {
	## Birdie uses these these functions and configs.
	## Please see ul2_config.py and utils.py for more details.
	"reward_fn": utils.reward_fn,
	"ds": utils.data_generator,
	"objectives": ul2_config,
	"tokenizer": tiktoken.get_encoding("o200k_base"),

	
	"batch_size": 8,
	"sequence_length": 2048,
	"num_workers": 16,
	"steps_between_evaluations": 32,
	"num_steps": 4096,
	"accelerator": accelerate.Accelerator(),
}

# Instantiate the Birdie object using our configuration
birdie = Birdie(config)

# Retrieve the initial action from Birdie
initial_action = birdie.get_current_action()
print(f"initial_action: {initial_action}")

# Create a deterministic NumPy random generator (for demonstration purposes)
seeded_np_rng = np.random.default_rng(0)

# Create a progress bar for the main training loop
progress_bar = tqdm(total=config["num_steps"], desc="Training")

# Main training loop
for step_idx in range(config["num_steps"]):
	# Update progress bar by one step
	progress_bar.update(1)

	# Retrieve the next sample (batch) from Birdie
	train_batch = birdie.get_next_training_sample()
	
	show_batch_stats = False

	if show_batch_stats:

		# Count how many input_ids are non-zero (used) vs. zero (wasted/padding)
		used_iids = torch.where(train_batch["input_ids"] == 0, 0, 1).sum().item()
		wasted_iids = torch.where(train_batch["input_ids"] == 0, 1, 0).sum().item()

		# Track the maximum token IDs in both inputs and labels
		max_input_ids = (train_batch["input_ids"]).max().item()
		max_label_ids = (train_batch["label_ids"]).max().item()

		# Compute the packer efficiency as the ratio of used tokens
		packer_efficiency = 1 - (wasted_iids / (wasted_iids + used_iids))

		_results = [
		    f"max_input_ids: {max_input_ids}, "
		    f"max_label_ids: {max_label_ids}, "
		    f"wasted_iids: {wasted_iids}, "
		    f"used_iids: {used_iids}, "
		    f"packer_efficiency: {packer_efficiency:.2%}"
		]
		print('\n'.join(_results))

	# Check if it's time to run a validation pass
	if birdie.time_for_eval(step_idx):
		# Measure validation losses for each objective
		for (objective_name, batch) in birdie.measure_validation_losses():
			# Here we simulate a loss using a random number 
			# (replace with a real model forward pass in practice)
			loss = seeded_np_rng.random()

			# Log this "loss" in Birdie
			birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step_idx)

		# Get a detailed string representation of the current action
		(status_dict, status_str) = birdie.get_verbose_action()
		print(status_str)

# Show that we are done
print("\n" * 3, end="")
print("All done. Closing Birdie...")

# Close Birdie and free associated resources
birdie.close()

# Hard exit to kill remaining threads or processes.
# Someone somewhere is holding onto VRAM...
os._exit(0)
