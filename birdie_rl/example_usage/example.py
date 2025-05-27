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
from ul2_config import dummy_config, ul2_config, ul2_decoder_config # Ensure ul2_config is appropriate
from birdie_config import birdie_light_config 
import utils

# Define the number of steps for this example
NUM_EXAMPLE_STEPS = 256
STEPS_BETWEEN_EVALS = 32 # Agent updates occur at these intervals

# Create a configuration dictionary for Birdie and training
config = {
    ## Birdie uses these these functions and configs.
    ## Please see ul2_config.py and utils.py for more details.
    "reward_fn": utils.reward_fn,
    "ds": utils.data_generator,
    "objectives": birdie_light_config, # Using `birdie_light_config` instead of `ul2_config` for comprehensiveness
    "tokenizer": tiktoken.get_encoding("o200k_base"),
    "batch_size": 128,
    "sequence_length": 512,
    "minimum_sequence_length": 256,
    "min_seq_len_for_packing": 256, # Minimum sequence length for packing
    "num_workers": 1, # Keep low for simple examples to avoid overhead
    "steps_between_evaluations": STEPS_BETWEEN_EVALS,
    "num_steps": NUM_EXAMPLE_STEPS, # Total steps for this example run
    "accelerator": accelerate.Accelerator(),

    # AgentBird specific exploration parameters.
    "agent_explore_num_steps": NUM_EXAMPLE_STEPS, 
    "agent_explore_decay_steps": NUM_EXAMPLE_STEPS // 2, 
    "agent_explore_warmup_steps": int((NUM_EXAMPLE_STEPS // 2) * 0.1), 
    "agent_explore_rate_min": 0.1,
    "agent_explore_rate_max": 0.6,
    "agent_explore_cook_period": 0.05, 
    "agent_explore_cook_prob": 1.0,  
    
    "grok_iterations": 200, 
    "lr": 5e-4, 
    "hidden_dims": [128, 128], 

    "num_validation_batches": 2, 
}

print("Initializing Birdie...")
birdie = Birdie(config)
print("Birdie initialized.")

initial_action_probs = birdie.get_current_action()
print(f"Initial objective sampling probabilities: {initial_action_probs}")
_, initial_status_str = birdie.get_verbose_action()
print(f"Initial verbose action status:\n{initial_status_str}\n")

seeded_np_rng = np.random.default_rng(0)
progress_bar = tqdm(total=config["num_steps"], desc="Training")

for step_idx in range(config["num_steps"]):
    progress_bar.update(1)
    train_batch = birdie.get_next_training_sample()
    for train_batch_idx,(key,value) in enumerate(train_batch.items()):
        print(f"  train_batch[{key}]: {value.shape}")
        
    show_batch_stats = False 

    if show_batch_stats and train_batch.get("input_ids") is not None:
        used_iids = torch.where(train_batch["input_ids"] == 0, 0, 1).sum().item()
        wasted_iids = torch.where(train_batch["input_ids"] == 0, 1, 0).sum().item()
        max_input_ids = (train_batch["input_ids"]).max().item()
        max_label_ids = (train_batch["label_ids"]).max().item()
        packer_efficiency = 0
        if (wasted_iids + used_iids) > 0:
             packer_efficiency = 1 - (wasted_iids / (wasted_iids + used_iids))
        _results = [
            f"Step {step_idx}:",
            f"  max_input_ids: {max_input_ids}, "
            f"max_label_ids: {max_label_ids}, "
            f"wasted_iids: {wasted_iids}, "
            f"used_iids: {used_iids}, "
            f"packer_efficiency: {packer_efficiency:.2%}"
        ]
        tqdm.write("\n".join(_results))

    if birdie.time_for_eval(step_idx):
        tqdm.write(f"\n--- Step {step_idx}: Evaluating and Updating Agent ---")
        
        # Call measure_validation_losses once to get all batches and reset internal counters
        validation_batches_to_process = birdie.measure_validation_losses()

        for objective_name_key, batch_data in validation_batches_to_process:
            loss_value = seeded_np_rng.random() 
            birdie.log_validation_loss(key=objective_name_key, loss=loss_value, step_idx=step_idx)
            # tqdm.write(f"  Logged validation loss for {objective_name_key}: {loss_value:.4f}")
        
        # After all losses are logged for this cycle, explicitly check if it's time to update the reward model
        updated_this_cycle = birdie._maybe_update_reward_model(current_training_step=step_idx)
        if updated_this_cycle:
             tqdm.write(f"  Reward model updated at step {step_idx}.")
        # else: # This can be noisy if printed every time
        #      tqdm.write(f"  Reward model NOT updated at step {step_idx} (fulfilled keys: {birdie.validation_num_fulfilled_keys}/{len(birdie.validation_objectives)}).")

        (status_dict, status_str) = birdie.get_verbose_action()
        tqdm.write(f"  Verbose action status after evaluation at step {step_idx}:\n{status_str}\n")
        progress_bar.set_postfix_str(f"Eval @ {step_idx}")


print("\n" * 3, end="")
print("All training steps completed.")
print("Closing Birdie and freeing associated resources...")
birdie.close()
print("Birdie closed.")
print("Exiting script.")
os._exit(0)
