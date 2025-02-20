import os
import accelerate
import numpy as np
from birdie_rl import Birdie
import tiktoken
from datasets import load_dataset


config = {}
accelerator = accelerate.Accelerator()

tokenizer = tiktoken.get_encoding("o200k_base")


ds = load_dataset("roneneldan/TinyStories",)

birdie = Birdie({
	"num_workers": 1,
	"accelerator": accelerator,
	"batch_size": config.get("batch_size", 32),
	"sequence_length": config.get("sequence_length", 128),
	"steps_between_evaluations": config.get("steps_between_evaluations", 512),
	"ds": ds,
	"tokenizer": tokenizer,
	"objectives": [
		{'name': 'next_token_prediction'},
		{'name': 'infilling'},
	],
	"validation_objectives": config.get('validation_objectives', config.get("objectives", [{"name": "next_token_prediction"}])),
})

print(f'  Created birdie! birdie: {birdie}')

# get action
action = birdie.get_current_action()

print(f"  action: {action}")


# get reward
old_loss = np.float32([5.0, 4.0,])
new_loss = np.float32([5.5, 3.5,])
action_taken = np.float32([0.0, 1.0,])
# action_taken = birdie.get_current_action()

action = birdie.get_current_action()

print(f"  action: {action}")

birdie.update_with_experience(
	old_loss_vector=old_loss,
	new_loss_vector=new_loss,
	action_taken=action_taken,
)


action = birdie.get_current_action()

print(f"  real action: {action}")

os._exit(0)




