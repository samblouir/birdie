"""
===============================================================================
reward_model.py

**Purpose**:
	Currently just a wrapper around agent_bird.AgentBird.
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from birdie_rl.birdie_reward_model import agent_bird



class RewardModel(nn.Module):
	"""
	RewardModel currently just acts as a wrapper around agent_bird.AgentBird.
	"""

	def __init__(self,
			#   reward_signal_dims:int,
			#   num_objectives: int= None,
			#   hidden_dims=(256,256,256,256),
			#   lr=5e-4,
			#   device="cpu",
			  config: dict,
		):
		print(f'  HELLO!!!!!!!!!!!!!!!!!!!!!')

		super().__init__()
		self.__dict__.update(config)

		d = self.__dict__
		for d_idx, (key, value) in enumerate(d.items()):
			print(f"  d[{key}]: {value}")

		# self.reward_signal_dims = reward_signal_dims
		self.num_objectives = (config.get("num_objectives", config.get("reward_signal_dims", -1)))
		assert(0 < self.num_objectives)
		assert(0 < self.reward_signal_dims)
		# self.input_dim = (2 * self.reward_signal_dims)
		# self.output_dim = num_objectives
		# self.device = device

		# Sets the unique classes for the objectives
		# Currently assumes all objectives are unique and should share an equal weight when calculating the reward
		xc = np.arange(self.num_objectives)
		xc -= 1

		# This sets the first two classes to be labeled as the same objective (assumed to just have different configurations, i.e.: span corruption/infilling with a mean span width of 3 and 8)
		# This tells the reward model to not incentivize improving the performance of this "single" objective as two objectives... or, in other terms, each objective's "importance of improving" is half that of a regular objective.
		# xc[0:2] = 0


			
		
		agent_bird_kwargs = {
			**config,
			**dict(
				reward_signal_dims=self.reward_signal_dims,
				num_objectives=self.num_objectives,
				explore_classes=xc,
				# accelerator=accelerator,
				device=self.device,
				hidden_dims=self.hidden_dims,
			),
		}
		self.agent = agent_bird.AgentBird(**agent_bird_kwargs)


	def forward(self, *args, **kwargs):
		return self.agent.update_and_sample(*args, **kwargs)['action']
	
	def update(self, *args, **kwargs):
		return self.forward(*args, **kwargs)
	