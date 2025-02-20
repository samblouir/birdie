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
			  reward_signal_dims:int,
			  num_objectives: int= None,
			  hidden_dims=(256,256,256,256),
			  lr=5e-4,
			  device="cpu",
		):

		super().__init__()
		self.reward_signal_dims = reward_signal_dims
		self.num_objectives = (num_objectives or self.reward_signal_dims)
		self.input_dim = (2 * self.reward_signal_dims)
		self.output_dim = num_objectives
		self.device = device

		xc = np.arange(self.num_objectives)
		xc -= 1
		# xc[0:2] = 0
		
		self.agent = agent_bird.AgentBird(
			reward_signal_dims=self.reward_signal_dims,
			num_objectives=self.num_objectives,
			explore_classes=xc,
			# accelerator=accelerator,
			device=self.device,
			hidden_dims=hidden_dims,
		)


	def forward(self, *args, **kwargs):
		return self.agent.update_and_sample(*args, **kwargs)['action']
	
	def update(self, *args, **kwargs):
		return self.forward(*args, **kwargs)
	