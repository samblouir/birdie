"""
===============================================================================
reward_model.py

**Purpose**:
    This module defines the RewardModel class, which serves as a wrapper around
    agent_bird.AgentBird to provide a reward model interface.
    Ultimately, AgentBird will train on an input signal of width (2 * num_objectives), where the first half of the signal is the 
      predict rewards based on the input signals.

===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from birdie_rl.birdie_reward_model import agent_bird

class RewardModel(nn.Module):
    """
    A wrapper around agent_bird.AgentBird to provide a reward model interface.

    This class integrates the AgentBird model to compute rewards based on input signals.
    It is designed to be used in reinforcement learning setups where multiple objectives
    need to be balanced.

    Attributes:
        num_objectives (int): The number of distinct objectives to optimize.
        reward_signal_dims (int): The dimensionality of the reward signal input.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        hidden_dims (tuple): The dimensions of the hidden layers in the underlying model.
        agent (AgentBird): The underlying AgentBird instance.
    """

    def __init__(self, config: dict):
        """
        Initializes the RewardModel with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys include:
                    - 'num_objectives': int, number of objectives.
                    - 'reward_signal_dims': int, dimensionality of the reward signal.
                    - 'hidden_dims': tuple, dimensions of hidden layers.
                    - 'device': str, device to run the model on.
                    - 'accelerator': object, an accelerator object for printing/logging.
                    - Additional keys can be seen agent_bird.Agentbird.py.
        """
        super().__init__()
        # Update instance attributes with config dictionary
        self.__dict__.update(config)

        # Determine the number of objectives, preferring 'num_objectives' if present
        self.num_objectives = config.get("num_objectives", config.get("reward_signal_dims"))
        assert self.num_objectives > 0, "Number of objectives must be positive."
        assert self.reward_signal_dims > 0, "Reward signal dimensions must be positive."

        # Log the configuration for debugging or monitoring
        # for key, value in sorted(config.items()):
        #     config['accelerator'].print(f"  RewardModel config [{key}]: {value}")

        # Define exploration class labels for objectives
        # This array assigns a unique label to each objective for exploration purposes.
        # Adjust this if objectives should be grouped (e.g., sharing importance weights).
        explore_class_labels = np.arange(self.num_objectives)
        # Example grouping: to treat the first two objectives as one group to explore together, uncomment this:
        # explore_class_labels[0:2] = 0  # This would set labels to [0, 0, 1, ...]
        # The objectives in a group will not necessarily have the same sampling probability.
        # It is used whne the random exploration is selecting objectives to explore.
        # In that case, the objectives groups should (probably) consist of those with obviously unique skills needed.
        # (This can help create an inductive bias for our goal: training the model on specific skills and seeing what mixtures of skills are useful.)

        # Prepare keyword arguments for initializing AgentBird, with some re-naming of keys.
        agent_bird_kwargs = {
            **config,  # Include all config parameters
            'reward_signal_dims': self.reward_signal_dims,
            'num_objectives': self.num_objectives,
            'explore_classes': explore_class_labels,  # Passes in the exploration labels
            'device': self.device,
            'hidden_dims': self.hidden_dims,
        }

        # Initialize the underlying AgentBird model with the prepared arguments
        self.agent = agent_bird.AgentBird(**agent_bird_kwargs)

    def forward(self, *args, **kwargs):
        """
        Computes the 1D reward vector using the AgentBird model.

        This method forwards the input arguments to the AgentBird's update_and_sample
        method and extracts the 'action' from the resulting dictionary.
        update_and_sample will cause AgentBird to train itself on the current (and historical) inputs, then generate new predicted best action to take.

        Args:
            *args: Variable length argument list passed to update_and_sample.
            **kwargs: Arbitrary keyword arguments passed to update_and_sample.

        Returns:
            The 'action' computed by the AgentBird model, typically representing
            the reward or decision output.
        """
        result = self.agent.update_and_sample(*args, **kwargs)
        return result['action']

    def update(self, *args, **kwargs):
        """
        Alias for the forward method.

        This method is provided for compatibility and directly calls the forward method.
        It may be used in contexts where 'update' is a more intuitive name.

        Args:
            *args: Variable length argument list passed to forward.
            **kwargs: Arbitrary keyword arguments passed to forward.

        Returns:
            The result of the forward method. Please see the forward method for details.
        """
        return self.forward(*args, **kwargs)
