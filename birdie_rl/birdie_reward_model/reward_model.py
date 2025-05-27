"""
===============================================================================
reward_model.py

**Purpose**:
    This module defines the RewardModel class, which serves as a wrapper around
    agent_bird.AgentBird to provide a reward model interface.
    Ultimately, AgentBird will train on an input signal of width (2 * num_objectives),
    where the first half of the signal is the loss vector and the second half is the action taken.
    It will predict rewards based on these input signals.

===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim # Optimizers are handled within AgentBird
from birdie_rl.birdie_reward_model import agent_bird

class RewardModel(nn.Module):
    """
    A wrapper around agent_bird.AgentBird to provide a reward model interface.

    This class integrates the AgentBird model to compute rewards based on input signals.
    It is designed to be used in reinforcement learning setups where multiple objectives
    need to be balanced.

    Attributes:
        num_objectives (int): The number of distinct objective configurations.
        reward_signal_dims (int): The dimensionality of the reward signal input to AgentBird's MLP (typically old_loss_vector).
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        hidden_dims (tuple): The dimensions of the hidden layers in the underlying AgentBird's MLP.
        agent (AgentBird): The underlying AgentBird instance.
    """

    def __init__(self, config: dict):
        """
        Initializes the RewardModel with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters.
                Expected keys include:
                    - 'objectives': list, A list of objective configuration dictionaries.
                                     This is used to determine num_objectives and to
                                     generate explore_class_labels by grouping objectives
                                     with the same 'name'.
                    - 'reward_signal_dims': int, dimensionality of the reward signal. If not
                                           provided, defaults to num_objectives.
                    - 'hidden_dims': tuple, dimensions of hidden layers for AgentBird's MLP.
                    - 'device': str, device to run the model on.
                    - 'accelerator': object, an accelerator object for printing/logging.
                    - Additional keys are passed to AgentBird (see agent_bird.AgentBird.py).
        """
        super().__init__()
        # Update instance attributes with config dictionary
        # self.__dict__.update(config) # Be careful with this, prefer explicit assignment

        self.config = config
        self.device = config.get("device", "cpu")
        self.hidden_dims = config.get("hidden_dims", (256, 256, 256, 256)) # Default from AgentBird
        self.accelerator = config.get("accelerator")
        self.print_fn = self.accelerator.print if self.accelerator else print


        # Determine the number of objectives based on the provided 'objectives' list in config
        objectives_list_from_config = config.get("objectives", [])
        self.num_objectives = len(objectives_list_from_config)
        
        if self.num_objectives == 0:
            # Fallback if 'objectives' list is not in config or is empty
            self.num_objectives = config.get("num_objectives", config.get("reward_signal_dims", 0))
        
        assert self.num_objectives > 0, "Number of objectives must be positive and determinable from config."

        # reward_signal_dims is the dimension of the loss vector part of AgentBird's input.
        # It should match the number of objectives if rewards are per-objective.
        self.reward_signal_dims = config.get("reward_signal_dims", self.num_objectives)
        assert self.reward_signal_dims > 0, "Reward signal dimensions must be positive."
        if self.reward_signal_dims != self.num_objectives:
            self.print_fn(f"[RewardModel Warning] reward_signal_dims ({self.reward_signal_dims}) "
                          f"differs from num_objectives ({self.num_objectives}). "
                          f"Ensure this is intended for AgentBird's input structure.")

        # --- MODIFICATION: Generate explore_class_labels based on objective names ---
        explore_class_labels = np.arange(self.num_objectives) # Default if no objectives list
        if objectives_list_from_config:
            unique_names = []
            name_to_class_id = {}
            explore_class_labels = np.zeros(len(objectives_list_from_config), dtype=int)
            current_class_id = 0
            for i, obj_conf in enumerate(objectives_list_from_config):
                obj_name = obj_conf.get("name")
                if obj_name is None: # Should not happen if configs are well-formed
                    obj_name = f"unnamed_objective_{i}"
                    self.print_fn(f"[RewardModel Warning] Objective at index {i} has no name. Treating as unique.")

                if obj_name not in name_to_class_id:
                    name_to_class_id[obj_name] = current_class_id
                    unique_names.append(obj_name)
                    current_class_id += 1
                explore_class_labels[i] = name_to_class_id[obj_name]
            
            self.print_fn(f"[RewardModel] Generated explore_class_labels: {explore_class_labels.tolist()}")
            self.print_fn(f"[RewardModel] Based on name_to_class_id mapping: {name_to_class_id}")
        else:
            self.print_fn(f"[RewardModel Warning] 'objectives' list not found or empty in config. "
                          f"Defaulting explore_class_labels to range({self.num_objectives}). "
                          f"Reward scaling for variations might not work as intended.")
        # --- END MODIFICATION ---

        # Prepare keyword arguments for initializing AgentBird
        agent_bird_kwargs = {
            **config,  # Pass through all original config parameters
            'reward_signal_dims': self.reward_signal_dims, # For AgentBird's MLP input construction (loss vector part)
            'num_objectives': self.num_objectives,       # For AgentBird's action space dimension
            'explore_classes': explore_class_labels,     # Crucial for grouping variations
            'device': self.device,
            'hidden_dims': self.hidden_dims, # Ensure this is passed
        }
        # Remove 'objectives' from kwargs if it was the list of dicts, AgentBird doesn't need it directly
        if 'objectives' in agent_bird_kwargs and isinstance(agent_bird_kwargs['objectives'], list):
            del agent_bird_kwargs['objectives']


        # Initialize the underlying AgentBird model
        self.agent = agent_bird.AgentBird(**agent_bird_kwargs)

    def forward(self, *args, **kwargs):
        """
        Computes the action probabilities (objective sampling distribution) using the AgentBird model.

        This method forwards the input arguments (typically old_loss_vector, new_loss_vector, action_taken, etc.)
        to the AgentBird's update_and_sample method. AgentBird then:
        1. Updates its internal MLP model based on the new observation.
        2. Samples a new action (objective distribution) based on its updated policy.

        Args:
            *args: Variable length argument list passed to agent.update_and_sample.
            **kwargs: Arbitrary keyword arguments passed to agent.update_and_sample.
                      Commonly includes: new_loss_vector, old_loss_vector, action_taken,
                                       old_step_idx, new_step_idx, training (bool).

        Returns:
            np.ndarray: The 'action' (new objective sampling probabilities) computed by AgentBird.
        """
        # The 'training' flag for agent.sample() is important.
        # It's usually passed in kwargs from Birdie.update_reward_model
        result_dict = self.agent.update_and_sample(*args, **kwargs)
        return result_dict['action']

    def update(self, *args, **kwargs):
        """
        Alias for the forward method. Provided for semantic clarity in some contexts.

        Args:
            *args: Variable length argument list passed to forward.
            **kwargs: Arbitrary keyword arguments passed to forward.

        Returns:
            The result of the forward method (new objective sampling probabilities).
        """
        return self.forward(*args, **kwargs)

