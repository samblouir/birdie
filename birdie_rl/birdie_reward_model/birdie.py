"""
birdie.py

Defines the main 'Birdie' class, which:
  - Encapsulates the reward model, data loaders, and preparing validation batches to measure model performance to ultimately update the objectives being sampled.
  - Birdie will
  - Keeping track of the current mixture of objectives.
  - Requesting new data samples from the pipeline (for training).
  - Periodically evaluating sub-loss improvements.
  - Updating the RewardModel based on observed improvements.
  - Interfacing with your training loop (time_for_eval, get_next_training_sample, etc.).

Usage example:
	birdie = Birdie(config)
	for step in range(total_steps):
		if birdie.time_for_eval(step):
			# measure sub-loss improvements
			...
		sample = birdie.get_next_training_sample()
		# pass sample to your model, do forward/backward
		...
"""

import os
import threading
from typing import List, Dict, Any, Optional
import accelerate
import numpy as np
import torch
import json
from functools import partial

from birdie_rl.birdie_reward_model.reward_model import RewardModel
from birdie_rl.pipeline_generator import pipeline_data_generator
from birdie_rl.objectives import utils


# Default list of objectives used if none are provided in Birdie config
default_objectives_config: List[dict] = [
	{"name": "autoencoding_with_deshuffling"},
	{"name": "autoencoding"},
	{"name": "copying"},
	{"name": "deshuffling"},
	{"name": "infilling"},
	{"name": "next_token_prediction"},
	{"name": "prefix_language_modeling"},
	{"name": "selective_copying"},
]

def add_hashes(x):
	"""
	Recursively adds a hash key to each dictionary in a list of dictionaries.
	"""
	for i, obj in enumerate(x):
		if "hash_str" not in obj:
			obj["hash_str"] = utils.sha_hash(obj)[:8]
		obj['nickname'] = f"{obj['name']}_{obj['hash_str']}"
	return x

def normalize_objective_probs(x):
	"""
	Normalizes the probabilities of objectives in a list of dictionaries.
	"""
	total = sum(obj.get("prob", 1.0) for obj in x)

	if (total <= 0.0):
		for obj in x:
			obj["prob"] = 1.0
		total = len(x)
		print(f"  WARNING: Total probability of objectives is {total}. Normalized each objective's probability to 1.0. (total number of objectives: {len(x)})")

	for obj in x:
		obj["prob_initial"] = (obj.get("prob", 1.0) / total)
	return x
	


def default_move_to_gpu_fn(batch: Dict[str, Any], accelerator: accelerate.Accelerator) -> Dict[str, torch.Tensor]:
	"""
	Moves each tensor in the batch to the appropriate device (GPU or otherwise).

	Args:
		batch (dict): A dictionary of data fields, typically containing
						model inputs as tensors or arrays.

	Returns:
		dict: A dictionary with the same keys, whose values are now on self.accelerator.device.
	"""

	if not isinstance(batch, dict):
		tp = []
		tp.append(f"#" * 60,)
		tp.append(f"*" * 60,)
		tp.append(f"  FATAL EXCEPTION: BATCH not a dict? dtype(batch): {dtype(batch)},  batch: {batch}")
		tp.append(f"*" * 60,)
		tp.append(f"#" * 60,)
		print("\n".join(tp), flush=True) # joined and flushed to improve multi-processing print clarity (so worker's overlap their print lines less often).
		raise ValueError(f"  FATAL EXCEPTION: batch is a string?? batch: {batch}")
	
	device = accelerator.device if accelerator else torch.device("cpu")
	for k, v in batch.items():
		if isinstance(v, torch.Tensor):
			batch[k] = v.to(device)
		else:
			batch[k] = torch.tensor(v, dtype=torch.long).to(device)
	return batch

class Birdie:
	"""
	Orchestrates the RL-based mixture sampling process in a simplified manner.

	Primary responsibilities:
	  - Keeps track of the mixture of objectives used for sampling.
	  - Requests training data batches from the pipeline (via get_next_training_sample()).
	  - Periodically measures validation losses (time_for_eval, measure_validation_losses,
		and log_validation_losses).
	  - Updates its internal RewardModel based on observed sub-loss improvements.
	  - Adjusts objective probabilities accordingly.

	Usage Example:
		>>> birdie = Birdie(config)
		>>> for step in range(total_steps):
		>>>     if birdie.time_for_eval(step):
		>>>         # measure sub-loss improvements
		>>>         ...
		>>>     sample = birdie.get_next_training_sample()
		>>>     # pass sample to your model, do forward/backward
		>>>     ...
	"""

	def __init__(self, config: Optional[dict] = None):
		"""
		Initialize Birdie with a given config dictionary.
		Note: If 'objectives' are not provided, the default objectives are used, and defaults assuming a training step count of 16_384 are assumed.

		Args:
			config (dict, optional): A dictionary containing configuration items. Some
				important keys:
				- 'accelerator': an Accelerate object or any custom device manager.
				- 'batch_size': integer, batch size.
				- 'dataset' or 'ds': the dataset / data generator for training samples.
				- 'ds_validation': optional dataset / data generator for validation.
				- 'objectives': list of dictionaries describing objectives for the mixture.
				- 'reward_signal_dims': integer, dimension of the reward signal vector.
				- 'sequence_length': integer, the default sequence length for training.
				- 'steps_between_evaluations': how often to run evaluations.
				- 'tokenizer': optional tokenizer object (if needed in pipeline).
				- 'total_steps': how many total training steps are planned.
				- 'num_workers': number of worker processes for data pipeline.
				- 'validation_objectives': optional set of objectives specifically for validation.
				- 'validation_sequence_length': sequence length for validation samples.
				- 'num_validation_batches': how many batches to fetch per objective for validation.
				...

				These are forwarded to agent_bird:
					- reward_signal_dims (int) (the model's output dimensions (the output values are currently in the range of [-1, 1]))
					- num_objectives (int) (the model's input dimensions (i.e.: the number of training objectives objectives))
					- hidden_dims (list) (hidden layer sizes for agent_birds Transformer-based reward model)
					- lr, weight_decay, etc. (floats) (hyperparameters for the agent's optimizer)
					- explore_classes (list or array) (defines which parts of the action vector correspond to different objectives. For example, if you have next_token_prediction seq_len 512 and next_token_prediction seq_len 1024, you might want to group them together as a single objective. Otherwise, the model will make this worth twice as much as a normal objective. Your desries may vary!)
					- device: "cpu", "cuda", etc.
					
					- agent_explore_warmup_steps: Number of steps to cosine-decay exploration over. The agent will explore at the prob. of (agent_explore_max_rate to agent_explore_min_rate) over (agent_explore_num_steps).
					- agent_explore_num_steps: Number of steps until the agent explores at agent_explore_min_rate.
					- agent_explore_decay_steps: Number of steps to decay exploration.
					- agent_explore_rate_min: Minimum exploration rate.
					- agent_explore_rate_max: Maximum exploration rate.
					- agent_explore_cook_period: Percentage of agent_explore_num_steps to hold exploration constant at agent_explore_cook_prob. This is similar to the common adafactor lr warmup over the first 10_000 steps. Can be over 1.0.

					Common defaults for these from a dict of functions that update a config:
					"agent_explore_num_steps": lambda x: x.get("agent_explore_num_steps", x['num_steps'] // 2),
					"agent_explore_warmup_steps": lambda x: x.get("agent_explore_warmup_steps", min(2048, x['agent_explore_num_steps'] * 0.1)),
					"agent_explore_decay_steps": lambda x: x.get("agent_explore_decay_steps", x['agent_explore_num_steps']//2), ## Causes exploration to decay to the minimum by the second half of training, allowing the agent to transition to exploitation.
					"agent_explore_rate_min": lambda x: x.get("agent_explore_rate_min", 0.2),
					"agent_explore_rate_max": lambda x: x.get("agent_explore_rate_max", 0.5),
					"agent_explore_cook_period": lambda x: x.get("agent_explore_cook_period", 0.1),
					"agent_explore_cook_prob": lambda x: x.get("agent_explore_cook_prob", 1.0),
			
		"""

		if config is None:
			config = {}

		# Store raw config for reference
		self.config = config

		# Core settings
		self.accelerator = config.get("accelerator", None)
		self.batch_size = config.get("batch_size", 8)
		self.dataset = config.get("dataset", None)  # HF dataset or generator (passed to pipeline)
		self.ds = config.get("ds", self.dataset)    # For training
		self.ds_validation = config.get("ds_validation", self.ds)  # By default, reuse ds if not set
		self.num_workers = config.get("num_workers", 16)
		self.sequence_length = config.get("sequence_length", 1024)
		self.steps_between_evaluations = config.get("steps_between_evaluations", 512)
		self.tokenizer = config.get("tokenizer", None)
		self.total_steps = config.get("total_steps", config.get("num_steps", 16_384))
		self.move_to_gpu_fn = config.get("move_to_gpu_fn", partial(default_move_to_gpu_fn, accelerator=self.accelerator))


		# Prepare objectives and their probabilities
		self.objectives = config.get("objectives", default_objectives_config)
		self.objectives = add_hashes(self.objectives)
		self.objectives = normalize_objective_probs(self.objectives)
		## Support for this argument is coming soon! It allows us to output a different sized vector than the number of objectives are sampling from.
		self.reward_signal_dims = int(config.get("reward_signal_dims", len(self.objectives)))


		# For validation performance tracking
		## Stores cached validation batches for evaluation 
		self.validation_ds_per_objective = {}

		## You can use a different set of objectives for validation.
		## Can be useful if you want to precisely measure performance based on different hyperparameters.
		self.validation_objectives = config.get("validation_objectives", self.objectives)
		self.validation_objectives = add_hashes(self.validation_objectives)
		self.validation_objectives = normalize_objective_probs(self.validation_objectives)

		## You can set a different sequence length for validation samples.
		self.validation_sequence_length = int(config.get("validation_sequence_length", self.sequence_length))
 		## How many batches to fetch per objective for validation. I would assume the higher the better, but it is slower.
		self.num_validation_batches = config.get("num_validation_batches", 16)

		# Sanity check: we need a dataset for training
		if self.ds is None:
			raise ValueError("Please provide a dataset or data generator when creating Birdie.")

		# Keep track of how many batches we measure for each objective
		self.measured_num_batches_per_objective: Dict[str, int] = {}

		# Prepare validation samples once and store them
		self.get_validation_samples()

		# Initialize the last action with default probabilities from the config (if any)
		self.last_action = np.array([
			obj.get("prob", 1.0) for obj in self.objectives
		])
		self.last_action /= np.sum(self.last_action)

		# Initialize a small RewardModel to predict sub-loss improvements
		reward_model_config = {
			**config,
			"reward_signal_dims": self.reward_signal_dims,
			"num_objectives": len(self.objectives),
			"hidden_dims": (256, 256, 256, 256),
			"lr": 5e-4,
			"device": self.accelerator.device if self.accelerator else "cpu",
		}
		for reward_model_config_idx, (key, value) in enumerate(reward_model_config.items()):
			print(f"  reward_model_config[{key}]: {value}")
			
		self.reward_model = RewardModel(reward_model_config)

		# Create data pipeline for training
		self.controller, self.ds = pipeline_data_generator(
			batch_size=self.batch_size,
			sequence_length=self.sequence_length,
			objectives_config=self.objectives,
			num_workers=self.num_workers,
			accelerator=self.accelerator,
			move_to_gpu_fn=self.move_to_gpu_fn,
			data_generator=self.ds,
			split='train',
			config=self.config,
		)

		# Track progress
		self.current_step = 0
		self.old_loss_vector: Optional[np.ndarray] = None
		self.validation_measured_losses: List[Dict[str, float]] = []
		self.validation_key_to_losses: Dict[str, List[float]] = {}
		self.validation_num_fulfilled_keys = 0
		self.current_validation_losses: Optional[np.ndarray] = None

	def close(self) -> None:
		"""
		Closes the processes created by pipeline_data_generator. 
		This also spawns a separate thread to forcefully exit after cleanup,
		as some data loader processes may otherwise hang.
		"""
		if hasattr(self, "controller") and self.controller is not None:
			self.controller.close()

		def _force_exit():
			os._exit(1)

		thread = threading.Thread(target=_force_exit)
		thread.start()

	def __del__(self):
		"""
		Ensures processes are cleaned up on deletion. 
		Calls self.close() and then any potential parent destructor.
		"""
		self.close()
		parent_del = getattr(super(), "__del__", None)
		if callable(parent_del):
			parent_del()

	def time_for_eval(self, step_idx: Optional[int] = None) -> bool:
		"""
		Determines if it's time to run an evaluation (e.g., every N steps).

		Args:
			step_idx (int, optional): The current training step. If not provided,
				uses self.current_step.

		Returns:
			bool: True if an evaluation should be run at this step.
		"""
		step = step_idx if step_idx is not None else self.current_step
		return (step % self.steps_between_evaluations) == 0

	def get_next_training_sample(self) -> Any:
		"""
		Produces the next training sample (batch) from the dataset, guided by 
		the mixture distribution. This is the function your training loop calls 
		every iteration to get data.

		Returns:
			A batch (usually a dictionary of Tensors) for training.
		"""
		self.current_step += 1
		return next(self.ds)

	def get_validation_samples(self) -> Dict[int, Dict[str, Any]]:
		"""
		Prepares and caches validation samples for each objective, if not already done.

		Returns:
			dict: A dictionary of:
				{
				  objective_idx: {
					"objective": <objective dict>,
					"batches": <list of validation batches>
				  },
				  ...
				}
		"""

		# Checks if we have already prepared validation samples
		if len(self.validation_ds_per_objective) == 0:
			
			intermediates = []

			# This loop will start the workers for each objective
			for objective_idx, objective in enumerate(self.validation_objectives):

				self.accelerator.print(
					f"  Starting async worker to create validation samples for {objective} "
					f"({objective_idx+1}/{len(self.validation_objectives)})"
				)

				controller, ds_validation_iter = pipeline_data_generator(
					data_generator=self.ds_validation,
					data_generator_fn_kwarg_overrides={"worker_id": 0, "num_workers": 1},
					batch_size=self.batch_size,
					sequence_length=self.validation_sequence_length,
					objectives_config=[objective],
					num_workers=1,
					accelerator=self.accelerator,
					# move_to_gpu_fn=self.move_to_gpu_fn,
					infinite_loop=False,
					split='validation',
					config=self.config,
				)
				intermediates.append((objective, objective_idx, controller, ds_validation_iter))

			# This collects all batches for each objective
			for (objective, objective_idx, controller, ds_validation_iter) in intermediates:
				self.accelerator.print(
					f"  Reading in validation samples for {objective} "
					f"({objective_idx+1}/{len(self.validation_objectives)})"
				)
				current_samples = []
				while len(current_samples) < self.num_validation_batches:
					batch = next(ds_validation_iter, None)
					if batch is None:
						if 0 < len(current_samples):
							break
						else:
							continue
					## Uncomment to see each batch coming in
					# print(f"  objective: {objective}, objective_idx: {objective_idx},  controller: {controller},  batch: {batch}")
					current_samples.append(batch)

				self.validation_ds_per_objective[objective_idx] = {
					"objective": objective,
					"batches": current_samples,
				}

				# Record how many batches we have for that objective (using nickname or name)
				nickname = objective.get("hash_str", objective["name"])
				self.measured_num_batches_per_objective[nickname] = len(current_samples)

				controller.close()

		return self.validation_ds_per_objective

	def measure_validation_losses(self) -> List[Any]:
		"""
		Initiates a new measurement of all validation losses. 
		 - This returns a flattened list of (nickname, batch) tuples that should be iterated over.
		 - After each, call log_validation_loss(nickname, loss, step_idx) to record the measured loss.
		 	(step_idx is your current training step)

		Steps:
			1. Clears old measurements.
			2. Collects cached validation batches for each objective.
			3. Flattens them into a list of (nickname, batch) pairs.

		Returns:
			list of (str, dict):
				A list of (nickname, batch) pairs, where 'batch' is the validation batch.
		"""
		# Clear out old measurements
		self.validation_measured_losses = []
		self.validation_key_to_losses = {}
		self.validation_num_fulfilled_keys = 0

		ds_validation = self.get_validation_samples()  # Ensure validation data is available

		# for ds_validation_idx, (key, value) in enumerate(ds_validation.items()):
		# 	print(f"  ds_validation[{key}]: {value}")
		# exit()

		flat_batches = []
		for _, value in ds_validation.items():
			nickname = value["objective"].get("hash_str", value["objective"]["nickname"])
			for batch in value["batches"]:
				# print(f"  batch: {batch}")
				if self.move_to_gpu_fn is not  None:
					batch = self.move_to_gpu_fn(batch)
				flat_batches.append((nickname, batch))

		return flat_batches

	def log_validation_loss(self, key: str, loss: float, step_idx: Optional[int] = None) -> None:
		"""
		Collects losses for a given objective (identified by 'key') and, once
		all expected losses for that objective are logged, updates the mixture probabilities.

		Args:
			key (str): The nickname or name of the objective.
			loss (float): The measured validation loss for this batch.
			step_idx (int, optional): Current step index, used for scheduling/time checks.
		"""
		# Append the new loss
		self.validation_key_to_losses.setdefault(key, []).append(loss)

		# Check if we reached the expected number of validation batches for this key
		desired_length = self.measured_num_batches_per_objective.get(key, 0)
		# print(f"  len(self.validation_key_to_losses[key]): {len(self.validation_key_to_losses[key])},  desired_length: {desired_length},  key: {key}")
		if len(self.validation_key_to_losses[key]) == desired_length:
			self.validation_num_fulfilled_keys += 1

			# print(f"  len(self.get_validation_samples()): {len(self.get_validation_samples())},  self.validation_num_fulfilled_keys: {self.validation_num_fulfilled_keys}")
			
			# If all objectives have fulfilled their batches, we can finalize our result
			if self.validation_num_fulfilled_keys == len(self.get_validation_samples()):
				means = {
					k: np.mean(losses)
					for k, losses in self.validation_key_to_losses.items()
				}
				self.validation_measured_losses.append(means)

				# Convert the dictionary to a vector in the same order as objectives
				current_losses_vector = []
				for obj in self.objectives:
					key = obj.get("hash_str", obj["nickname"])
					current_losses_vector.append(means.get(key, 0.0))
				self.current_validation_losses = current_losses_vector

				# If we don't have a reference old_loss_vector yet, initialize it
				if self.old_loss_vector is None:
					self.old_loss_vector = current_losses_vector.copy()
					self.last_reward_step_idx = step_idx
				else:
					# Update mixture probabilities via the RewardModel
					new_action = self.update_reward_model(
						action_taken=self.last_action,
						old_loss_vector=self.old_loss_vector,
						new_loss_vector=current_losses_vector,
						old_step_idx=self.last_reward_step_idx,
						new_step_idx=step_idx,
					)
					self.last_action = new_action
					self.last_reward_step_idx = step_idx

					# Update objective probabilities and reconfigure the data pipeline
					for idx, entry in enumerate(self.objectives):
						entry["prob"] = float(np.round(new_action[idx], 3))
						entry["loss"] = float(np.round(current_losses_vector[idx], 4))

					# Instruct the data generator to adjust its sampling distribution
					self.controller.update(self.objectives, clear_prefetched=False)

	def update_reward_model(
		self,
		action_taken: np.ndarray,
		old_loss_vector: np.ndarray,
		new_loss_vector: np.ndarray,
		old_step_idx: Optional[int] = None,
		new_step_idx: Optional[int] = None,
	) -> np.ndarray:
		"""
		Updates the internal RewardModel by sending it
		action_taken, old_loss_vector, new_loss_vector, old_step_idx, and new_step_idx.
		The reward function being called insude self.reward_model.update was set in the Config when initializing Birdie.

		Args:

			action_taken (np.ndarray): The mixture distribution used for training.
			old_loss_vector (np.ndarray): Sub-losses before the new training.
			new_loss_vector (np.ndarray): Sub-losses after the new training.
			old_step_idx (int, optional): Old step index.
			new_step_idx (int, optional): New step index.

		Returns:
			np.ndarray: The next action (new mixture distribution) recommended by the RewardModel.
		"""
		# Get new recommended action from the RewardModel
		new_action = self.reward_model.update(
			action_taken=action_taken,
			old_loss_vector=old_loss_vector,
			new_loss_vector=new_loss_vector,
			old_step_idx=old_step_idx,
			new_step_idx=new_step_idx,
		)
		self.old_loss_vector = new_loss_vector.copy()

		return new_action

	def get_current_validation_losses(self) -> Optional[np.ndarray]:
		"""
		Retrieves the most recently computed validation losses as a NumPy vector.

		Returns:
			np.ndarray or None: The latest array of validation losses (one entry per objective),
			or None if no validation step has been completed yet.
		"""
		if self.current_validation_losses is not None:
			assert(len(self.current_validation_losses) == len(self.objectives)), f"  FATAL EXCEPTION: len(self.current_validation_losses): {len(self.current_validation_losses)},  len(self.objectives): {len(self.objectives)}"
		return self.current_validation_losses

	def get_current_action(self) -> np.ndarray:
		"""
		Returns the most recent mixture distribution (probabilities) over objectives.

		Returns:
			np.ndarray: A 1D array of floats, each dimension representing the probability of an objective configuration being sampled.
		"""
		if self.last_action is not None:
			assert(len(self.last_action) == len(self.objectives)), f"  FATAL EXCEPTION: len(self.last_action): {len(self.last_action)},  len(self.objectives): {len(self.objectives)}"
		return self.last_action
	
	def get_verbose_action(self) -> Dict[str, float]:
		"""
		Returns the most recent mixture distribution (probabilities) over objectives, 
		including the objective name and probability.

		Returns:
			ret_dict: A list of dictionaries, each containing (at least) 'name' and 'prob' keys for an objective.
			formatted_ret_dict_str: A formatted string representation of the return value.
		"""
		ret_dict = {}
		current_action = self.get_current_action()

		def flatten_dict(d, parent_key='', sep='.'):
			items = []
			for k, v in d.items():
				new_key = f"{parent_key}{sep}{k}" if parent_key else k
				if isinstance(v, dict):
					items.extend(flatten_dict(v, new_key, sep=sep).items())
				else:
					items.append((new_key, v))
			return dict(items)

		flat_objectives = [flatten_dict(self.objectives[_]) for _ in range(len(self.objectives))]
		for objectives_idx, (objective) in enumerate(flat_objectives):
			serializable_dict = {}
			for k,v in objective.items():
				try:
					json.dumps(v)
				except TypeError:
					continue
				serializable_dict[k] = v
			objective = serializable_dict
			config = objective
			name = config.get("nickname", config["name"])
			from birdie_rl.objectives import utils
			hash_str = utils.sha_hash(config)
			ret_dict[name] = {
				"current_sampling_probability": float(current_action[objectives_idx]),
				"config": dict(sorted({k:v for k,v in config.items() if k not in ["name"]}.items())),
			}

		formatted_ret_dict_str = json.dumps(ret_dict, indent=4)
		return (ret_dict, formatted_ret_dict_str)
			




if __name__ == "__main__":
	import os
	import numpy as np
	import accelerate
	import tiktoken

	os.system("clear")
	print("  Hi! This is a test file for the Birdie class.\n")

	objectives_config = [
		{"name": "autoencoding_with_deshuffling"},
		{"name": "autoencoding"},
		{"name": "copying"},
		{"name": "deshuffling"},
		{"name": "infilling"},
		{"name": "next_token_prediction"},
		{"name": "prefix_language_modeling"},
		{"name": "selective_copying"},
	]

	def data_generator_fn(split, worker_id, num_workers, rng_seed=0):
		# This dummy fn returns the same list of sstring for any worker or split.
		return [str(np.arange(x+1)) for x in range(512, 1024)]
	
	def text_grabber_fn(x):
		# This just returns a string - our data_generator_fn is returning a list of strings.
		# If data_generator_fn was returning a list of dicts, we would receieve each dict here and would be free to apply whichever transformations we want.
		return x

	tokenizer = tiktoken.get_encoding("o200k_base")
	
	accelerator = accelerate.Accelerator()
	config = {
		"tokenizer": tokenizer,
		"total_steps": 16_384,
		"steps_between_evaluations": 1,
		"batch_size": 8,
		"max_sequence_length": 2048,
		# "minimum_sample_length": 256,
		"accelerator": accelerator,
		"ds": data_generator_fn,
		"text_grabber_fn": text_grabber_fn,
	}

	birdie = Birdie(config)


	print("*" * 60)
	print("  Showing validation samples:")
	ds_validation = birdie.get_validation_samples()
	# for idx, val_info in ds_validation.items():
	# 	val_info["batches"] = f"{len(val_info['batches'])} samples"
	# 	print(f"  ds_validation[{idx}]: {val_info}")
	print("  Created Birdie!")
	print("*" * 60)

	seeded_np_rng = np.random.RandomState(42)

	# Mocking two 'training + validation' cycles
	for _ in range(2):
		if birdie.time_for_eval(step_idx=1):
		# if True: # unneeded when setting steps_between_evaluations to 1
			# Start measuring
			flat_batches = birdie.measure_validation_losses()
			# Fake computation of losses
			for validation_batches_idx, (nickname, batch) in enumerate(flat_batches):
				fake_loss = 1.1 + validation_batches_idx * 0.1 + seeded_np_rng.rand()
				birdie.log_validation_loss(nickname, fake_loss)
			action = birdie.get_current_action()
			print(f"  action: {action}")
			verbose_action, verbose_action_str = birdie.get_verbose_action()
			print(f"  verbose_action: {verbose_action_str}")
	
	birdie.close()
