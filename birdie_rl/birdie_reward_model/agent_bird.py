"""
agent_bird.py

PURPOSE:
	- Defines functionality related to scheduling or sampling actions (like objective distributions)
	  and computing decaying schedules for RL or bandit-like approaches.
	- Provides helper methods for safe tensor/numpy casting and scheduling.
"""

import numpy as np
import torch
import torch.distributions as dist   # For Pareto distribution sampling, to choose random actions to be ranked by the reward model
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW        # Optimizer
from tqdm import tqdm                # Progress bar
from birdie_rl.birdie_reward_model import gated_ssm_agent

# Define constants that are used in multiple places.
fp32_zero = torch.tensor(0.0, dtype=torch.float32)
fp64_zero = torch.tensor(0.0, dtype=torch.float64)
fp32_one = torch.tensor(1.0, dtype=torch.float32)
fp64_one = torch.tensor(1.0, dtype=torch.float64)
fp64_neg_one = torch.tensor(-1.0, dtype=torch.float64)


def make_cosine_decay_schedule(
	total_steps,
	min_val,
	max_val,
	warmup_steps=0,
	decay_factor=1.0,
	cook_steps=0,
	cook_val=1.0,
):
	"""
	Creates a function that returns a decaying value following a single cosine schedule.

	The schedule has:
	  - A warmup period where no decay occurs until `warmup_steps`.
	  - After warmup, a cosine decay is applied.
	  - 'cook_steps' is an optional additional period at the start of training
		during which the value remains constant at `cook_val`.
	  - 'decay_factor' applies exponential decay to the amplitude of the cosine
		to gradually reduce it over time.

	Args:
		total_steps (int): The total number of steps for the schedule.
		min_val (float): Minimum value of the schedule after full decay.
		max_val (float): Maximum value at the beginning (or after warmup).
		warmup_steps (int): Steps to skip decay at the start (0 by default).
		decay_factor (float): Exponential factor that further reduces amplitude.
		cook_steps (int): Additional steps at the start to hold a fixed value.
		cook_val (float): The value used during the cook_steps phase.

	Returns:
		function: A function f(step) that returns the scheduled value at `step`.
	"""

	# If max_val <= 0, schedule will always return 0; early exit with dummy function.
	if max_val <= 0.0:
		def dummy_schedule(*args, **kwargs):
			return 0.0
		return dummy_schedule

	# Adjusted step counts to handle warmup offsets and scaled total
	adjusted_total_steps = total_steps - 1 - warmup_steps
	scaled_total_steps = (total_steps * 2) - 2

	def single_decaying_cosine_value(current_step):
		"""
		Inner function that, given a step, computes the schedule's value.
		"""
		# If we are still in cook_steps, hold the constant cook_val
		if current_step < cook_steps:
			return cook_val

		# Shift current_step by warmup
		current_step = current_step - warmup_steps
		current_step = np.minimum(current_step, adjusted_total_steps)
		current_step = np.maximum(0, current_step)

		# Calculate amplitude and apply decay factor
		amplitude = (max_val - min_val) / 2
		decayed_amplitude = amplitude * (decay_factor ** (current_step / scaled_total_steps))

		# Calculate base and the cosine portion
		base_value = min_val + amplitude
		cosine_value = np.cos(np.pi * current_step / adjusted_total_steps)
		result_val = base_value + decayed_amplitude * cosine_value

		# Clip to [min_val, max_val]
		result_val = np.maximum(min_val, result_val)
		result_val = np.minimum(max_val, result_val)

		return result_val

	# Run some test checks to ensure the function is consistent.
	init_val = single_decaying_cosine_value(int(total_steps * 0.1))
	end_val = single_decaying_cosine_value(int(total_steps * 0.9))

	try:
		assert warmup_steps < total_steps, (
			f"warmup_steps: {warmup_steps}, total_steps: {total_steps}"
		)
		# If min_val < max_val, initial value should be bigger than the near-end value.
		if min_val < max_val:
			assert init_val > end_val, (
				f"init_val: {init_val}, end_val: {end_val}, "
				f"min_val: {min_val}, max_val: {max_val}"
			)
		else:
			assert init_val < end_val, (
				f"init_val: {init_val}, end_val: {end_val}, "
				f"min_val: {min_val}, max_val: {max_val}"
			)
		# By the end, the schedule should be near min_val
		final_val = single_decaying_cosine_value(total_steps)
		assert np.isclose(final_val, min_val), (
			f"single_decaying_cosine_value(total_steps): {final_val}, min_val: {min_val}"
		)
	except Exception as e:
		# debug_msgs = [
		# 	f"make_cosine_decay_schedule()",
		# 	f"total_steps: {total_steps}",
		# 	f"min_val: {min_val}",
		# 	f"max_val: {max_val}",
		# 	f"warmup_steps: {warmup_steps}",
		# 	f"decay_factor: {decay_factor}",
		# 	f"init_val: {init_val}",
		# 	f"end_val: {end_val}",
		# 	f"Exception: {e}",
		# ]
		# Print the debug messages if desired (commented out to remove extraneous prints)
		# print('\n'.join(debug_msgs))
		raise e

	return single_decaying_cosine_value


def custom_format_fn(x):
	"""
	Example formatting function that formats a float with three decimal places.

	Args:
		x (float): The value to format.

	Returns:
		str: The formatted string.
	"""
	return f"{x:.3f}"


def to_numpy(x, dtype=np.float32):
	"""
	Safely cast a PyTorch tensor or numeric value to a numpy array of a given dtype.

	Args:
		x (any): A numeric value, PyTorch tensor, or array.
		dtype (np.dtype): Numpy data type for casting.

	Returns:
		np.ndarray or scalar: The resulting numpy representation.
	"""
	try:
		# If x is a PyTorch tensor, detach and convert to numpy
		return dtype(x.detach().numpy())
	except:
		# If x is not a tensor, attempt direct cast
		try:
			return dtype(x)
		except:
			return x


def safe_cast_to_numpy(x, dtype=np.float32):
	"""
	Safely cast a PyTorch tensor (possibly on GPU) to a numpy array.

	Args:
		x (Tensor or any): The input that might be a tensor or numeric value.
		dtype (numpy dtype): Desired numpy type, default float32.

	Returns:
		np.ndarray or the original type if conversion fails.
	"""
	# Move to CPU if possible
	try:
		x = x.cpu()
	except:
		pass
	# Attempt to detach and convert
	try:
		return dtype(x.detach().numpy())
	except:
		# Final fallback
		try:
			return dtype(x)
		except:
			return x


def safe_cast_to_tensor(x, dtype=None):
	"""
	Safely cast an input x to a PyTorch tensor with optional dtype.

	Args:
		x (any): The input, possibly already a tensor or a numeric/array.
		dtype (str): A string representing the torch dtype ('float16','float32','float64','int32','int64').

	Returns:
		Tensor: The result as a detached tensor on default device.
	"""
	try:
		# If x is already a Tensor, clone it to avoid side effects
		if isinstance(x, torch.Tensor):
			ret_val = x.clone().detach()
		else:
			ret_val = torch.tensor(x)
	except Exception as e:
		# In case of error, try a fallback
		ret_val = torch.tensor(x)

	# If a dtype string was provided, map it to torch.dtype and cast
	if dtype is not None:
		torch_dtype_map = {
			"float16": torch.float16,
			"float32": torch.float32,
			"float64": torch.float64,
			"int32": torch.int32,
			"int64": torch.int64,
		}
		ret_val = ret_val.to(torch_dtype_map[dtype])

	return ret_val


def remap(x, in_min=None, in_max=None, out_min=-1.0, out_max=1.0):
	"""
	Remap array values from [in_min, in_max] to [out_min, out_max].

	If in_min or in_max is None, it uses x.min() or x.max() respectively.

	Args:
		x (Tensor or array): The values to map.
		in_min (float): Minimum of the input domain. Uses x.min() if None.
		in_max (float): Maximum of the input domain. Uses x.max() if None.
		out_min (float): The new range's minimum (default -1.0).
		out_max (float): The new range's maximum (default 1.0).

	Returns:
		Tensor: The remapped values in the specified new range.
	"""
	x = safe_cast_to_tensor(x)
	if in_min is None:
		in_min = torch.min(x)
	if in_max is None:
		in_max = torch.max(x)
	div = (in_max - in_min) + 1e-8
	return (x - in_min) * (out_max - out_min) / div + out_min


def loss_penalty(x, scale='exp'):
	"""
	A placeholder function that can penalize or rescale actions based on their loss.
	By default, it just returns x unchanged (no penalty).

	Args:
		x (Tensor): The input loss values.
		scale (str): A key to determine the scaling strategy. Unused here.

	Returns:
		Tensor: The adjusted values, currently the same as input.
	"""
	# In principle, you could modify this function to re-scale
	# or transform x based on the objective's difficulty.
	return x


class AgentBird:
	"""
	Class representing a bandit-like or RL-like agent that uses a small Transformer to
	decide how to distribute actions (objectives) based on observed improvements.

	

	kwargs include:

		- reward_signal_dims (int) (the model's output dimensions (the output values are currently in the range of [-1, 1]))
		- num_objectives (int) (the model's input dimensions (i.e.: the number of training objectives objectives))
		- hidden_dims (list) (hidden layer sizes for agent_bird's Transformer-based reward model)
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

	# Default class-level attributes
	num_samples = 1
	num_objectives = 8 # Uses the default 8 objectives in Birdie
	reward_signal_dims = num_objectives # Output reward signal: see comments in __init__

	num_iterations = 16384

	# Schedule params
	decay_steps = int(num_iterations * 0.1)
	warmup_steps = int(decay_steps * 0.1)
	smoothing_window = -1
	last_only = 0
	dropout_rate = 0.0
	steps_between_evals = 250
	num_test_samples = 2048

	# For exploration
	explore_classes = None
	accelerator = None
	device = None


	# Transformer config
	### Number of top actions to average over
	top_n = 8
	### Number of hidden layers and hidden dims in the Transformer
	hidden_dims = [256, 256, 256, 256]
	### Forces this many training steps per update to the reward model, regardless of the loss.
	grok_iterations = 200
	### Optimization hyperparams
	lr = 5e-4
	lr_warmup_steps = -1
	weight_decay = 0.1
	div_loss_mult = 0
	

	# Minimum step_idx where the exploration prob is forced to be at "agent_explore_cook_prob" (which defaults to 1.0)
	must_explore_steps = 10
	

	# Training data placeholders
	train_y = None
	X = None

	# RNG
	np_rng_seed = 0


	training_stats = {}
	training_explore_history = []

	# Exploration hyperparams
	exploration_rate_min = 0.25
	exploration_rate_max = 0.50
	agent_num_actions_to_try = 4096

	# placeholders for extra model references (currently not implemented)
	# "slop" correction model... not implemented. Was correcting disparities between observed and commanded objective sampling ratios. This is to be integrated into the worker to up-sample under-sampled objectives.
	conversion_model = None 
	# Diffusion model that predicts N-steps into the future, rather than this current greedy setup. Coming in a future update.
	diffusion_model = None

	# For bootstrapping (currently not implemented)
	bootstrap__data = None
	bootstrap__strategy = None
	bootstrap__random_seed = None
	bootstrap__num_examples = None
	bootstrap__config = None
	bootstrap__enabled = False
	bootstrap_ctrs = {}

	def __init__(self, **kwargs):
		"""
			Please see the comments for this class for additional details on the arguments.

			Initialize the AgentBird instance. Merges kwargs into internal attributes.
			Note: If arguments are not provided, several defaults are used that assume a training step count of 16_384.
		"""

		# Merge any user-provided kwargs with the class defaults
		self.__dict__.update(kwargs)

		# Ensure num_objectives is an int
		self.num_objectives = int(self.num_objectives)

		# If reward_signal_dims not specified, set it to num_objectives
		if self.reward_signal_dims is None:
			self.reward_signal_dims = self.num_objectives

		# The function used to evaluate reward from old->new losses
		self.reward_fn = kwargs.get("reward_fn", self.calculate_reward)

		# Adjust decay steps based on num_iterations
		self.num_iterations = kwargs.get("agent_explore_num_steps", self.num_iterations)
		self.decay_steps = kwargs.get("agent_explore_decay_steps", int(self.num_iterations * 0.50))
		self.warmup_steps = kwargs.get("agent_explore_warmup_steps", int(self.num_iterations * 0.10))
		self.exploration_rate_min = kwargs.get("agent_explore_rate_min", 0.20)
		self.exploration_rate_max = kwargs.get("agent_explore_rate_max", 0.50)

		# Cook steps are an optional period to hold exploration at a certain value until we reach a certain step (defined by cook period)
		explore_cook = float(kwargs.get("agent_explore_cook_period", 0.05))
		# This is the probability of exploration during the cook period
		agent_explore_cook_prob = float(kwargs.get("agent_explore_cook_prob", 1.0))
		cook_steps = int(self.num_iterations * explore_cook)

		# Step counters and histories
		self.step_counter = 0
		self.action_history = []
		self.reward_history_observed = []
		self.reward_history_estimated = []
		self.explore_history = []

		# If initial X, y are passed in, store them as Tensors
		initial_X = kwargs.get("initial_X", None)
		initial_y = kwargs.get("initial_y", None)
		if initial_X is not None:
			self.X = safe_cast_to_tensor(initial_X)
		if initial_y is not None:
			self.train_y = safe_cast_to_tensor(initial_y)

		# Setup for bootstrapping from an external data source (optional feature)
		if self.bootstrap_ctrs is None:
			self.bootstrap_ctrs = {}
		self.bootstrap__strategy = None

		# Not imlemented
		if self.bootstrap__strategy is not None:
			raise Exception("  Agent state bootstrapping not implemented yet.")
		# 	self.bootstrap__enabled = True
		# 	import formatted_history
		# 	self.bootstrap__data = formatted_history.get_data(config=self.bootstrap__config)
		# 	# Sort by final mean loss, then pick best/worst/random, etc. as per strategy

		# Construct a schedule for exploration probability
		self.cosine_decay_schedule_kwargs = dict(
			total_steps=self.decay_steps,
			warmup_steps=self.warmup_steps,
			cook_steps=cook_steps,
			min_val=self.exploration_rate_min,
			max_val=self.exploration_rate_max,
			cook_val=agent_explore_cook_prob,
		)
		self.explore_schedule = make_cosine_decay_schedule(**self.cosine_decay_schedule_kwargs)

		# RNG for random actions
		self.np_rng = np.random.RandomState(self.np_rng_seed)
		self.explore_np_rng = np.random.RandomState(self.np_rng_seed + 1)

		# Pre-generate floats for exploration checks
		self.explore_floats = self.explore_np_rng.random(128_000)
		self.explore_schedule_materialized = np.float32([
			self.explore_schedule(step) for step in range(len(self.explore_floats))
		])
		self.explore_results = (self.explore_floats < self.explore_schedule_materialized)

		# Create a dictionary mapping step -> boolean of "did we explore?"
		self.explore_history_dict_step_to_did_explore = {
			idx: result for idx, result in enumerate(self.explore_results)
		}


		# Build the Transformer model used to predict rewards. By default we use a small Transformer in gated_ssm_agent
		self.model = gated_ssm_agent.MLPModel(
			input_dim=self.reward_signal_dims + self.num_objectives,
			output_dim=self.num_objectives,
			hidden_dims=self.hidden_dims,
			dropout_rate=self.dropout_rate,
		)
		self.model = torch.compile(self.model).to(self.device)

		# Initialize optimizer (AdamW) for the model
		self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1)

		# Example LR scheduler (cyclic). You could change this as needed.
		self.scheduler = torch.optim.lr_scheduler.CyclicLR(
			self.optimizer,
			base_lr=self.lr * 0.001,
			max_lr=self.lr,
			step_size_up=self.grok_iterations / 2,
			mode='exp_range',
		)

	def check_should_explore(self, step=None):
		"""
		Checks if the agent should explore at a given training step.

		This uses the precomputed explore_results array and schedules.
		If the maximum exploration rate is extremely low, we skip exploring.
		"""
		step = step or self.step_counter

		# If we've gone past the decay steps entirely and min_val is near zero, we skip
		if self.decay_steps < step:
			if self.exploration_rate_min <= 1e-4:
				return False

		# Use the step modulo the length of our precomputed array
		result = self.explore_history_dict_step_to_did_explore[step % len(self.explore_history_dict_step_to_did_explore)]
		# If the exploration_rate_max is effectively zero, skip
		if self.exploration_rate_max <= 1e-4:
			result = False

		return result

	def scale_actions(self, x, current_loss=None):
		"""
		Rescale actions based on the current loss if desired. 
		This normalizes x so that it sums to 1, ensuring it is a valid probability distribution.

		Args:
			x (Tensor or array): The raw action vector.
			current_loss (Tensor): The current loss vector for each objective.

		Returns:
			Tensor: The normalized action vector summing to 1.
		"""
		x = safe_cast_to_tensor(x)
		if current_loss is not None:
			current_loss = safe_cast_to_tensor(current_loss)
			x *= loss_penalty(current_loss)
		# Shift up so no negative entries
		if x.min() < 0:
			x -= x.min()
		# Normalize to sum to 1
		x = x / x.sum()
		return x

	def random_uniform(self, *shape):
		"""
		Produce uniform random values in [0,1) with the internal numpy RNG, shaped as 'shape'.

		Args:
			*shape: Arbitrary dimensions for the output array.

		Returns:
			np.ndarray: The random array of floats in [0,1).
		"""
		return self.np_rng.random(np.prod(shape)).reshape(shape).astype(np.float32)

	def random_uniform_in_range(self, out_min, out_max, shape):
		"""
		Produce uniform random values in [out_min, out_max].

		Args:
			out_min (float): lower bound.
			out_max (float): upper bound.
			shape (tuple): shape of the output array.

		Returns:
			np.ndarray: The random array within [out_min, out_max].
		"""
		values = self.random_uniform(*shape)
		return remap(values, out_min=out_min, out_max=out_max)

	def scale_actions_batch(self, x, current_loss=None):
		"""
		Similar to scale_actions but processes a batch (2D) of action vectors.

		Args:
			x (Tensor): shape [batch_size, num_objectives].
			current_loss (Tensor): A single or batched loss factor.

		Returns:
			Tensor: Normalized action distribution for each row in x.
		"""
		x = safe_cast_to_tensor(x)
		if current_loss is not None:
			x *= loss_penalty(current_loss)[None, None]
		if x.min() < 0:
			# Subtract the min along each row
			x -= x.min(dim=-1, keepdim=True)[0]
		x = x / x.sum(dim=-1, keepdim=True)
		return x

	def predict_rewards(self, test_input):
		"""
		Runs the MLP model on the given test_input to get a predicted reward.

		Args:
			test_input (Tensor): shape [batch_size, seq_len, features].

		Returns:
			Tensor: shape [batch_size, num_objectives]. The predicted reward.
		"""
		self.model.eval()
		test_input = test_input.to(self.device)

		# By default, we pass the entire sequence. If we want a single step, we might clamp it.
		current_seq_len = test_input.shape[1]

		# Optionally apply a 'conversion_model' if set (unused by default).
		if self.conversion_model is not None:
			raise NotImplementedError("Anti-slop transformation not implemented yet!")
			chunk_size = 64
			for chunk_idx in range(0, test_input.shape[0], chunk_size):
				test_input[chunk_idx:chunk_idx + chunk_size, :, -self.num_objectives:] = \
					self.conversion_model(test_input[chunk_idx:chunk_idx + chunk_size, :, -self.num_objectives:])

		# We pad the input if shorter than the default. 
		# In practice, the MLP might not need strict padding, but this is an example from the code.
		# "gated_ssm_agent.default_max_seq_len" is used to ensure shape consistency.
		test_input = torch.cat([
			test_input,
			torch.zeros(
				(
					test_input.shape[0],
					gated_ssm_agent.default_max_seq_len - current_seq_len,
					*test_input.shape[2:],
				), device=test_input.device,
			),
		], dim=1)

		# Run inference in chunks to avoid OOM if large
		chunks = []
		chunk_size = 64

		if self.accelerator is not None:
			disable = (not self.accelerator.is_main_process)
		else:
			disable = False

		with torch.no_grad():
			for idx in tqdm(range(0, test_input.shape[0], chunk_size), desc="Predicting rewards... (done in chunks to save VRAM)", disable=disable):
				chunk = test_input[idx:idx + chunk_size]
				# The model returns shape [batch_size, seq_len, output_dim].
				# We pick the last time step or a specific one (just an example).
				chunk_output = self.model(chunk, current_seq_len=current_seq_len)[..., current_seq_len, :]
				chunks.append(chunk_output)
		sampled_preds = torch.cat(chunks, dim=0)
		self.model.train()

		return sampled_preds

	def mass_sample(self, current_loss, num_test_samples, action=None, top_n=None):
		"""
		Samples a large set of candidate actions, either random or partially guided.

		Args:
			current_loss (Tensor): The current objective losses.
			num_test_samples (int): How many random samples to consider.
			action (Tensor): If provided, we replicate it as a base for randomization.
			top_n (int): Number of top actions to pick from the sorted predictions.

		Returns:
			(best_action, estimated_reward, predicted_rewards, test_actions)
		"""
		top_n = top_n or self.top_n

		# If no explicit action is passed, start with uniform random
		if action is None:
			test_actions = torch.abs(torch.tensor(self.random_uniform(num_test_samples, 1, self.num_objectives)).float())
		else:
			# If an action is provided, replicate it for each test sample
			test_actions = action[None][None].repeat((num_test_samples, 1, 1)).float()

		# For demonstration, sample from some Pareto distributions for additional exploration
		pareto_0 = dist.Pareto(scale=2.0, alpha=1.0).sample(test_actions.shape).float()
		pareto_1 = dist.Pareto(scale=3.0, alpha=2.0).sample(test_actions.shape).float()
		pareto_2 = dist.Pareto(scale=5.0, alpha=3.0).sample(test_actions.shape).float()

		# Concatenate
		test_actions = torch.cat([test_actions, pareto_2, pareto_1, pareto_0], dim=0).float()

		# Scale each row to sum to 1
		test_actions = self.scale_actions_batch(test_actions, current_loss)

		# Build the input for the MLP:
		# We simply combine the current_loss and the candidate action
		axis_0_reps = (test_actions.shape[0] // num_test_samples)
		broadcast_current_loss = safe_cast_to_tensor(current_loss)[None][None].repeat(
			(num_test_samples * axis_0_reps, 1, 1)
		).float()
		test_input = torch.cat([broadcast_current_loss, test_actions], dim=-1).float()

		# If we have X, repeat it to match test_input shape and optionally combine
		if self.X is not None:
			repeated_X = self.X.repeat((broadcast_current_loss.shape[0], 1, 1))
			repeated_X = safe_cast_to_tensor(repeated_X, dtype="float32")
			test_input = safe_cast_to_tensor(test_input, dtype="float32")
			test_input = torch.cat([repeated_X, test_input], dim=-2).float()

		predicted_rewards = self.predict_rewards(test_input)

		# Example weighting code to handle "explore_classes"
		# Here we scale each objective by a factor 1/(count of objectives with same ID)
		# This is somewhat domain-specific logic.
		uids = np.unique(self.explore_classes)
		objective_scaling = np.zeros(self.num_objectives, dtype=np.float32)
		for uid in uids:
			matches = (self.explore_classes == uid)
			scaled_match_value = 1 / np.sum(np.int32(matches))
			objective_scaling = np.where(matches, scaled_match_value, objective_scaling)
		predicted_rewards *= torch.tensor(objective_scaling[None], dtype=predicted_rewards.dtype,
										  device=predicted_rewards.device) * torch.tensor(
											  len(uids), dtype=predicted_rewards.dtype, device=predicted_rewards.device)

		# Convert predicted_rewards to CPU for further calculations
		predicted_rewards = predicted_rewards.cpu()
		# Summation along objectives for each sample
		per_sample_predicted_rewards = predicted_rewards.sum(dim=1)
		# Pick the best top_n samples
		top_n_indices = torch.argsort(per_sample_predicted_rewards, dim=-1, descending=False)[-top_n:].cpu().numpy()

		predicted_rewards = predicted_rewards.numpy()
		selected_rewards = predicted_rewards[top_n_indices]
		estimated_reward = selected_rewards.mean()

		# Convert test_actions to numpy and pick best actions
		test_actions = safe_cast_to_numpy(test_actions)
		top_n_indices = safe_cast_to_numpy(top_n_indices, dtype=np.int32)
		selected_actions = test_actions[top_n_indices, -1]
		best_action = np.mean(selected_actions, axis=0)

		return best_action, estimated_reward, predicted_rewards, test_actions

	def sample(self, current_loss, num_test_samples=None, training=True, force_explore=False, watchdog_counter=2):
		"""
		Produce a single action sample, either exploring or exploiting.

		Args:
			current_loss (Tensor): The current objective losses.
			num_test_samples (int): How many candidate actions to evaluate.
			training (bool): If True, checks exploration logic.
			force_explore (bool): If True, forcibly explore ignoring schedules.
			watchdog_counter (int): A fallback to force exploration if we get degenerate solutions.

		Returns:
			dict: Contains action, estimated_reward, sampled_preds, test_actions, and a boolean 'explored'.
		"""
		if num_test_samples is None:
			num_test_samples = self.agent_num_actions_to_try

		# Convert the current_loss to a Tensor
		current_loss = safe_cast_to_tensor(current_loss, dtype="float32")

		# Determine if we explore
		should_explore = force_explore or (training and self.check_should_explore(self.step_counter))

		# If exploration rate is effectively zero, skip exploring
		if self.exploration_rate_max <= 1e-4:
			should_explore = False

		# If exploring, pick random distribution for some objectives
		if should_explore:
			if training:
				self.training_explore_history.append(self.step_counter)

			max_choices = len(np.unique(self.explore_classes))
			max_choices = max(2, max_choices)
			num_choices = self.np_rng.randint(1, 3)
			explore_action = np.zeros(self.num_objectives, dtype=np.float32)

			# Randomly pick subsets from unique classes
			uids = np.unique(self.explore_classes)
			selected_uids = self.np_rng.choice(uids, num_choices)
			for uid in selected_uids:
				matches = (self.explore_classes == uid)
				scaled_match_value = self.np_rng.random(np.sum(matches))
				scaled_match_value /= np.sum(scaled_match_value)
				explore_action[matches] = scaled_match_value

			# Add a small portion for unselected
			shared_unexplored_objective_probability = (
				len(self.explore_classes) / max(1, len(self.explore_classes) - num_choices)
			) * 0.005 * num_choices
			for uid in uids:
				if uid not in selected_uids:
					matches = (self.explore_classes == uid)
					scaled_match_value = shared_unexplored_objective_probability / np.sum(matches)
					explore_action[matches] = scaled_match_value
			explore_action = safe_cast_to_tensor(explore_action, dtype="float32")
		else:
			explore_action = None

		# Evaluate many random or partially-random actions
		(best_action, estimated_reward, sampled_preds, test_actions) = self.mass_sample(
			current_loss=current_loss, num_test_samples=num_test_samples, action=explore_action
		)

		# Final normalizing
		best_action = self.scale_actions(best_action, current_loss)
		self.action_history.append(best_action)
		self.reward_history_estimated.append(estimated_reward)

		return {
			"action": best_action,
			"estimated_reward": estimated_reward,
			"sampled_preds": sampled_preds,
			"test_actions": test_actions,
			"explored": should_explore,
		}

	def append_to_history(self, action_taken, old_loss_vector, new_loss_vector, observed_reward):
		"""
		Appends a new sample (action, old_loss_vector, new_loss_vector, observed_reward)
		to the training history (X, y) so the MLP can be updated.

		Args:
			action_taken (Tensor): The chosen action.
			old_loss_vector (Tensor): The old sub-losses.
			new_loss_vector (Tensor): The new sub-losses.
			observed_reward (Tensor): The observed reward from old->new.

		Returns:
			Tensor: The observed_reward (unchanged).
		"""
		# If no reward is given, compute from the reward function
		if observed_reward is None:
			observed_reward = self.reward_fn(
				old_loss_vector=old_loss_vector,
				new_loss_vector=new_loss_vector,
			)

		# Prepare new_x_entry => a single "row" with shape (1, 2, #objectives)
		# We stack [old_loss_vector, action_taken]
		new_x_entry = torch.cat([old_loss_vector[None], action_taken[None]], dim=1).float()[None]

		# Expand X with new_x_entry
		if self.X is None:
			self.X = new_x_entry
		else:
			self.X = torch.cat([self.X, new_x_entry], dim=-2).float()

		# Similarly, expand train_y with the observed reward
		observed_reward = observed_reward[None][None]
		if self.train_y is None:
			self.train_y = observed_reward
		else:
			self.train_y = torch.cat([self.train_y, observed_reward], dim=-2).float()

		return observed_reward

	def update(
		self,
		new_loss_vector=None,
		old_loss_vector=None,
		action_taken=None,
		observed_reward=None,
		old_step_idx=None,
		new_step_idx=None,
	):
		"""
		Perform an update step to incorporate new (old_loss, new_loss, action),
		and then run a small training loop on the MLP to fit the new data.

		Args:
			new_loss_vector (Tensor): The new sub-losses.
			old_loss_vector (Tensor): The old sub-losses.
			action_taken (Tensor): The chosen action.
			observed_reward (Tensor): Optionally provided reward. If None, compute from reward_fn.
			old_step_idx (int): Not used here, but kept for compatibility.
			new_step_idx (int): Not used here, but can be used for scheduling.

		Returns:
			Tensor: The observed_reward.
		"""
		# If no explicit action, fallback to the last in self.action_history
		if action_taken is None:
			action_taken = self.action_history[-1]

		# Convert inputs to safe Tensors
		action_taken = safe_cast_to_tensor(action_taken, dtype="float32")
		old_loss_vector = safe_cast_to_tensor(old_loss_vector, dtype="float32")
		new_loss_vector = safe_cast_to_tensor(new_loss_vector, dtype="float32")

		# If no explicit reward, compute
		if observed_reward is None:
			observed_reward = self.reward_fn(
				old_loss_vector=old_loss_vector,
				new_loss_vector=new_loss_vector,
			)

		# Append to X and y
		observed_reward = self.append_to_history(
			action_taken=action_taken,
			old_loss_vector=old_loss_vector,
			new_loss_vector=new_loss_vector,
			observed_reward=observed_reward,
		)

		self.model.train()

		# Ensure our stored X, y are on device
		x = self.X.to(self.device).float()
		y = self.train_y.to(self.device).float()

		# We limit the sequence length if it is bigger than the MLP can handle
		seq_len_limit = gated_ssm_agent.default_max_seq_len
		x = x[:, -seq_len_limit:]
		y = y[:, -seq_len_limit:]

		# We can optionally apply a conversion model if available
		if self.conversion_model is not None:
			raise NotImplementedError("Anti-slop transformation not implemented yet!")
		# 	chunk_size = 64
		# 	for chunk_idx in range(0, x.shape[0], chunk_size):
		# 		x[chunk_idx:chunk_idx + chunk_size, :, x.shape[-1] // 2:] = \
		# 			self.conversion_model(x[chunk_idx:chunk_idx + chunk_size, :, x.shape[-1] // 2:])

		# Pad x to default sequence length if needed
		pad_length = seq_len_limit - x.shape[1]
		if pad_length > 0:
			x = torch.cat([
				x,
				torch.zeros(
					(x.shape[0], pad_length, x.shape[2]),
					device=x.device
				)
			], dim=1)

		# Similarly y can be padded. Example omitted for brevity, or to keep consistent shape.

		num_iterations = self.grok_iterations
		# A small training loop
		self.bootstrap_ctrs["just_initialized"] = self.bootstrap_ctrs.get("just_initialized", 0) + 1
		if self.bootstrap__enabled:
			# If we are "bootstrapping," we might do more iterations initially
			if self.bootstrap_ctrs["just_initialized"] == 1:
				num_iterations *= 5
			elif self.bootstrap_ctrs["just_initialized"] == 2:
				num_iterations *= 2.5
			elif self.bootstrap_ctrs["just_initialized"] == 3:
				num_iterations *= 1.5

		num_iterations = int(num_iterations)

		progress_bar = tqdm(range(num_iterations), desc="Training Agent Bird...", leave=False)
		loss_val = 999
		# We'll do a small training loop
		with torch.enable_grad():
			for _ in progress_bar:
				self.optimizer.zero_grad()
				# The model output is shape [batch, seq_len, objectives], we only match up to y's shape
				predictions = self.model(x, current_seq_len=y.shape[1])
				# Calculate mean-squared error between predictions and y
				# Align shapes if needed
				pred_slice = predictions[:, :y.shape[1]]
				loss_val = ((pred_slice - y) ** 2).mean()
				loss_val.backward()
				clip_grad_norm_(self.model.parameters(), max_norm=1.0)
				self.optimizer.step()
				self.scheduler.step()
				progress_bar.set_postfix({"loss": loss_val.item()})

		progress_bar.close()

		# Return the observed reward
		return observed_reward

	def update_and_sample(
		self,
		new_loss_vector=None,
		old_loss_vector=None,
		action_taken=None,
		observed_reward=None,
		old_step_idx=None,
		new_step_idx=None,
		training=True,
	):
		"""
		Convenience function that calls 'update' then immediately calls 'sample'.

		Args:
			new_loss_vector (Tensor): The new sub-losses.
			old_loss_vector (Tensor): The old sub-losses.
			action_taken (Tensor): The chosen action.
			observed_reward (Tensor): Observed reward (optional).
			old_step_idx (int): Past step index (for logging).
			new_step_idx (int): Current step index (for logging).
			training (bool): If True, the subsequent sample method can do exploration.

		Returns:
			dict: The dictionary from sample() with new action, reward, etc.
		"""
		if new_step_idx is not None:
			self.step_counter = new_step_idx

		observed_reward = self.update(
			new_loss_vector=new_loss_vector,
			old_loss_vector=old_loss_vector,
			action_taken=action_taken,
			observed_reward=observed_reward,
			old_step_idx=old_step_idx,
			new_step_idx=new_step_idx,
		)
		self.last_observed_reward = observed_reward

		# Convert new_loss_vector to numpy
		new_loss_vector = safe_cast_to_numpy(new_loss_vector, dtype="float32")

		# Now return a new action
		return self.sample(new_loss_vector, training=training)

	def calculate_reward(self, old_loss_vector, new_loss_vector):
		"""
		Default reward function: a sample that tries to encourage improvements in loss.
		Larger negative delta -> larger reward. We clip values to [-1.0, 1.0].

		Args:
			old_loss_vector (Tensor): The old sub-losses.
			new_loss_vector (Tensor): The new sub-losses.

		Returns:
			Tensor: The reward, shape matching the objective dimension.
		"""
		old_loss_vector = safe_cast_to_tensor(old_loss_vector, dtype="float32")
		new_loss_vector = safe_cast_to_tensor(new_loss_vector, dtype="float32")

		# Example: compute the difference ratio
		delta_loss = (new_loss_vector - old_loss_vector)
		ratio = delta_loss / (old_loss_vector + 1e-8)
		n = (new_loss_vector * old_loss_vector).sqrt() * ratio.pow(3) * torch.e
		reward = (-100 * torch.tanh(n) * torch.e)
		reward = torch.where(torch.isnan(reward), fp64_zero, reward)
		reward = torch.clamp(reward, fp64_neg_one, fp64_one)

		return reward
