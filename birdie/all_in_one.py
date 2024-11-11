import os
import torch

from torch.optim import AdamW
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm

try:
	from birdie import gated_ssm_agent
except:
	import gated_ssm_agent

# import jax
import builtins

import numpy as np
import threading


filename = f"{__file__.rsplit('/', 1)[-1]}"

def print(*args, **kwargs):
	# Only allow process 0 to print
	# if jax.process_index() != 0:
	# 	return
	return builtins.print(f"{filename}  ", *args, **kwargs)


def make_cosine_decay_schedule(total_steps, min_val, max_val, warmup_steps=0, decay_factor=1, cook_steps=0, cook_val=1.0,):
	"""
	Computes a decaying value for a single cosine schedule at a specific step.

	Parameters:
		total_steps (int): Total number of steps over which the schedule spans.
		min_val (float): Minimum value of the cosine function.
		max_val (float): Maximum value of the cosine function.
		warmup_steps (int, optional): Number of steps for the warmup period. Defaults to 0.
		decay_factor (float, optional): Exponential decay factor applied to the amplitude. Defaults to 1.

	Returns:
		function: A function that computes the decaying value at a given step.
	"""
	adjusted_total_steps = total_steps - 1 - warmup_steps
	scaled_total_steps = (total_steps * 2) - 2

	def single_decaying_cosine_value(current_step):

		if (current_step < cook_steps):
			return cook_val
		
		current_step = current_step - warmup_steps
		current_step = np.minimum(current_step, adjusted_total_steps)
		current_step = np.maximum(0, current_step)

		amplitude = (max_val - min_val) / 2
		decayed_amplitude = amplitude * (decay_factor ** (current_step / scaled_total_steps))
		base_value = min_val + amplitude
		cosine_value = np.cos(np.pi * current_step / adjusted_total_steps)
		rv = base_value + decayed_amplitude * cosine_value

		rv = np.maximum(min_val, rv)
		rv = np.minimum(max_val, rv)
		return rv

	# Test assertions to ensure schedule correctness
	init_val = single_decaying_cosine_value(total_steps * 0.1)
	end_val = single_decaying_cosine_value(total_steps * 0.9)

	try:
		assert warmup_steps < total_steps, f"warmup_steps: {warmup_steps}, total_steps: {total_steps}"
		if min_val < max_val:
			assert init_val > end_val, f"init_val: {init_val}, end_val: {end_val}, min_val: {min_val}, max_val: {max_val}"
		else:
			assert init_val < end_val, f"init_val: {init_val}, end_val: {end_val}, min_val: {min_val}, max_val: {max_val}"
		assert np.isclose(single_decaying_cosine_value(total_steps), min_val), f"single_decaying_cosine_value(total_steps): {single_decaying_cosine_value(total_steps)}, min_val: {min_val}"

		# assert np.isclose(single_decaying_cosine_value(0), max_val), f"single_decaying_cosine_value(0): {single_decaying_cosine_value(0)}, max_val: {max_val}"
	except Exception as e:
		omsgs = [
			f"make_cosine_decay_schedule()",
			f"total_steps: {total_steps}",
			f"min_val: {min_val}",
			f"max_val: {max_val}",
			f"warmup_steps: {warmup_steps}",
			f"decay_factor: {decay_factor}",
			f"init_val: {init_val}",
			f"end_val: {end_val}",
			f"e: {e}",
		]
		print('\n'.join(omsgs))
		raise e

	return single_decaying_cosine_value




if __name__ == "__main__":
	np.set_printoptions(precision=3)
	torch.set_printoptions(precision=3, sci_mode=False,)
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def custom_format_fn(x):
	return f"{x:.3f}"

def to_numpy(x, dtype=np.float32):
	try:
		return dtype(x.detach().numpy())
	except:
		try:
			return dtype(x)
		except:
			return x


fp32_zero = torch.tensor(0.0, dtype=torch.float32)
fp64_zero = torch.tensor(0.0, dtype=torch.float64)
fp32_one = torch.tensor(1.0, dtype=torch.float32)
fp64_one = torch.tensor(1.0, dtype=torch.float64)

fp64_neg_one = torch.tensor(-1.0, dtype=torch.float64)




def safe_cast_to_tensor(x, dtype=None):
	
	try:
		if isinstance(x, torch.Tensor):
			ret_val = x.clone().detach()
		else:
			ret_val = torch.tensor(x)
	except Exception as e:
		print(f"  safe_cast_to_tensor() exception e: {e}")
		ret_val = torch.tensor(x)
	
	if dtype is not None:
		torch_dtype = {
			"float16": torch.float16,
			"float32": torch.float32,
			"float64": torch.float64,
			"int32": torch.int32,
			"int64": torch.int64,
		}[dtype]
		ret_val = ret_val.to(torch_dtype)


	return ret_val




def remap(x, in_min=None, in_max=None, out_min=-1.0, out_max=1.0):
	x = safe_cast_to_tensor(x)

	if in_min is None:
		in_min = torch.min(x)
	if in_max is None:
		in_max = torch.max(x)
	div = (in_max - in_min) + 1e-8
	return (x - in_min) * (out_max - out_min) / div + out_min




def loss_penalty(x, scale='exp'):
	# Disabled - no longer used after modifying the reward function itself to lower available rewards on objectives with very low losses
	return torch.ones_like(x)



class AgentBird:
	"""
	Class representing a bandit model for reinforcement learning with configurable objectives.
	"""
	num_samples = 1
	num_objectives = 6
	num_buckets = 1
	num_metrics = num_objectives * 2

	training_iter = 100_000
	num_iterations = 10000
	init_iters = 10
	debug_steps = 0
	decay_steps = int(num_iterations * 0.1)
	warmup_steps = int(decay_steps * 0.1)
	smoothing_window = -1
	last_only = 0
	dropout_rate = 0.0
	steps_between_evals = 250
	num_test_samples = 4096

	explore_classes = None

	lr = 5e-4

	lr_warmup_steps = -1
	weight_decay = 0.1
	div_loss_mult = 0

	must_explore_steps = 10

	train_y = None
	X = None

	np_rng_seed = 0

	top_n = 8
	# top_n = 1
	# hidden_dims = [32, 32] * 4
	# hidden_dims = [128, 128] * 4
	hidden_dims = [256, 256, 256, 256,]
	grok_iterations = 200


	rolling_rewards_limit = 16
	rolling_rewards = []

	n_steps_to_decay_per = 1000
	decay_per_n_steps = 0.1
	decay_per_n_steps = -1

	scheduler = None

	training_stats = {}




	def __init__(self, **kwargs):
		"""
		Initializes the AgentBird instance with given parameters or default settings.
		"""
		self.__dict__.update(kwargs)
		self.num_objectives = int(self.num_objectives)

		self.loaded_from_checkpoint = 0
		self.decay_steps = int(self.num_iterations * 0.45)
		self.warmup_steps = int(self.num_iterations * 0.10)
		cook_steps = int(self.num_iterations * 0.05)


		self.explore_schedule = make_cosine_decay_schedule(
			total_steps=self.decay_steps,
			warmup_steps=self.warmup_steps,
			cook_steps=cook_steps,
			min_val=0.25,
			max_val=0.50,
			cook_val=1.0,
		)

		self.np_rng = np.random.RandomState(self.np_rng_seed)
		self.explore_np_rng = np.random.RandomState(self.np_rng_seed + 1)

		self.explore_floats = self.explore_np_rng.random(50_000)
		self.explore_schedule_materialized = np.float32([self.explore_schedule(step) for step in range(len(self.explore_floats))])
		self.explore_results = (self.explore_floats < self.explore_schedule_materialized)

		explore_results = self.explore_results
		for explore_results_idx, (_explore_results) in enumerate(explore_results):
			ef = self.explore_floats[explore_results_idx]
			esm = self.explore_schedule_materialized[explore_results_idx]
			print(f"  explore_results[{explore_results_idx}]: {_explore_results},  esm: {esm},  ef: {ef}")
			if explore_results_idx >= self.num_iterations:
				break
			
		self.explore_history_dict_step_to_did_explore = {
			idx: result for idx, result in enumerate(self.explore_results)
		}

		self.num_metrics = self.num_objectives * 2

		self.model = gated_ssm_agent.MLPModel(
			input_dim=self.num_metrics,
			output_dim=self.num_objectives,
			hidden_dims=self.hidden_dims,
			dropout_rate=self.dropout_rate,
		)

		no_weight_decay_params = []
		default_params = []

		# for name, param in self.model.named_parameters():
		# 	(no_weight_decay_params if 'K_proj' in name else default_params).append(param)
		# param_groups = [
		# 	{'params': no_weight_decay_params, 'weight_decay': 0.0},
		# 	{'params': default_params, 'weight_decay': self.weight_decay},
		# ]

		# self.optimizer = AdamW(param_groups, lr=self.lr)

		self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.1,)
		self.prep_optimizer()

		self.step_counter = 0
		self.action_history = []
		self.reward_history_observed = []
		self.reward_history_estimated = []
		self.explore_history = []



	def prep_optimizer(self):
		if self.scheduler is None:
			# self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			# 	self.optimizer,
			# 	T_0=self.decay_steps,
			# 	T_mult=1,
			# 	eta_min=0.25,
			# 	last_epoch=-1,
			# )
			self.scheduler = torch.optim.lr_scheduler.CyclicLR(
				self.optimizer,
				base_lr=self.lr * 0.001,
				max_lr=self.lr,
				step_size_up=self.grok_iterations / 2,
				mode='exp_range',
			)




	def check_should_explore(self, step=None):
		step = (step or self.step_counter)
		result = self.explore_history_dict_step_to_did_explore[step]
		print(f"  check_should_explore({step}): {result}")
		return result



	def scale_actions(self, x, current_loss=None):
		x = safe_cast_to_tensor(x)
		if current_loss is not None:
			current_loss = safe_cast_to_tensor(current_loss)
			x *= loss_penalty(current_loss)
		if (x.min() < 0):
			x -= x.min()
		x = x / x.sum()
		return x
	

	def random_uniform(self, *shape: tuple,):
		return self.np_rng.random(np.prod(shape)).reshape(shape).astype(np.float32)



	def scale_actions_batch(self, x, current_loss=None):
		x = safe_cast_to_tensor(x)
		if current_loss is not None:
			current_loss = safe_cast_to_tensor(current_loss)
			x *= loss_penalty(current_loss)[None, None]
		if (x.min() < 0):
			x -= x.min(dim=-1, keepdim=True)[0]
		x = (x) / x.sum(dim=-1, keepdim=True)
		return x



	def predict_rewards(self, test_input):
		self.model.eval()
		with torch.no_grad():
			sampled_preds = self.model(test_input)[..., -1, :]

		return sampled_preds # torch.clamp(sampled_preds, -1, 1)



	def mass_sample(self, current_loss, num_test_samples, action=None, top_n=None):
		top_n = (top_n or self.top_n)

		broadcast_current_loss = safe_cast_to_tensor(current_loss)[None][None].repeat((num_test_samples*2, 1, 1)).float()

		if action is None:
			test_actions = torch.abs(torch.tensor(self.random_uniform(num_test_samples, 1, self.num_objectives)).float())
		else:
			test_actions = action[None][None].repeat((num_test_samples, 1, 1)).float()

		test_actions = self.scale_actions_batch(test_actions, current_loss)
		test_actions = torch.cat([test_actions, test_actions ** 4], dim=0).float()
		test_input = torch.cat([broadcast_current_loss, test_actions], dim=-1).float()

		if self.X is not None:
			repeated_X = self.X.repeat((broadcast_current_loss.shape[0], 1, 1))
			# noise_for_repeated_X = remap(torch.tensor(self.random_uniform(*repeated_X.shape)), out_min=0.98, out_max=1.02)
			# repeated_X *= noise_for_repeated_X
			test_input = torch.cat([repeated_X, test_input], dim=-2).float()

		predicted_rewards = self.predict_rewards(test_input)

		uids = np.unique(self.explore_classes)
		objective_scaling = np.zeros(self.num_objectives, dtype=np.float32)
		for uid in uids:
			matches = (self.explore_classes == uid)
			scaled_match_value = 1 / np.sum(np.int32(matches))
			objective_scaling = np.where(matches, scaled_match_value, objective_scaling)
		predicted_rewards *= objective_scaling[None] * len(uids)

		per_sample_predicted_rewards = predicted_rewards.sum(dim=1)
		top_n_indices = torch.argsort(per_sample_predicted_rewards, dim=-1, descending=False)[-top_n:]

		selected_rewards = predicted_rewards[top_n_indices]
		estimated_reward = selected_rewards.mean() # / len(uids)

		selected_actions = test_actions[top_n_indices, -1]
		best_action = selected_actions.mean(dim=0)

		return best_action, estimated_reward, predicted_rewards, test_actions



	def sample(self, current_loss, num_test_samples=4096, training=True, force_explore=False, watchdog_counter=2):
		current_loss = safe_cast_to_tensor(current_loss, dtype="float32")
		should_explore = force_explore or (training and self.check_should_explore(self.step_counter))

		if should_explore:

			# num_choices = self.np_rng.randint(1, max(3, len(np.unique(self.explore_classes)) // 2 + 1))
			num_choices = self.np_rng.randint(1, 3)
			explore_action = np.zeros(self.num_objectives, dtype=np.float32)

			uids = np.unique(self.explore_classes)
			selected_uids = self.np_rng.choice(uids, num_choices)
			for uid in selected_uids:
				matches = (self.explore_classes == uid)
				scaled_match_value = self.np_rng.random(np.sum(matches))
				scaled_match_value /= np.sum(scaled_match_value)
				explore_action[matches] = scaled_match_value

			# shared_unexplored_objective_probability = remap(self.random_uniform(1)[0], 0.0, 1.0, out_min=0.01, out_max=0.1)
			shared_unexplored_objective_probability = (len(self.explore_classes) / (len(self.explore_classes) - num_choices)) * 0.01 * num_choices
			for uid in uids:
				if uid not in selected_uids:
					matches = (self.explore_classes == uid)
					# scaled_match_value = shared_unexplored_objective_probability / np.sum(matches)

					scaled_match_value = current_loss[matches] * shared_unexplored_objective_probability / np.sum(matches)
					# scaled_match_value = current_loss[matches]
					# scaled_match_value /= scaled_match_value.sum()
					explore_action[matches] = scaled_match_value

			# explore_action = np.where(explore_action == 0, unexplored_objective_probability, explore_action)
			explore_action = safe_cast_to_tensor(explore_action, dtype="float32")
			explore_action *= current_loss
			explore_action /= explore_action.sum()
			explore_action = safe_cast_to_tensor(explore_action, dtype="float32")
			print(f"  explore_action: {explore_action}")
		else:
			explore_action = None

		(best_action, estimated_reward, sampled_preds, test_actions) = self.mass_sample(
			current_loss=current_loss, num_test_samples=num_test_samples, action=explore_action
		)


		# if training and not should_explore and sampled_preds.std(dim=-2).mean() < 1e-4 and watchdog_counter > 0:
		# 	return self.sample(current_loss, num_test_samples, training, force_explore=True, watchdog_counter=watchdog_counter - 1)
		
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



	def append_to_history(self, action_taken, old_loss, new_loss, observed_reward):
		if observed_reward is None:
			observed_reward = self.calculate_reward(old_loss=old_loss, new_loss=new_loss,)

		# Create new entry for X
		new_x_entry = torch.cat([old_loss[None], action_taken[None]], dim=1).float()[None]
		if self.X is None:
			self.X = new_x_entry
		else:
			self.X = torch.cat([self.X, new_x_entry], dim=-2).float()

		# Add observed reward to train_y
		observed_reward = observed_reward[None][None]
		if self.train_y is None:
			self.train_y = observed_reward
		else:
			self.train_y = torch.cat([self.train_y, observed_reward], dim=-2).float()

		return observed_reward



	def update(self, new_loss=None, old_loss=None, action_taken=None, observed_reward=None, step_idx=None):
		if action_taken is None:
			action_taken = self.action_history[-1]
		action_taken = safe_cast_to_tensor(action_taken, dtype="float32")

		old_loss = safe_cast_to_tensor(old_loss, dtype="float32")
		new_loss = safe_cast_to_tensor(new_loss, dtype="float32")

		observed_reward = self.append_to_history(
			action_taken=action_taken, old_loss=old_loss, new_loss=new_loss, observed_reward=observed_reward
		)

		# if self.X.shape[-2] <= 1:
		# 	return observed_reward

		self.model.train()
		self.prep_optimizer()

		random_noise = self.random_uniform(self.grok_iterations, *self.X.shape)
		random_noise = remap(random_noise, out_min=0.98, out_max=1.02)
		random_noise = safe_cast_to_tensor(random_noise, dtype="float32")

		label_noise = self.random_uniform(self.grok_iterations, *self.train_y.shape)
		label_noise = remap(label_noise, out_min=0.98, out_max=1.02)
		label_noise = safe_cast_to_tensor(label_noise, dtype="float32")

		for step_idx in tqdm(range(self.grok_iterations), desc="Training Agent Bird...", leave=False):
			x = (self.X * random_noise[step_idx])
			y = (self.train_y * label_noise[step_idx])
			# x = self.X
			# y = self.train_y
			loss = ((self.model(x) - y) ** 2).mean()
			loss.backward()
			clip_grad_norm_(self.model.parameters(), max_norm=1.0)
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.scheduler.step()

		self.step_counter += 1

		return observed_reward
	
	
	def update_and_sample(self, new_loss=None, old_loss=None, action_taken=None, observed_reward=None, training=True, step_idx=None):
		observed_reward = self.update(new_loss=new_loss, old_loss=old_loss, action_taken=action_taken, observed_reward=observed_reward)
		return self.sample(new_loss, training=training)


	def calculate_reward(self, old_loss, new_loss,):
		old_loss = safe_cast_to_tensor(old_loss, dtype="float32")
		new_loss = safe_cast_to_tensor(new_loss, dtype="float32")

		delta_loss = (new_loss - old_loss)
		rv = (delta_loss / (old_loss + 1e-8))
		n = ((new_loss * old_loss).sqrt() * rv.pow(3) * torch.e)
		reward = (-100 * torch.tanh(n) * torch.e)

		reward = torch.where(torch.isnan(reward), fp64_zero, reward)
		reward = torch.clamp(reward, fp64_neg_one, fp64_one)

		return reward



class Birdie:
	def __init__(self, config):
		self.dataset = config['dataset']
		self.objectives = config['indexed_pretraining_objectives']
		self.num_objectives = len(self.objectives)
		self.explore_classes = config['indexed_unique_objective_ids']
		self.steps_between_evaluations = config['steps_between_evaluations']

		self.total_steps = config['total_steps']

		self.agent = AgentBird(
			num_objectives=self.num_objectives,
			explore_classes=self.explore_classes,
			num_iterations=self.total_steps,
		)

		self.train_dataset = config['dataset']['train']
		self.validation_dataset = config['dataset']['validation']
		self.batch_size = config['batch_size']
		self.wrap_dataset()
		self.max_sequence_length = config['max_sequence_length']

		self.reward_scaling_vector = config['reward_scaling_vector']
		self.action = config['reward_scaling_vector']

		self.new_losses = []
		self.old_losses = []
		self.step_counter = 0
		self.random_generator = np.random.RandomState(0)

		self.prefetch_queue = threading.Queue(maxsize=5)
		self.prefetch_thread = threading.Thread(target=self.prefetch_samples)
		self.prefetch_thread.start()

		
		self.validation_samples_per_objective = 8
		validation_samples = [next(self.ds_valid_iterator) for _ in range(self.validation_samples_per_objective*2)]

		all_validation_samples = []
		for i in range(self.num_objectives):
			current_objective = self.objectives[i]
			just_added = 0

			for next_sample in validation_samples:
				print(f"  Working on objective {i} - {just_added} of {self.validation_samples_per_objective}") 

				result = current_objective(next_sample, remaining_space=self.max_sequence_length)

				if result['status'] == "error":
					continue

				input_ids = np.pad(result['input_ids'], (0, self.max_sequence_length - len(result['input_ids'])), constant_values=0)
				labels = np.pad(result['labels'], (0, self.max_sequence_length - len(result['labels'])), constant_values=0)

				loss_mask = np.zeros(len(input_ids) + len(labels), dtype=np.float32)
				loss_mask[len(input_ids):len(labels)] = 1

				all_validation_samples.append({'input_ids': input_ids, 'labels': labels, 'loss_mask': loss_mask})
				just_added += 1

				if just_added >= self.validation_samples_per_objective:
					break

			while just_added < self.validation_samples_per_objective:

				print(f"  Working on objective {i} - {just_added} of {self.validation_samples_per_objective}") 

				next_sample = next(self.ds_valid_iterator)
				validation_samples.append(next_sample)

				result = current_objective(next_sample, remaining_space=self.max_sequence_length)
				if result['status'] == "error":
					continue

				input_ids = np.pad(result['input_ids'], (0, self.max_sequence_length - len(result['input_ids'])), constant_values=0)
				labels = np.pad(result['labels'], (0, self.max_sequence_length - len(result['labels'])), constant_values=0)

				loss_mask = np.zeros(len(input_ids) + len(labels), dtype=np.float32)
				loss_mask[len(input_ids):len(labels)] = 1

				all_validation_samples.append({'input_ids': input_ids, 'labels': labels, 'loss_mask': loss_mask})

				just_added += 1
				


		





	def time_for_eval(self):
		return self.step_counter % self.steps_between_evaluations == 0

	def update_losses(self, loss):
		if isinstance(self.old_losses, list):
			self.old_losses.append(loss)
		else:
			self.new_losses.append(loss)
			if len(self.new_losses) >= self.num_objectives * self.validation_samples_per_objective:
				self.new_losses = np.float32(self.new_losses)

				average_losses = []
				for i in range(0, self.num_objectives * self.validation_samples_per_objective, self.validation_samples_per_objective):
					average_losses.append(np.mean(self.new_losses[i::self.num_objectives]))
				self.new_losses = np.float32(average_losses)

				result = self.agent.update_and_sample(new_loss=self.new_losses, old_loss=self.old_losses, actions_taken=self.action)
				self.action = result['action']
				self.old_losses = self.new_losses
				self.new_losses = []

	def wrap_dataset(self):
		def generator(dataset):
			while True:
				iterator = iter(dataset)
				for item in iterator:
					yield item['text']

		self.ds_train_iterator = generator(self.train_dataset)
		self.ds_valid_iterator = generator(self.validation_dataset)

	
	def _get_next_training_sample(self):
		'''
			Basic example, adding padding and such soon.
		'''
		while True:
			samples = []
			for _ in range(self.batch_size):
				idx = self.random_generator.choice(range(self.num_objectives), p=self.action)
				current_objective = self.objectives[idx]
				next_sample = next(self.ds_train_iterator)
				result = current_objective(next_sample, remaining_space=self.max_sequence_length)

				if result['status'] == "error":
					return self.get_next_training_sample()

				input_ids = np.pad(result['input_ids'], (0, self.max_sequence_length - len(result['input_ids'])), constant_values=0)
				labels = np.pad(result['labels'], (0, self.max_sequence_length - len(result['labels'])), constant_values=0)

				loss_mask = np.zeros(len(input_ids) + len(labels), dtype=np.float32)
				loss_mask[len(input_ids):len(labels)] = 1

				samples.append({'input_ids': input_ids, 'labels': labels, 'loss_mask': loss_mask})

				self.step_counter += 1
			batched_samples = {'input_ids': np.array([x['input_ids'] for x in samples]), 'labels': np.array([x['labels'] for x in samples]), 'loss_mask': np.array([x['loss_mask'] for x in samples])}
			self.prefetch_queue.put(batched_samples)
	
	def get_next_training_sample(self):
		return self.prefetch_queue.get()

	def get_validation_samples(self):
		return self.validation_samples
	
	def get_validation_losses(self):
		return self.old_losses









if __name__ == "__main__":
	total_steps = 200
	sched = make_cosine_decay_schedule(
		total_steps=total_steps,
		min_val=0.25,
		max_val=1.0,
		decay_factor=1.0,
		warmup_steps=(total_steps*0.1),
	)

	for i in range(total_steps * 2):
		interp = sched(i)
		print(f"  interp {i}: {interp}")

if __name__ == "__main__":
	arg_dict = {}

	num_objs = 16
	num_iterations = 100

	seeded_np_rng = np.random.RandomState(0)
	losses = np.zeros(num_objs, dtype=np.float32)*2 + 4

	probs = np.ones(num_objs, np.float32) / num_objs
	prob_uid_map = seeded_np_rng.choice(np.arange(min(5, num_objs-1)), size=num_objs, replace=True)
	prob_uid_map = sorted(prob_uid_map.tolist())
	prob_uid_map = np.array(prob_uid_map, dtype=np.int32)

	total_steps = num_iterations
	birdie_kwargs = dict(
		num_objectives=len(probs),
		explore_classes=prob_uid_map,
		num_iterations=total_steps,
		pid=0,
		# pid=jax.process_index(),
	)

	bandit = AgentBird(**birdie_kwargs)

	def simulate_train_get_new_loss(loss, action):
		"""
		Simulates a training step and computes the new loss based on the action taken.

		Args:
			loss (np.ndarray): Current loss values.
			action (torch.Tensor): Action taken by the bandit.

		Returns:
			np.ndarray: Updated loss values.
		"""
		# sub = seeded_np_rng.random(num_objs) * 0.1
		action = action.cpu().numpy().copy() * 0.4

		loss = loss.copy()




		for uid in np.unique(prob_uid_map)[::-1]:
			matches = (prob_uid_map == uid)
			magic_sub = np.mean(action[matches])
			loss -= magic_sub
			break

		prob_uid_avg = {}
		for uid in np.unique(prob_uid_map[0:1]):
			matches = (prob_uid_map == uid)
			loss[matches] -= magic_sub*2


		for uid in np.unique(prob_uid_map[1:-1]):
			matches = (prob_uid_map == uid)
			loss[matches] += np.mean(action[matches])

		print(f"  prob_uid_map: {prob_uid_map}")
		prob_uid_avg = {}
		for uid in np.unique(prob_uid_map):
			matches = (prob_uid_map == uid)
			sub = np.mean(action[matches])*0.05 + action[matches]*0.1
			# loss[matches] -= sub

			aveerage_action = (action[matches]/0.4).sum()
			prob_uid_avg[uid] = aveerage_action
		
		for prob_uid_avg_idx, (key, value) in enumerate(prob_uid_avg.items()):
			print(f"  prob_uid_avg[{key}]: {value:0.2%}")
			

		# sub = np.mean(action[-1:])
		# sub -= np.mean(action[:-1])
		return np.maximum(0.0, loss)



	def get_action():
		"""
		Generates a random action.

		Returns:
			torch.Tensor: A tensor representing the action.
		"""
		action = safe_cast_to_tensor(seeded_np_rng.random(num_objs), dtype="float32")
		action /= action.sum()
		return action

	actions = get_action()
	og_loss = losses

	historical_actions = []


	for iteration_idx in range(num_iterations):
		new_loss = simulate_train_get_new_loss(og_loss, actions)


		results = bandit.update_and_sample(old_loss=og_loss, new_loss=new_loss, action_taken=actions)

		if iteration_idx >= 3:
			actions = results['action']

		estimated_reward = results['estimated_reward']
		explored = results['explored']

		print(f"\n" * 3, end='',)
		
		if not explored:
			historical_actions.append(actions)
			print(f"  iteration_idx: {iteration_idx}")
			print(f"  explored: {explored}")
			print(f"  og_loss: {og_loss}")
			print(f"  new_loss: {new_loss}")
			print(f"  actions: {actions}")
			print(f"  np.mean(actions): {np.mean(np.float32(historical_actions[-64:]), axis=0)}")
			print(f"  estimated_reward: {estimated_reward}")

		og_loss = new_loss

	print("Training complete.")
