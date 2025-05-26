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

import time
import os
import threading # For datagen_thread
import multiprocessing as mp # For batcher_stop_event
from typing import List, Dict, Any, Optional
import accelerate
import numpy as np
import torch
import json
from functools import partial
import queue # For queue.Empty
import traceback # For printing stack traces

from birdie_rl.birdie_reward_model.reward_model import RewardModel
from birdie_rl.pipeline_generator import pipeline_data_generator
from birdie_rl.objectives import utils as objective_utils # Renamed to avoid conflict
from birdie_rl.load_objective import load_objective # For direct objective loading in validation


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
			obj["hash_str"] = objective_utils.sha_hash(obj)[:8]
		obj['nickname'] = f"{obj['name']}_{obj['hash_str']}"
	return x

def normalize_objective_probs(x):
	"""
	Normalizes the probabilities of objectives in a list of dictionaries.
	"""
	if not x: # Handle empty list
		return x
	total = sum(obj.get("prob", 1.0) for obj in x)

	if (total <= 0.0):
		equal_prob = 1.0 / len(x) if len(x) > 0 else 0.0
		for obj in x:
			obj["prob"] = equal_prob 
			obj["prob_initial"] = equal_prob
	else:
		for obj in x:
			initial_prob = obj.get("prob", 1.0) / total
			obj["prob"] = initial_prob 
			obj["prob_initial"] = initial_prob
	return x
	


def default_move_to_gpu_fn(batch: Dict[str, Any], accelerator: accelerate.Accelerator) -> Dict[str, torch.Tensor]:
	"""
	Moves each tensor in the batch to the appropriate device (GPU or otherwise).
	"""
	if not isinstance(batch, dict):
		print_fn = accelerator.print if accelerator else print
		print_fn(f"  FATAL EXCEPTION: BATCH not a dict? type(batch): {type(batch)},  batch: {str(batch)[:200]}", flush=True) 
		raise ValueError(f"  FATAL EXCEPTION: batch is not a dict. Got: {type(batch)}")
	
	device = accelerator.device if accelerator else torch.device("cpu")
	processed_batch = {}
	for k, v in batch.items():
		if isinstance(v, torch.Tensor):
			processed_batch[k] = v.to(device)
		elif isinstance(v, np.ndarray): 
			processed_batch[k] = torch.from_numpy(v).to(device)
		else:
			try: 
				processed_batch[k] = torch.tensor(v, dtype=torch.long).to(device)
			except Exception as e:
				print_fn = accelerator.print if accelerator else print
				print_fn(f"Warning: Could not convert value for key '{k}' to tensor. Type: {type(v)}. Error: {e}")
				processed_batch[k] = v # Keep original if conversion fails
	return processed_batch

class Birdie:
	"""
	Orchestrates the RL-based mixture sampling process in a simplified manner.
	"""

	def __init__(self, config: Optional[dict] = None):
		if config is None:
			config = {}

		self.config = config
		self.accelerator = config.get("accelerator", None)
		self.print_fn = self.accelerator.print if self.accelerator else print
		
		self.batch_size = config.get("batch_size", 8)
		self.dataset_callable = config.get("ds", None) 
		if not callable(self.dataset_callable):
			raise ValueError("'ds' in config must be a callable data generator function.")

		self.ds_validation_callable = config.get("ds_validation", self.dataset_callable)
		if not callable(self.ds_validation_callable):
			raise ValueError("'ds_validation' in config must be a callable data generator function.")

		self.num_workers = config.get("num_workers", 1) 
		self.sequence_length = config.get("sequence_length", 1024)
		self.steps_between_evaluations = config.get("steps_between_evaluations", 512)
		self.tokenizer = config.get("tokenizer", None)
		if self.tokenizer is None:
			raise ValueError("Tokenizer must be provided in config.")
		self.total_steps = config.get("total_steps", config.get("num_steps", 16_384))
		
		self.move_to_gpu_fn = config.get("move_to_gpu_fn", partial(default_move_to_gpu_fn, accelerator=self.accelerator))

		self.objectives = config.get("objectives", default_objectives_config)
		self.objectives = add_hashes(self.objectives)
		self.objectives = normalize_objective_probs(self.objectives)
		self.reward_signal_dims = int(config.get("reward_signal_dims", len(self.objectives)))

		self.validation_ds_per_objective: Dict[int, Dict[str, Any]] = {}
		self.validation_objectives = config.get("validation_objectives", self.objectives) 
		self.validation_objectives = add_hashes(self.validation_objectives)
		self.validation_objectives = normalize_objective_probs(self.validation_objectives)
		self.validation_sequence_length = int(config.get("validation_sequence_length", self.sequence_length))
		self.num_validation_batches = config.get("num_validation_batches", 2) 

		self.measured_num_batches_per_objective: Dict[str, int] = {}
		self._validation_samples_prepared = False
		self.get_validation_samples() # Prepare validation samples during init

		self.last_action = np.array([obj.get("prob", 1.0) for obj in self.objectives], dtype=np.float32)
		if np.sum(self.last_action) == 0 and len(self.last_action) > 0 : 
			self.last_action = np.array([1.0/len(self.last_action)] * len(self.last_action), dtype=np.float32)
		elif np.sum(self.last_action) > 0 :
			self.last_action /= np.sum(self.last_action)

		reward_model_config = {
			**config, 
			"reward_signal_dims": self.reward_signal_dims,
			"num_objectives": len(self.objectives),
			"hidden_dims": config.get("hidden_dims", (256, 256, 256, 256)), 
			"lr": config.get("lr", 5e-4), 
			"device": self.accelerator.device if self.accelerator else torch.device("cpu"),
			"accelerator": self.accelerator, 
		}
			
		self.reward_model = RewardModel(reward_model_config)

		pipeline_components = pipeline_data_generator(
			batch_size=self.batch_size,
			sequence_length=self.sequence_length,
			objectives_config=self.objectives,
			num_workers=self.num_workers,
			accelerator=self.accelerator,
			move_to_gpu_fn=self.move_to_gpu_fn,
			data_generator=self.dataset_callable, 
			split='train',
			config=self.config, 
			infinite_loop=True 
		)
		self.controller = pipeline_components[0]
		self.training_data_generator = pipeline_components[1]
		self.datagen_thread = pipeline_components[2] 
		self.batcher_stop_event = pipeline_components[3] 
		self.datagen_thread_stop_event = pipeline_components[4] if len(pipeline_components) > 4 else None

		self.current_step = 0
		self.old_loss_vector: Optional[np.ndarray] = None
		self.validation_key_to_losses: Dict[str, List[float]] = {}
		self.validation_num_fulfilled_keys = 0
		self.current_validation_losses: Optional[np.ndarray] = None
		self.last_reward_step_idx = -1 # Ensure eval happens at step 0 if steps_between_evaluations allows

	def _shutdown_pipeline_components(self, controller, datagen_thread, batcher_stop_event, datagen_thread_stop_event=None, pipeline_name="Training"):
		"""Helper to shut down a set of pipeline components."""
		self.print_fn(f"[Birdie] Shutting down {pipeline_name} pipeline components...")
		
		if batcher_stop_event and hasattr(batcher_stop_event, 'set') and not batcher_stop_event.is_set():
			self.print_fn(f"[Birdie] Setting {pipeline_name} batcher_stop_event.")
			try: batcher_stop_event.set()
			except Exception as e: self.print_fn(f"[Birdie Warning] Error setting {pipeline_name} batcher_stop_event: {e}")

		if datagen_thread_stop_event and hasattr(datagen_thread_stop_event, 'set') and not datagen_thread_stop_event.is_set():
			self.print_fn(f"[Birdie] Setting {pipeline_name} datagen_thread_stop_event.")
			try: datagen_thread_stop_event.set()
			except Exception as e: self.print_fn(f"[Birdie Warning] Error setting {pipeline_name} datagen_thread_stop_event: {e}")

		if controller and hasattr(controller, 'close'):
			self.print_fn(f"[Birdie] Closing {pipeline_name} controller...")
			try:
				controller.close(timeout_join=5.0) # Reduced timeout for faster shutdown attempt
				self.print_fn(f"[Birdie] {pipeline_name} controller closed.")
			except Exception as e: self.print_fn(f"[Birdie Warning] Error closing {pipeline_name} controller: {e}")
		
		if datagen_thread and hasattr(datagen_thread, 'is_alive') and datagen_thread.is_alive():
			self.print_fn(f"[Birdie] Joining {pipeline_name} datagen_thread...")
			try:
				datagen_thread.join(timeout=5.0)
				if datagen_thread.is_alive(): self.print_fn(f"[Birdie Warning] {pipeline_name} datagen_thread did not terminate.")
				else: self.print_fn(f"[Birdie] {pipeline_name} datagen_thread joined.")
			except Exception as e: self.print_fn(f"[Birdie Warning] Error joining {pipeline_name} datagen_thread: {e}")
		self.print_fn(f"[Birdie] {pipeline_name} pipeline shutdown sequence finished.")

	def close(self) -> None:
		self.print_fn("[Birdie] close() called.")
		self._shutdown_pipeline_components(
			self.controller, 
			self.datagen_thread, 
			self.batcher_stop_event,
			self.datagen_thread_stop_event,
			pipeline_name="Training"
		)
		# Validation pipelines are now generated and cleaned up within get_validation_samples directly
		self.print_fn("[Birdie] Main close sequence finished.")

	def __del__(self):
		if hasattr(self, 'controller') and self.controller is not None: 
			self.print_fn("[Birdie __del__] Instance being deleted. Ensuring close() is called.")
			self.close()

	def time_for_eval(self, step_idx: Optional[int] = None) -> bool:
		step = step_idx if step_idx is not None else self.current_step
		return (step % self.steps_between_evaluations) == 0 and step != self.last_reward_step_idx

	def get_next_training_sample(self) -> Any:
		self.current_step += 1
		try:
			batch = next(self.training_data_generator)
			if batch is None: 
				self.print_fn("[Birdie Error] Training data generator yielded None. This indicates premature termination.")
				raise RuntimeError("Training data generator stopped producing data (yielded None).")
			return batch
		except StopIteration:
			self.print_fn("[Birdie Error] Training data generator exhausted unexpectedly. This should not happen with infinite_loop=True for training.")
			raise RuntimeError("Training data generator stopped producing data (StopIteration).")
		except Exception as e:
			self.print_fn(f"[Birdie Error] Unexpected error in get_next_training_sample: {e}")
			traceback.print_exc()
			raise

	def get_validation_samples(self) -> Dict[int, Dict[str, Any]]:
		if self._validation_samples_prepared:
			return self.validation_ds_per_objective

		self.print_fn(f"[Birdie] Preparing validation samples directly for {len(self.validation_objectives)} objectives ({self.num_validation_batches} items each)...")
		self.validation_ds_per_objective.clear() 
		self.measured_num_batches_per_objective.clear()

		text_grabber = self.config.get("text_grabber_fn", lambda x: x.get("text", "") if isinstance(x, dict) else str(x))

		for objective_idx, objective_config_from_list in enumerate(self.validation_objectives):
			obj_name_for_log = objective_config_from_list.get('nickname', objective_config_from_list.get('name', f"val_obj_{objective_idx}"))
			# self.print_fn(f"  Processing validation for: {obj_name_for_log}")

			current_samples_for_this_objective = []
			
			try:
				data_gen_kwargs_for_val = dict(
					split='validation', 
					worker_id=0, 
					num_workers=1, 
					rng_seed=self.config.get("seed", int(time.time())) + objective_idx + 1000 
				)
				# data_gen_kwargs_for_val.update(self.config.get("data_generator_fn_kwarg_overrides", {}))
				
				if not callable(self.ds_validation_callable):
					self.print_fn(f"  ERROR: self.ds_validation_callable is not callable for {obj_name_for_log}. Skipping.")
					self.measured_num_batches_per_objective[obj_name_for_log] = 0
					self.validation_ds_per_objective[objective_idx] = {"objective": objective_config_from_list, "batches": []}
					continue
				
				# Create a fresh iterator for each objective to ensure data isolation and correct sharding/reset
				val_data_iter = iter(self.ds_validation_callable(**data_gen_kwargs_for_val))
			except Exception as e:
				self.print_fn(f"  ERROR creating data iterator for validation objective {obj_name_for_log}: {e}")
				traceback.print_exc()
				self.measured_num_batches_per_objective[obj_name_for_log] = 0
				self.validation_ds_per_objective[objective_idx] = {"objective": objective_config_from_list, "batches": []}
				continue
			
			# Load the objective instance
			# Make a deep copy of the objective config to avoid modifying the original list
			current_obj_config_dict = objective_config_from_list.copy()
			current_obj_config_dict["tokenizer"] = self.tokenizer
			current_obj_config_dict["remaining_space"] = self.validation_sequence_length 
			current_obj_config_dict["rng_seed"] = self.config.get("seed", int(time.time())) + objective_idx + 2000 # New seed for objective
			
			try:
				objective_instance = load_objective(current_obj_config_dict["name"], current_obj_config_dict)
			except Exception as e:
				self.print_fn(f"  ERROR loading objective {obj_name_for_log} for validation: {e}")
				traceback.print_exc()
				self.measured_num_batches_per_objective[obj_name_for_log] = 0
				self.validation_ds_per_objective[objective_idx] = {"objective": objective_config_from_list, "batches": []}
				continue
				
			items_generated_count = 0
			for i in range(self.num_validation_batches):
				try:
					raw_data_item = next(val_data_iter)
					text_sample = text_grabber(raw_data_item)
					if not text_sample or not isinstance(text_sample, str):
						self.print_fn(f"    Got invalid text sample (type: {type(text_sample)}) for {obj_name_for_log}, item {i+1}. Stopping for this objective.")
						break
				except StopIteration:
					self.print_fn(f"    Data iterator for {obj_name_for_log} exhausted before generating {self.num_validation_batches} samples (got {items_generated_count}).")
					break
				except Exception as e:
					self.print_fn(f"    Error getting text for {obj_name_for_log}, item {i+1}: {e}")
					traceback.print_exc()
					break

				transformed_sample = objective_instance(text_sample) 

				if transformed_sample.get("status") == "ok" and \
				   isinstance(transformed_sample.get("input_ids"), (list, np.ndarray)) and \
				   len(transformed_sample.get("input_ids")) > 0:
					
					final_batch_item = {}
					required_keys = ["input_ids", "label_ids", "attention_mask", "segment_ids"]
					input_ids_len = len(transformed_sample["input_ids"])

					for k in required_keys:
						val = transformed_sample.get(k)
						if k == "input_ids":
							val_np = np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32)
						elif k == "label_ids":
							# Ensure label_ids match input_ids length, padding with -100
							label_val = np.full(input_ids_len, -100, dtype=np.int32)
							if val is not None and len(val) > 0:
								actual_len = min(len(val), input_ids_len)
								label_val[:actual_len] = np.array(val[:actual_len], dtype=np.int32) if not isinstance(val, np.ndarray) else val[:actual_len].astype(np.int32)
							val_np = label_val
						elif k == "attention_mask":
							# Default attention mask (all 1s up to input_ids_len) if not provided
							val_np = np.ones(input_ids_len, dtype=np.int32) if val is None else (np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32))
						elif k == "segment_ids":
							# Default segment_ids (all 0s or 1s up to input_ids_len)
							val_np = np.zeros(input_ids_len, dtype=np.int32) if val is None else (np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32))
						else: # Should not happen with required_keys
							val_np = np.array(val, dtype=np.int32) if val is not None else np.zeros(input_ids_len, dtype=np.int32)
						
						# Pad or truncate to validation_sequence_length
						padded_val = np.full(self.validation_sequence_length, 0, dtype=np.int32)
						if k == "label_ids": padded_val.fill(-100)

						actual_len = min(len(val_np), self.validation_sequence_length)
						padded_val[:actual_len] = val_np[:actual_len]
						final_batch_item[k] = padded_val

					# This final_batch_item is a single sample, needs to be a batch of 1
					batch_of_one = {key: np.expand_dims(value, axis=0) for key, value in final_batch_item.items()}
					current_samples_for_this_objective.append(batch_of_one)
					items_generated_count +=1
				else:
					self.print_fn(f"    Objective {obj_name_for_log} failed for item {i+1}. Status: {transformed_sample.get('status')}, Msg: {transformed_sample.get('message')}")
			
			self.validation_ds_per_objective[objective_idx] = {
				"objective": objective_config_from_list, # Store original config from list
				"batches": current_samples_for_this_objective,
			}
			self.measured_num_batches_per_objective[obj_name_for_log] = items_generated_count
			if items_generated_count < self.num_validation_batches:
				self.print_fn(f"  Warning: Expected {self.num_validation_batches} val items for {obj_name_for_log}, but generated {items_generated_count}.")

		self._validation_samples_prepared = True
		self.print_fn("[Birdie] Direct validation samples preparation complete.")
		return self.validation_ds_per_objective

	def measure_validation_losses(self) -> List[Any]:
		if not self._validation_samples_prepared:
			self.get_validation_samples() 

		self.validation_key_to_losses = {} 
		self.validation_num_fulfilled_keys = 0

		flat_batches_to_eval = []
		if not self.validation_ds_per_objective:
			self.print_fn("[Birdie Warning] measure_validation_losses: validation_ds_per_objective is empty.")
			return []

		for objective_idx, data in self.validation_ds_per_objective.items():
			objective_config = data["objective"]
			key = objective_config.get("hash_str", objective_config.get("nickname", objective_config["name"]))
			
			if not data["batches"]:
				# self.print_fn(f"  No validation batches found for objective '{key}' (idx {objective_idx}). Skipping.")
				if self.measured_num_batches_per_objective.get(key, -1) == 0: 
					if key not in self.validation_key_to_losses: 
						 self.validation_key_to_losses[key] = [] 
				continue

			for batch_idx, batch_of_one in enumerate(data["batches"]):
				if batch_of_one is None: 
					self.print_fn(f"  Warning: Found None batch at index {batch_idx} for objective '{key}'. Skipping.")
					continue
				
				# Batch_of_one is already {key: np.array batch_dim=0}, ready for move_to_gpu_fn
				processed_batch = batch_of_one 
				if self.move_to_gpu_fn is not None:
					try:
						processed_batch = self.move_to_gpu_fn(processed_batch) 
					except Exception as e:
						self.print_fn(f"  Error moving batch to GPU for objective '{key}': {e}. Batch content: {str(processed_batch)[:200]}")
						continue 
				flat_batches_to_eval.append((key, processed_batch))
		
		return flat_batches_to_eval


	def log_validation_loss(self, key: str, loss: float, step_idx: Optional[int] = None) -> None:
		current_training_step = step_idx if step_idx is not None else self.current_step
		
		self.validation_key_to_losses.setdefault(key, []).append(loss)

		num_expected_batches_for_key = self.measured_num_batches_per_objective.get(key, 0)

		print(f"  key: {key}")
		print(f"  self.validation_num_fulfilled_keys: {self.validation_num_fulfilled_keys}")
		print(f"  len(self.validation_key_to_losses[{key}]): {len(self.validation_key_to_losses[key])}")
		
		if len(self.validation_key_to_losses[key]) >= num_expected_batches_for_key:
			if len(self.validation_key_to_losses[key]) == num_expected_batches_for_key:
				# Only increment if it's the first time reaching the exact count for this key in this eval cycle
				# This check is tricky if validation_num_fulfilled_keys is not reset per eval *before* this loop.
				# Let's assume validation_num_fulfilled_keys is reset correctly before calling measure_validation_losses.
				# A more robust way is to track fulfilled keys in a set for the current eval pass.
				# For now, this simple increment should work if called correctly.
				self.validation_num_fulfilled_keys += 1
			# self.print_fn(f"  Logged {len(self.validation_key_to_losses[key])}/{num_expected_batches_for_key} batches for objective '{key}'. Fulfilled keys: {self.validation_num_fulfilled_keys}/{len(self.validation_objectives)}")

		if self.validation_num_fulfilled_keys >= len(self.validation_objectives):
			# self.print_fn(f"All validation losses collected for step {current_training_step}.")
			
			mean_losses_dict = {
				obj_key: np.mean(losses_list) if losses_list else 0.0 
				for obj_key, losses_list in self.validation_key_to_losses.items()
			}
			
			current_losses_vector_list = []
			for obj_setting in self.objectives: 
				obj_eval_key = obj_setting.get("hash_str", obj_setting.get("nickname", obj_setting["name"]))
				loss_val = mean_losses_dict.get(obj_eval_key, 0.0) 
				current_losses_vector_list.append(loss_val)
			
			self.current_validation_losses = np.array(current_losses_vector_list, dtype=np.float32)

			if self.old_loss_vector is None: 
				self.old_loss_vector = self.current_validation_losses.copy()
				self.print_fn(f"Initialized old_loss_vector at step {current_training_step} with losses: {self.old_loss_vector}")
			else:
				new_action_probs = self.update_reward_model(
					action_taken=self.last_action,
					old_loss_vector=self.old_loss_vector,
					new_loss_vector=self.current_validation_losses,
					old_step_idx=self.last_reward_step_idx, 
					new_step_idx=current_training_step,   
				)
				self.last_action = new_action_probs 
				self.old_loss_vector = self.current_validation_losses.copy() 

				for i, obj_setting in enumerate(self.objectives):
					obj_setting["prob"] = float(np.round(new_action_probs[i], 4)) 
					obj_setting["loss"] = float(np.round(self.current_validation_losses[i], 4)) 

				if hasattr(self, 'controller') and self.controller:
					self.controller.update(self.objectives, clear_prefetched=False) 
				else:
					self.print_fn("Controller not available to update objectives.")
			
			self.last_reward_step_idx = current_training_step 
			self.validation_key_to_losses = {} 
			self.validation_num_fulfilled_keys = 0 


	def update_reward_model(
		self,
		action_taken: np.ndarray,
		old_loss_vector: np.ndarray,
		new_loss_vector: np.ndarray,
		old_step_idx: Optional[int] = None,
		new_step_idx: Optional[int] = None,
	) -> np.ndarray:
		new_action = self.reward_model.update( 
			action_taken=action_taken,
			old_loss_vector=old_loss_vector,
			new_loss_vector=new_loss_vector,
			old_step_idx=old_step_idx,
			new_step_idx=new_step_idx,
		)
		return new_action 

	def get_current_validation_losses(self) -> Optional[np.ndarray]:
		if self.current_validation_losses is not None:
			if len(self.current_validation_losses) != len(self.objectives):
				self.print_fn(f"  FATAL EXCEPTION: len(self.current_validation_losses): {len(self.current_validation_losses)},  len(self.objectives): {len(self.objectives)}")
				return None 
		return self.current_validation_losses

	def get_current_action(self) -> np.ndarray:
		if self.last_action is not None:
			if len(self.last_action) != len(self.objectives):
				self.print_fn(f"  FATAL EXCEPTION: len(self.last_action): {len(self.last_action)},  len(self.objectives): {len(self.objectives)}")
				return np.array([1.0/len(self.objectives)] * len(self.objectives), dtype=np.float32) if self.objectives else np.array([], dtype=np.float32)
		return self.last_action
	
	def get_verbose_action(self) -> tuple[Dict[str, Any], str]:
		ret_dict = {}
		current_action_probs = self.get_current_action() 

		def flatten_dict(d, parent_key='', sep='.'): 
			items = []
			for k, v in d.items():
				new_key = f"{parent_key}{sep}{k}" if parent_key else k
				if isinstance(v, dict): 
					items.extend(flatten_dict(v, new_key, sep=sep).items())
				else:
					items.append((new_key, v))
			return dict(items)

		flat_objectives_configs = [flatten_dict(obj_conf) for obj_conf in self.objectives]
		
		for idx, objective_config_flat in enumerate(flat_objectives_configs):
			serializable_config = {}
			for k,v in objective_config_flat.items():
				if k in ["tokenizer", "accelerator", "ds", "reward_fn"]: 
					continue
				try:
					json.dumps({k:v}) 
					serializable_config[k] = v
				except TypeError:
					serializable_config[k] = str(v) 

			name_key = serializable_config.get("nickname", serializable_config.get("name", f"objective_{idx}"))
			
			ret_dict[name_key] = {
				"current_sampling_probability": float(current_action_probs[idx]) if idx < len(current_action_probs) else 0.0,
				"config": {k:v for k,v in serializable_config.items() if k not in ["name", "nickname", "prob", "prob_initial", "loss"]} 
			}
			if "prob" in serializable_config: ret_dict[name_key]["config"]["original_prob_config"] = serializable_config["prob"]
			if "prob_initial" in serializable_config: ret_dict[name_key]["config"]["normalized_initial_prob"] = serializable_config["prob_initial"]
			if "loss" in serializable_config: ret_dict[name_key]["last_recorded_val_loss"] = serializable_config["loss"]

		formatted_ret_dict_str = json.dumps(ret_dict, indent=4, sort_keys=True)
		return ret_dict, formatted_ret_dict_str

