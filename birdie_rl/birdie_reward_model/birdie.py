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
			validation_batches = birdie.measure_validation_losses()
			for objective_name, batch_content in validation_batches:
				loss = model_forward_pass(batch_content) # Your model's forward pass
				birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step)
			birdie._maybe_update_reward_model(current_training_step=step) # Explicitly try to update agent
			
		sample = birdie.get_next_training_sample()
		# pass sample to your model, do forward/backward
		...
"""

import time
import os
import threading # For datagen_thread
import multiprocessing as mp # For batcher_stop_event
from typing import List, Dict, Any, Optional, Set # Added Set
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
		# Ensure nickname is consistently generated using name and hash
		obj['nickname'] = f"{obj.get('name', f'obj_{i}')}_{obj['hash_str']}"
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
		config['min_seq_len_for_packing'] = config.get("min_seq_len_for_packing", config.get("minimum_sequence_length", 256))
		assert( config['min_seq_len_for_packing'] < config['sequence_length'] )

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

		# Training objectives
		self.objectives = config.get("objectives", default_objectives_config)
		self.objectives = add_hashes(self.objectives) # Ensures nicknames are consistent
		self.objectives = normalize_objective_probs(self.objectives)
		self.reward_signal_dims = int(config.get("reward_signal_dims", len(self.objectives)))


		# Validation objectives and samples
		self.validation_ds_per_objective: Dict[str, Dict[str, Any]] = {} # Key is now string (nickname/hash)
		self.validation_objectives = config.get("validation_objectives", self.objectives) 
		self.validation_objectives = add_hashes(self.validation_objectives) # Ensures nicknames are consistent
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
			"reward_signal_dims": self.reward_signal_dims, # Should match len(self.objectives) for agent input
			"num_objectives": len(self.objectives), # Agent's action space dimension
			"hidden_dims": config.get("hidden_dims", (256, 256, 256, 256)), 
			"lr": config.get("lr", 5e-4), 
			"device": self.accelerator.device if self.accelerator else torch.device("cpu"),
			"accelerator": self.accelerator, 
		}
			
		self.reward_model = RewardModel(reward_model_config)

		pipeline_components = pipeline_data_generator(
			batch_size=self.batch_size,
			sequence_length=self.sequence_length,
			objectives_config=self.objectives, # Use training objectives for pipeline
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
		self._fulfilled_keys_this_cycle: Set[str] = set() # Tracks keys fulfilled in current eval cycle
		self.current_validation_losses: Optional[np.ndarray] = None
		self.last_reward_step_idx = -1 

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
				controller.close(timeout_join=5.0) 
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

	def get_validation_samples(self) -> Dict[str, Dict[str, Any]]: # Key is now string
		if self._validation_samples_prepared:
			return self.validation_ds_per_objective

		self.print_fn(f"[Birdie] Preparing validation samples directly for {len(self.validation_objectives)} objectives ({self.num_validation_batches} items each)...")
		self.validation_ds_per_objective.clear() 
		self.measured_num_batches_per_objective.clear()

		text_grabber = self.config.get("text_grabber_fn", lambda x: x.get("text", "") if isinstance(x, dict) else str(x))

		for objective_idx, objective_config_from_list in enumerate(self.validation_objectives):
			# Use nickname as the primary key for consistency
			obj_key = objective_config_from_list.get('nickname', 
						objective_config_from_list.get('hash_str', 
						objective_config_from_list.get('name', f"val_obj_{objective_idx}")))
			
			current_samples_for_this_objective = []
			
			try:
				data_gen_kwargs_for_val = dict(
					split='validation', 
					worker_id=0, 
					num_workers=1, 
					rng_seed=self.config.get("seed", int(time.time())) + objective_idx + 1000 
				)
				
				if not callable(self.ds_validation_callable):
					self.print_fn(f"  ERROR: self.ds_validation_callable is not callable for {obj_key}. Skipping.")
					self.measured_num_batches_per_objective[obj_key] = 0
					self.validation_ds_per_objective[obj_key] = {"objective": objective_config_from_list, "batches": []}
					continue
				
				val_data_iter = iter(self.ds_validation_callable(**data_gen_kwargs_for_val))
			except Exception as e:
				self.print_fn(f"  ERROR creating data iterator for validation objective {obj_key}: {e}")
				traceback.print_exc()
				self.measured_num_batches_per_objective[obj_key] = 0
				self.validation_ds_per_objective[obj_key] = {"objective": objective_config_from_list, "batches": []}
				continue
			
			current_obj_config_dict = objective_config_from_list.copy()
			current_obj_config_dict["tokenizer"] = self.tokenizer
			current_obj_config_dict["remaining_space"] = self.validation_sequence_length 
			current_obj_config_dict["rng_seed"] = self.config.get("seed", int(time.time())) + objective_idx + 2000 
			
			try:
				objective_instance = load_objective(current_obj_config_dict["name"], current_obj_config_dict)
			except Exception as e:
				self.print_fn(f"  ERROR loading objective {obj_key} for validation: {e}")
				traceback.print_exc()
				self.measured_num_batches_per_objective[obj_key] = 0
				self.validation_ds_per_objective[obj_key] = {"objective": objective_config_from_list, "batches": []}
				continue
				
			items_generated_count = 0
			for i in range(self.num_validation_batches):
				try:
					raw_data_item = next(val_data_iter)
					text_sample = text_grabber(raw_data_item)
					if not text_sample or not isinstance(text_sample, str):
						self.print_fn(f"    Got invalid text sample (type: {type(text_sample)}) for {obj_key}, item {i+1}. Stopping for this objective.")
						break
				except StopIteration:
					self.print_fn(f"    Data iterator for {obj_key} exhausted before generating {self.num_validation_batches} samples (got {items_generated_count}).")
					break
				except Exception as e:
					self.print_fn(f"    Error getting text for {obj_key}, item {i+1}: {e}")
					traceback.print_exc()
					break

				transformed_sample = objective_instance(text_sample) 

				if transformed_sample.get("status") == "ok" and \
				   isinstance(transformed_sample.get("input_ids"), (list, np.ndarray)) and \
				   len(transformed_sample.get("input_ids")) > 0:
					
					final_batch_item = {}
					required_keys = ["input_ids", "label_ids", "attention_mask", "segment_ids"]
					input_ids_len = len(transformed_sample["input_ids"])

					for k_req in required_keys:
						val = transformed_sample.get(k_req)
						if k_req == "input_ids":
							val_np = np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32)
						elif k_req == "label_ids":
							label_val = np.full(input_ids_len, -100, dtype=np.int32)
							if val is not None and len(val) > 0:
								actual_len = min(len(val), input_ids_len)
								label_val[:actual_len] = np.array(val[:actual_len], dtype=np.int32) if not isinstance(val, np.ndarray) else val[:actual_len].astype(np.int32)
							val_np = label_val
						elif k_req == "attention_mask":
							val_np = np.ones(input_ids_len, dtype=np.int32) if val is None else (np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32))
						elif k_req == "segment_ids":
							val_np = np.zeros(input_ids_len, dtype=np.int32) if val is None else (np.array(val, dtype=np.int32) if not isinstance(val, np.ndarray) else val.astype(np.int32))
						else: 
							val_np = np.array(val, dtype=np.int32) if val is not None else np.zeros(input_ids_len, dtype=np.int32)
						
						padded_val = np.full(self.validation_sequence_length, 0, dtype=np.int32)
						if k_req == "label_ids": padded_val.fill(-100)

						actual_len = min(len(val_np), self.validation_sequence_length)
						padded_val[:actual_len] = val_np[:actual_len]
						final_batch_item[k_req] = padded_val

					batch_of_one = {key_fin: np.expand_dims(value_fin, axis=0) for key_fin, value_fin in final_batch_item.items()}
					current_samples_for_this_objective.append(batch_of_one)
					items_generated_count +=1
				else:
					self.print_fn(f"    Objective {obj_key} failed for item {i+1}. Status: {transformed_sample.get('status')}, Msg: {transformed_sample.get('message')}")
			
			self.validation_ds_per_objective[obj_key] = { # Use string key
				"objective": objective_config_from_list, 
				"batches": current_samples_for_this_objective,
			}
			self.measured_num_batches_per_objective[obj_key] = items_generated_count
			if items_generated_count < self.num_validation_batches:
				self.print_fn(f"  Warning: Expected {self.num_validation_batches} val items for {obj_key}, but generated {items_generated_count}.")

		self._validation_samples_prepared = True
		self.print_fn("[Birdie] Direct validation samples preparation complete.")
		return self.validation_ds_per_objective

	def measure_validation_losses(self) -> List[Any]:
		if not self._validation_samples_prepared:
			self.get_validation_samples() 

		self.validation_key_to_losses = {} 
		self.validation_num_fulfilled_keys = 0
		self._fulfilled_keys_this_cycle.clear() # Reset the set for tracking fulfilled keys in this cycle

		flat_batches_to_eval = []
		if not self.validation_ds_per_objective:
			self.print_fn("[Birdie Warning] measure_validation_losses: validation_ds_per_objective is empty.")
			return []

		for obj_key, data in self.validation_ds_per_objective.items(): # Iterate using string keys
			# objective_config = data["objective"] # Not strictly needed here, key is enough
			
			if not data["batches"]:
				if self.measured_num_batches_per_objective.get(obj_key, -1) == 0: 
					if obj_key not in self._fulfilled_keys_this_cycle: 
						 self.validation_num_fulfilled_keys += 1
						 self._fulfilled_keys_this_cycle.add(obj_key)
				continue

			for batch_idx, batch_of_one in enumerate(data["batches"]):
				if batch_of_one is None: 
					self.print_fn(f"  Warning: Found None batch at index {batch_idx} for objective '{obj_key}'. Skipping.")
					continue
				
				processed_batch = batch_of_one 
				if self.move_to_gpu_fn is not None:
					try:
						processed_batch = self.move_to_gpu_fn(processed_batch) 
					except Exception as e:
						self.print_fn(f"  Error moving batch to GPU for objective '{obj_key}': {e}. Batch content: {str(processed_batch)[:200]}")
						continue 
				flat_batches_to_eval.append((obj_key, processed_batch))
		
		return flat_batches_to_eval


	def log_validation_loss(self, key: str, loss: float, step_idx: Optional[int] = None) -> None:
		# current_training_step = step_idx if step_idx is not None else self.current_step # Not used here
		
		self.validation_key_to_losses.setdefault(key, []).append(loss)
		num_expected_batches_for_key = self.measured_num_batches_per_objective.get(key, 0)

		# Debug prints (can be removed or controlled by verbosity later)
		# self.print_fn(f"  log_validation_loss for key: {key}, loss: {loss:.4f}")
		# self.print_fn(f"    len_losses: {len(self.validation_key_to_losses[key])}, expected_batches: {num_expected_batches_for_key}")
		# self.print_fn(f"    fulfilled_keys_count_before_check: {self.validation_num_fulfilled_keys}, _fulfilled_keys_this_cycle: {self._fulfilled_keys_this_cycle}")

		if key not in self._fulfilled_keys_this_cycle:
			if len(self.validation_key_to_losses[key]) >= num_expected_batches_for_key:
				# This key is now fulfilled for this cycle. Increment only if it actually expected batches or expected 0 and got 0.
				if num_expected_batches_for_key > 0: # Standard case: batches were expected and now logged
					self.validation_num_fulfilled_keys += 1
					self._fulfilled_keys_this_cycle.add(key)
					# self.print_fn(f"    Key {key} fulfilled ({num_expected_batches_for_key} batches). Fulfilled keys count: {self.validation_num_fulfilled_keys}")
				elif num_expected_batches_for_key == 0 and not self.validation_key_to_losses.get(key, []): 
					# Edge case: 0 batches were expected, and 0 were logged (list is empty or key not present)
					self.validation_num_fulfilled_keys += 1
					self._fulfilled_keys_this_cycle.add(key)
					# self.print_fn(f"    Key {key} (0 expected batches) fulfilled. Fulfilled keys count: {self.validation_num_fulfilled_keys}")
        # The decision to update the reward model is now handled by _maybe_update_reward_model,
        # called externally after all losses for an eval cycle are logged.

	def _maybe_update_reward_model(self, current_training_step: int) -> bool:
		"""
		Checks if all validation objectives are fulfilled and, if so, updates the reward model.
		This should be called AFTER all log_validation_loss calls for an evaluation cycle.
		"""
		if self.validation_num_fulfilled_keys >= len(self.validation_objectives):
			self.print_fn(f"[Birdie] All {self.validation_num_fulfilled_keys}/{len(self.validation_objectives)} validation objective losses collected for step {current_training_step}. Updating reward model.")

			mean_losses_dict: Dict[str, float] = {}
			for obj_key_from_val, losses_list in self.validation_key_to_losses.items():
				mean_losses_dict[obj_key_from_val] = np.mean(losses_list) if losses_list else 0.0
			
			current_losses_vector_list = []
			# The loss vector fed to the agent should correspond to self.objectives (training objectives)
			# and be in the same order.
			for train_obj_setting in self.objectives: 
				# Determine the key used during validation for this training objective
				eval_key_for_train_obj = train_obj_setting.get("nickname", 
										train_obj_setting.get("hash_str", 
										train_obj_setting.get("name")))
				
				loss_val = mean_losses_dict.get(eval_key_for_train_obj, 0.0) 
				current_losses_vector_list.append(loss_val)
			
			self.current_validation_losses = np.array(current_losses_vector_list, dtype=np.float32)

			if len(self.current_validation_losses) != len(self.objectives):
				self.print_fn(f"[Birdie CRITICAL ERROR] Mismatch between current_validation_losses length ({len(self.current_validation_losses)}) "
							  f"and number of training objectives ({len(self.objectives)}). This can happen if validation_objectives "
							  f"and objectives have different key structures or counts. Agent update might be compromised.")
				# Fallback: if lengths mismatch, try to use a zero vector or skip update to prevent crash
				# For now, we'll proceed, but this indicates a config issue.
				# A more robust solution would be to ensure a clear mapping or handle this more gracefully.
				if len(self.current_validation_losses) < len(self.objectives):
					padding = np.zeros(len(self.objectives) - len(self.current_validation_losses), dtype=np.float32)
					self.current_validation_losses = np.concatenate((self.current_validation_losses, padding))
				elif len(self.current_validation_losses) > len(self.objectives):
					self.current_validation_losses = self.current_validation_losses[:len(self.objectives)]


			if self.old_loss_vector is None: 
				self.old_loss_vector = self.current_validation_losses.copy()
				self.print_fn(f"[Birdie] Initialized old_loss_vector at step {current_training_step} with losses: {self.old_loss_vector}")
			else:
				# Ensure old_loss_vector also matches the number of training objectives for the agent
				if len(self.old_loss_vector) != len(self.objectives):
					self.print_fn(f"[Birdie Warning] old_loss_vector length ({len(self.old_loss_vector)}) "
								  f"mismatches training objectives count ({len(self.objectives)}). Resetting old_loss_vector.")
					self.old_loss_vector = np.zeros(len(self.objectives), dtype=np.float32) # Or use current_validation_losses if lengths match now


				new_action_probs = self.update_reward_model(
					action_taken=self.last_action, # last_action should also match len(self.objectives)
					old_loss_vector=self.old_loss_vector,
					new_loss_vector=self.current_validation_losses,
					old_step_idx=self.last_reward_step_idx, 
					new_step_idx=current_training_step,   
				)
				self.last_action = new_action_probs 
				self.old_loss_vector = self.current_validation_losses.copy() 

				for i, obj_setting in enumerate(self.objectives):
					obj_setting["prob"] = float(np.round(new_action_probs[i], 4)) 
					obj_setting["loss"] = float(np.round(self.current_validation_losses[i], 4)) if i < len(self.current_validation_losses) else 0.0

				if hasattr(self, 'controller') and self.controller:
					self.controller.update(self.objectives, clear_prefetched=False) 
					self.print_fn(f"[Birdie] Updated controller with new objective probabilities at step {current_training_step}.")
				else:
					self.print_fn("[Birdie] Controller not available to update objectives.")
			
			self.last_reward_step_idx = current_training_step 
			return True # Indicates update happened
		else:
			# This case is normal if not all objectives are fulfilled yet.
			# self.print_fn(f"[Birdie] Reward model NOT updated at step {current_training_step} (fulfilled keys: {self.validation_num_fulfilled_keys}/{len(self.validation_objectives)}).")
			pass
		return False


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

		# Use self.objectives (training objectives) for reporting current sampling probabilities
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

			# Use nickname as the primary key for the output dictionary
			name_key = serializable_config.get("nickname", 
						 serializable_config.get("hash_str", 
						 serializable_config.get("name", f"objective_{idx}")))
			
			ret_dict[name_key] = {
				"current_sampling_probability": float(current_action_probs[idx]) if idx < len(current_action_probs) else 0.0,
				"config": {k_cfg:v_cfg for k_cfg,v_cfg in serializable_config.items() if k_cfg not in ["name", "nickname", "hash_str", "prob", "prob_initial", "loss"]} 
			}
			if "prob" in serializable_config: ret_dict[name_key]["config"]["original_prob_config"] = serializable_config["prob"]
			if "prob_initial" in serializable_config: ret_dict[name_key]["config"]["normalized_initial_prob"] = serializable_config["prob_initial"]
			if "loss" in serializable_config: ret_dict[name_key]["last_recorded_val_loss"] = serializable_config["loss"]

		formatted_ret_dict_str = json.dumps(ret_dict, indent=4, sort_keys=True)
		return ret_dict, formatted_ret_dict_str

