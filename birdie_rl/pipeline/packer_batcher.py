import numpy as np
# import copy # No longer needed for deepcopy here
import heapq
import sys # For sys.stdout.flush() in example


def debug_alignments(current_dict, sub_idx=0):
	# This function seems to be for debugging and not directly used in the core logic.
	# Keeping it as is.
	for idx in range(len(current_dict["input_ids"][sub_idx])):
		print(
			f"idx: {idx}, "
			f"input_ids: {current_dict['input_ids'][sub_idx][idx]}, "
			f"label_ids: {current_dict['label_ids'][sub_idx][idx]}, "
			f"segment_ids: {current_dict['segment_ids'][sub_idx][idx]}, "
			f"attention_mask: {current_dict['attention_mask'][sub_idx][idx]}"
		)


class Packer:
	def __init__(self, tokenizer, sequence_length, minimum_sequence_length, start_generating_paradigm):
		if tokenizer is None:
			raise ValueError("Packer requires a tokenizer.")
		
		self.tokenizer = tokenizer 
		self.sequence_length = int(sequence_length)
		self.minimum_sequence_length = int(minimum_sequence_length) 
		self.start_generating_paradigm_str = str(start_generating_paradigm)
		
		self.tokenized_start_generating_paradigm = self.tokenizer.encode(self.start_generating_paradigm_str)
		
		if self.sequence_length < self.minimum_sequence_length:
			# This warning is fine, as minimum_sequence_length is a threshold for is_ready
			# print(f"Warning: Packer sequence_length ({self.sequence_length}) is less than minimum_sequence_length ({self.minimum_sequence_length}). This might be unintended if min_seq_len is meant as a hard lower bound for packer capacity.", flush=True)
			pass
		
		self.reset()


	def reset(self, sequence_length=None): 
		target_sequence_length = sequence_length or self.sequence_length
		self.sequence_length = target_sequence_length 

		self.current_dict = {
			"input_ids": np.zeros((target_sequence_length,), dtype=np.int32),
			"attention_mask": np.zeros((target_sequence_length,), dtype=np.int32),
			"label_ids": np.zeros((target_sequence_length,), dtype=np.int32) - 100,
			"segment_ids": np.zeros((target_sequence_length,), dtype=np.int32),
		}
		self.remaining_space = target_sequence_length 
		self.data_index = 0
		self.segment_counter = 0
		return self

	def get_remaining_space(self):
		"""Returns space available for objective's content, accounting for paradigm."""
		paradigm_len = len(self.tokenized_start_generating_paradigm)
		return max(0, self.remaining_space - paradigm_len)

	def is_ready(self):
		"""
		A Packer is "ready" if its true remaining_space is less than or equal to 
		the minimum_sequence_length threshold it's configured with (meaning it's too full
		to reliably add more diverse/larger objective outputs), or if it's completely full.
		"""
		minimal_meaningful_add = 1 + len(self.tokenized_start_generating_paradigm) 
		return self.remaining_space < minimal_meaningful_add or self.remaining_space <= self.minimum_sequence_length


	def _is_label_ids_empty(self, label_ids):
		"""Checks if label_ids is effectively empty (None, empty list, or empty numpy array)."""
		if label_ids is None:
			return True
		if isinstance(label_ids, np.ndarray):
			return label_ids.size == 0
		return not bool(label_ids) # For lists or other iterables

	def can_accept(self, input_ids, label_ids):
		paradigm_len = len(self.tokenized_start_generating_paradigm)
		if self._is_label_ids_empty(label_ids):
			total_length_to_add = len(input_ids) + paradigm_len 
		else: 
			total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1
		return total_length_to_add <= self.remaining_space

	def add(self, input_ids, label_ids, loss_mask=None): 
		paradigm_len = len(self.tokenized_start_generating_paradigm)
		
		label_ids_is_empty = self._is_label_ids_empty(label_ids)

		if label_ids_is_empty:
			# This was the original intent for the error, but Worker now filters this for most objectives
			# For "Infilling (fallback, unmasked)", empty labels are allowed.
			# Packer.add should be able to handle empty label_ids by not adding the teacher-forcing part.
			# raise ValueError("label_ids cannot be empty for Packer.add() unless it's a specific fallback case handled by the objective.")
			total_length_to_add = len(input_ids) + paradigm_len
		else: 
			total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1
		
		# Check if it can be accepted (this uses the corrected total_length_to_add)
		if total_length_to_add > self.remaining_space:
			raise ValueError(
				f"Insufficient space to add {total_length_to_add:,} tokens. "
				f"remaining_space: {self.remaining_space:,}"
			)
		
		input_ids_with_paradigm = np.concatenate([input_ids, self.tokenized_start_generating_paradigm])
		self.segment_counter += 1
		
		input_start = self.data_index
		input_prompt_end = input_start + len(input_ids_with_paradigm) 
		
		self.current_dict["input_ids"][input_start:input_prompt_end] = input_ids_with_paradigm
		self.current_dict["attention_mask"][input_start:input_prompt_end] = 1 
		self.current_dict["segment_ids"][input_start:input_prompt_end] = self.segment_counter

		# Only add label part if label_ids is not effectively empty
		if not label_ids_is_empty and len(label_ids) > 0: # Second check for len > 0 after ensuring it's not None
			label_target_start_offset_in_input = input_prompt_end - 1 
			# Ensure label_ids[:-1] is valid
			if len(label_ids) > 0: # This check is now redundant due to label_ids_is_empty and the outer if
				self.current_dict["input_ids"][input_prompt_end : input_prompt_end + len(label_ids) -1] = label_ids[:-1]
				self.current_dict["attention_mask"][input_prompt_end : input_prompt_end + len(label_ids) -1] = 1
				self.current_dict["label_ids"][label_target_start_offset_in_input : label_target_start_offset_in_input + len(label_ids)] = label_ids
				self.current_dict["segment_ids"][input_prompt_end : input_prompt_end + len(label_ids) -1] = self.segment_counter
		
		self.data_index += total_length_to_add
		self.remaining_space -= total_length_to_add

		return self.is_ready() 


class Batcher:
	def __init__(self, batch_size, tokenizer, sequence_length, minimum_sequence_length, start_generating_paradigm):
		self.batch_size = int(batch_size)
		
		if tokenizer is None:
			raise ValueError("Batcher requires a tokenizer.")
		self.tokenizer = tokenizer 
		
		self.sequence_length = int(sequence_length)
		self.minimum_sequence_length = int(minimum_sequence_length)
		self.start_generating_paradigm_str = str(start_generating_paradigm) 

		self.packers = []
		self.pq = [] 
		self.reset() 

	def reset(self, sequence_length=None, batch_size=None): 
		target_sequence_length = sequence_length or self.sequence_length
		target_batch_size = batch_size or self.batch_size
		
		self.sequence_length = target_sequence_length 
		self.batch_size = target_batch_size 

		self.packers = []
		self.pq = []
		heapq.heapify(self.pq) 

		for i in range(self.batch_size):
			packer = Packer( 
				tokenizer=self.tokenizer, 
				sequence_length=self.sequence_length, 
				minimum_sequence_length=self.minimum_sequence_length,
				start_generating_paradigm=self.start_generating_paradigm_str 
			)
			self.packers.append(packer)
			heapq.heappush(self.pq, (packer.get_remaining_space(), i, packer)) 
		return self

	def get_remaining_space(self, max_or_min="max"):
		if not self.packers: 
			return 0
		if self.batch_size == 1 and self.packers:
			return self.packers[0].get_remaining_space()

		if not self.pq: return 0 
		valid_packers = [entry[2] for entry in self.pq if entry[2] is not None]
		if not valid_packers: return 0
		if max_or_min == "max":
			return max(p.get_remaining_space() for p in valid_packers)
		else: 
			return self.pq[0][2].get_remaining_space() if self.pq and self.pq[0][2] is not None else 0


	def is_ready(self): 
		if not self.packers: return "empty"
		if self.batch_size == 1 and self.packers:
			return "ready" if self.packers[0].is_ready() else "not ready"
			
		valid_packers = [entry[2] for entry in self.pq if entry[2] is not None]
		if not valid_packers : return "empty" 
		return "ready" if all(p.is_ready() for p in valid_packers) else "not ready"

	def can_accept(self, input_ids, label_ids):
		if not self.packers: return False
		if self.batch_size == 1 and self.packers:
			return self.packers[0].can_accept(input_ids, label_ids)
			
		valid_packers = [entry[2] for entry in self.pq if entry[2] is not None]
		return any(p.can_accept(input_ids, label_ids) for p in valid_packers)

	def add(self, input_ids, label_ids, loss_mask=None, force_finish_pack=False):
		if self.batch_size == 1:
			if not self.packers: return "full" 
			packer_to_use = self.packers[0]
			if not packer_to_use.can_accept(input_ids, label_ids):
				return "full" 
			packer_became_ready = packer_to_use.add(input_ids, label_ids, loss_mask=loss_mask)
			return "ready" if packer_became_ready else "not ready"

		if not self.pq: return "full" 
		eligible_packers = []
		temp_buffer = []
		while self.pq:
			remaining_space, original_idx, packer_instance = heapq.heappop(self.pq)
			if packer_instance is None: continue 
			if packer_instance.can_accept(input_ids, label_ids):
				eligible_packers.append({'space': remaining_space, 'idx': original_idx, 'packer': packer_instance})
			else:
				temp_buffer.append((remaining_space, original_idx, packer_instance))
		
		for item in temp_buffer: heapq.heappush(self.pq, item)
		if not eligible_packers: return "full" 

		eligible_packers.sort(key=lambda x: x['space'], reverse=True)
		chosen_packer_info = eligible_packers[0]
		packer_to_use = chosen_packer_info['packer']
		original_idx_of_packer = chosen_packer_info['idx']
		
		packer_became_ready_after_add = packer_to_use.add(input_ids, label_ids, loss_mask=loss_mask)
		new_remaining_space_for_pq = 0 if force_finish_pack else packer_to_use.get_remaining_space() 
		self.packers[original_idx_of_packer] = packer_to_use 
		heapq.heappush(self.pq, (new_remaining_space_for_pq, original_idx_of_packer, packer_to_use))
		return self.is_ready()


	def pop(self, peek=False):
		if self.is_ready() != "ready" and not peek : 
			return None 
		if not self.packers: return None
		try:
			keys_to_stack = self.packers[0].current_dict.keys()
			stacked_dict = {
				key: np.stack([packer.current_dict[key] for packer in self.packers])
				for key in keys_to_stack
			}
		except IndexError: return None
		except AttributeError: return None 
		
		if not peek:
			self.reset() 

		return stacked_dict

	def get_sample_count(self): 
		return sum(packer.segment_counter for packer in self.packers if packer is not None)

