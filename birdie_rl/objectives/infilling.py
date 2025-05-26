"""
Infilling Objective:
-------------------
- Places placeholders for certain spans in the input (decided stochastically).
- The label is built by concatenating [placeholder + masked tokens] for each span.
- We skip any span if adding it would exceed `remaining_space`.
- If no spans are inserted after `max_attempts`, we revert to returning unmasked text.
- Optimizes by pre-tokenizing static strings like paradigm, mask_prefix, and mask_suffix.
- Corrects TypeError in slicing by ensuring span_len_to_mask is an integer.
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

@dataclasses.dataclass
class InfillingConfig(BaseObjectiveConfig):
	corruption_rate: float = 0.15
	mean_tokens_per_span: float = 3.0 # Can be float
	max_mask_spans: int = 20 
	mask_prefix: str = " [[mask_" 
	mask_suffix: str = "]]"    
	shuffle: bool = False 
	separator: str = " " 
	paradigm: str = "<|INFILL|>" 
	gap_between_spans: int = 1 
	max_attempts: int = 100
	paradigm_end: str = "" 

	def __post_init__(self):
		assert(0.0 <= self.corruption_rate <= 1.0)
		self.minimum_corruption_rate = max(0.0, self.corruption_rate * 0.5)
		self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
		# Ensure these are floats if mean_tokens_per_span is float, or cast later
		self.minimum_mean_tokens_per_span = max(1.0, float(self.mean_tokens_per_span) / 3.0)
		self.maximum_mean_tokens_per_span = max(1.0, float(self.mean_tokens_per_span) * 3.0)


class InfillingObjective(BaseObjective):
	"""
	InfillingObjective. Pre-tokenizes static strings.
	Ensures span lengths are integers for slicing.
	"""

	def __init__(self, config: InfillingConfig) -> None:
		super().__init__(config)
		if not isinstance(config.mean_tokens_per_span, (int, float)) or config.mean_tokens_per_span <=0:
			raise ValueError(f"mean_tokens_per_span must be a positive number. Got: {config.mean_tokens_per_span}")
		if not isinstance(config.minimum_mean_tokens_per_span, (int, float)) or config.minimum_mean_tokens_per_span <=0:
			raise ValueError(f"minimum_mean_tokens_per_span must be a positive number. Got: {config.minimum_mean_tokens_per_span}")
		if not isinstance(config.maximum_mean_tokens_per_span, (int, float)) or config.maximum_mean_tokens_per_span <=0:
			raise ValueError(f"maximum_mean_tokens_per_span must be a positive number. Got: {config.maximum_mean_tokens_per_span}")


		self.tokenized_paradigm = []
		if self.config.paradigm and self.tokenizer:
			self.tokenized_paradigm = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm))
		
		self.tokenized_paradigm_end = []
		if self.config.paradigm_end and self.tokenizer:
			self.tokenized_paradigm_end = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_end))
		
	def build_input_and_labels(
		self, input_text: str, config: InfillingConfig
	) -> Dict[str, Any]:
		tokenizer = self.tokenizer
		encoded_input = tokenizer.encode(input_text)

		prompt_toks = self.tokenized_paradigm 

		n_tokens = len(encoded_input)
		max_n_tokens_to_process = min(n_tokens, config.remaining_space - len(prompt_toks) - 16) 

		if max_n_tokens_to_process <= 0:
			return {
				"status": "fail", "objective": "Infilling", "input_ids": [], "label_ids": [],
				"unused_input_string": input_text, "unused_input_ids": np.array(encoded_input, dtype=np.int32), # Ensure numpy array
				"masked_count": 0, "original_length": 0,
			}

		def sample_span_length(current_text_idx: int) -> int:
			# np.random.poisson can return float if lam is float. Ensure int output.
			raw_len = self.np_rng.poisson(float(config.mean_tokens_per_span)) 
			# Ensure raw_len is compared with float versions of min/max from config
			raw_len = max(raw_len, float(config.minimum_mean_tokens_per_span))
			raw_len = min(raw_len, float(config.maximum_mean_tokens_per_span))
			limit = max_n_tokens_to_process - current_text_idx
			# Final result must be an integer for slicing
			return int(max(1.0, min(raw_len, float(limit))))


		for attempt_i in range(config.max_attempts):
			current_input_ids_list = list(prompt_toks) 
			label_blocks_list_of_lists = [] 
			
			placeholders_inserted_count = 0
			masked_tokens_total_count = 0
			text_idx = 0 

			while text_idx < max_n_tokens_to_process and placeholders_inserted_count < config.max_mask_spans:
				current_input_len = len(current_input_ids_list)
				current_labels_len = sum(len(block) for block in label_blocks_list_of_lists) + len(self.tokenized_paradigm_end)

				if (current_input_len + current_labels_len) >= config.remaining_space:
					break 

				local_corruption_rate = self.np_rng.uniform(
					config.minimum_corruption_rate, config.maximum_corruption_rate
				)
				span_len_to_mask = sample_span_length(text_idx) # This now returns int
				
				prob_to_mask = local_corruption_rate / span_len_to_mask if span_len_to_mask > 0 else 0.0

				if self.np_rng.uniform() < prob_to_mask and (max_n_tokens_to_process - text_idx >= span_len_to_mask):
					snippet_to_mask = encoded_input[text_idx : text_idx + span_len_to_mask] # Slicing with integers
					if not snippet_to_mask: continue 

					ph_str = f"{config.mask_prefix}{placeholders_inserted_count}{config.mask_suffix}"
					ph_toks = tokenizer.encode(ph_str)

					prospective_input_len = current_input_len + len(ph_toks)
					prospective_label_block_len = len(ph_toks) + len(snippet_to_mask)
					prospective_total_labels_len = current_labels_len - len(self.tokenized_paradigm_end) + prospective_label_block_len + len(self.tokenized_paradigm_end) # Corrected labels_len logic
					
					if (prospective_input_len + prospective_total_labels_len) <= config.remaining_space:
						current_input_ids_list.extend(ph_toks)
						label_blocks_list_of_lists.append(list(ph_toks) + list(snippet_to_mask))
						
						text_idx += span_len_to_mask
						placeholders_inserted_count += 1
						masked_tokens_total_count += len(snippet_to_mask)
						continue
				
				if (current_input_len + 1 + current_labels_len) <= config.remaining_space:
					if text_idx < len(encoded_input): # Boundary check
						current_input_ids_list.append(encoded_input[text_idx])
						text_idx += 1
					else: # Should not happen if max_n_tokens_to_process is respected
						break
				else:
					break 

			if placeholders_inserted_count > 0:
				final_label_ids_list = []
				for block in label_blocks_list_of_lists:
					final_label_ids_list.extend(block)
				
				if self.tokenized_paradigm_end: 
					if len(current_input_ids_list) + len(final_label_ids_list) + len(self.tokenized_paradigm_end) <= config.remaining_space:
						final_label_ids_list.extend(self.tokenized_paradigm_end)
				
				unused_input_ids_list = encoded_input[text_idx:]
				unused_input_str = tokenizer.decode(unused_input_ids_list)

				return {
					"status": "ok", "objective": "Infilling",
					"input_ids": np.array(current_input_ids_list, dtype=np.int32),
					"label_ids": np.array(final_label_ids_list, dtype=np.int32),
					"unused_input_string": unused_input_str,
					"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
					"masked_count": masked_tokens_total_count,
					"original_length": text_idx, 
				}

		final_input_ids = list(prompt_toks)
		final_input_ids.extend(encoded_input[:max_n_tokens_to_process]) 
		
		final_label_ids_list = []
		if self.tokenized_paradigm_end:
			if len(final_input_ids) + len(self.tokenized_paradigm_end) <= config.remaining_space:
				final_label_ids_list.extend(self.tokenized_paradigm_end)

		unused_input_ids_list = encoded_input[max_n_tokens_to_process:]
		unused_input_str = tokenizer.decode(unused_input_ids_list)
		return {
			"status": "ok", 
			"objective": "Infilling (fallback, unmasked)",
			"input_ids": np.array(final_input_ids, dtype=np.int32),
			"label_ids": np.array(final_label_ids_list, dtype=np.int32),
			"unused_input_string": unused_input_str,
			"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
			"masked_count": 0,
			"original_length": max_n_tokens_to_process,
		}

if __name__ == "__main__":
	try:
		from birdie_rl.modeling.tokenizer import Tokenizer 
	except ImportError:
		class Tokenizer: 
			def encode(self, t): return [ord(c) for c in t] if isinstance(t, str) else [[ord(c) for c in s] for s in t]
			def decode(self, ids): 
				if not ids: return ""
				if isinstance(ids, np.ndarray): ids = ids.tolist() # Handle numpy array
				if not ids: return "" # Check again after tolist
				if isinstance(ids[0], list): return ["".join([chr(i) for i in id_list if i >=0]) for id_list in ids]
				return "".join([chr(i) for i in ids if i >=0])

	tok = Tokenizer()
	text = "Final test for infilling with minimal debug prints and pre-tokenization. This text is made longer to ensure spans can be selected." * 2
	cfg = InfillingConfig(
		remaining_space=100, 
		corruption_rate=0.25,
		mean_tokens_per_span=3.5, # Test with float
		max_attempts=5,
		mask_prefix=" <MASK_", 
		mask_suffix="> ",    
		paradigm="<|START_INFILL|> ",
		paradigm_end=" <|END_INFILL|>",
		tokenizer=tok 
	)
	obj = InfillingObjective(cfg)
	
	result = obj(text)
	print("\n--- Infilling Test Output ---")
	print("Status =", result["status"])
	print("Input IDs (len {}):".format(len(result["input_ids"])), result["input_ids"][:30], "...")
	print("Decoded input =", tok.decode(result["input_ids"]))
	print("-" * 20)
	print("Label IDs (len {}):".format(len(result["label_ids"])), result["label_ids"][:30], "...")
	print("Decoded label =", tok.decode(result["label_ids"]))
	print("-" * 20)
	print("Masked token count:", result["masked_count"])
	print("Original text segment length used:", result["original_length"])
	print("Unused text:", result["unused_input_string"][:100] + "..." if result["unused_input_string"] else "None")

	# Test with edge case of very small mean_tokens_per_span
	cfg_small_span = InfillingConfig(
		remaining_space=50, 
		corruption_rate=0.5,
		mean_tokens_per_span=1.0, # Smallest mean
		minimum_mean_tokens_per_span=1.0, # Ensure it can be 1
		maximum_mean_tokens_per_span=2.0,
		tokenizer=tok
	)
	obj_small_span = InfillingObjective(cfg_small_span)
	result_small = obj_small_span("Short example.")
	print("\n--- Infilling Test Output (Small Span) ---")
	print("Status =", result_small["status"])
	if result_small["status"] == "ok":
		print("Decoded input =", tok.decode(result_small["input_ids"]))
		print("Decoded label =", tok.decode(result_small["label_ids"]))
