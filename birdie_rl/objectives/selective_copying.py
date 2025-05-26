"""
SelectiveCopying Objective:
- Optimizes by pre-tokenizing static delimiter strings.
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

# Static tokens (string form for config, will be tokenized in __init__)
COPY_TOKEN_STR    = "[COPY]" # Not used in current logic directly as a token, but concept exists
START_TOKEN_STR   = "find"   # Used in f-string like f"[{START_TOKEN_STR} {idx}]"
RESULT_TOKEN_STR  = "result" # Used in f-string like f"[{RESULT_TOKEN_STR} {idx}]"
# END_TOKEN_STR     = "/find"  # Used in f-string like f"[/{START_TOKEN_STR} {idx}]" -> now uses START_TOKEN_STR
CONTEXT_TOKEN_STR = "\n\n[context]\n"
CLOSE_CONTEXT_STR = "\n[/context]"
SEP_TOKEN_STR     = "sep"    # Used in f-string like f"\n[{SEP_TOKEN_STR}]\n"
DONE_TOKEN_STR    = "\n\n[done]"


@dataclasses.dataclass
class SelectiveCopyingConfig(BaseObjectiveConfig):
	corruption_rate: float = 0.5 # Effectively determines probability of selecting a span
	tokens_per_mask: int = 8     # Average length of a span to be copied
	shuffle: bool = True         # Shuffle the order of "find" instructions and "result" blocks
	# separator: str = " "       # Not directly used as a pre-tokenized part
	paradigm_prompt: str = "<|Selective Copying|>"
	# gap_between_spans: int = 1 # Not directly used
	max_attempts: int = 100      # Attempts to insert at least one copy instruction
	paradigm_end: str = ""       # Optional suffix for the entire label sequence (e.g., EOS)
	
	min_delimiter_prefix_length: int = 8
	max_delimiter_prefix_length: int = 64
	min_delimiter_suffix_length: int = 8
	max_delimiter_suffix_length: int = 64
	format_style: str = "query_context" # "query_context" or "context_query"

	# String versions of tokens for config, to be tokenized in __init__
	start_token_template: str = START_TOKEN_STR 
	result_token_template: str = RESULT_TOKEN_STR
	# end_token_template: str = END_TOKEN_STR # Will use start_token_template for /find
	context_token: str = CONTEXT_TOKEN_STR
	close_context_token: str = CLOSE_CONTEXT_STR
	sep_token_template: str = SEP_TOKEN_STR
	done_token: str = DONE_TOKEN_STR


	def __post_init__(self):
		"""
		Adjusts some derived fields for min/max corruption rates
		and tokens_per_mask range.	
		"""
		self.minimum_corruption_rate = self.corruption_rate * 0.5
		self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
		self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
		self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


class SelectiveCopyingObjective(BaseObjective):
	"""
	SelectiveCopyingObjective. Pre-tokenizes static delimiter strings.
	"""

	def __init__(self, config: SelectiveCopyingConfig) -> None:
		super().__init__(config)
		# Pre-tokenize static parts
		if self.tokenizer:
			self.tokenized_paradigm_prompt = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_prompt)) if self.config.paradigm_prompt else []
			self.tokenized_paradigm_end = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_end)) if self.config.paradigm_end else []
			
			self.tokenized_context_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.context_token))
			self.tokenized_close_context_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.close_context_token))
			self.tokenized_done_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.done_token))
			# Templates will be used in f-strings, so their components are tokenized as needed
			# e.g., self.tokenizer.encode(f"\n\n[{self.config.start_token_template} {idx}]\n")
		else:
			raise ValueError("Tokenizer must be provided to SelectiveCopyingObjective via config.")


	def build_input_and_labels(
		self, input_text: str, config: SelectiveCopyingConfig
	) -> Dict[str, Any]:
		tokenizer = self.tokenizer
		encoded_input = tokenizer.encode(input_text)
		n_total_original_tokens = len(encoded_input)

		# Reserve space for prompt, context markers, done token, and some buffer
		# This is a rough estimate; precise checks are done in the loop.
		overhead_estimate = len(self.tokenized_paradigm_prompt) + \
							len(self.tokenized_context_token) + \
							len(self.tokenized_close_context_token) + \
							len(self.tokenized_done_token) + 64 # Buffer for instruction tokens
		max_n_tokens_for_content = config.remaining_space - overhead_estimate
		
		if max_n_tokens_for_content <= config.min_delimiter_prefix_length + config.min_delimiter_suffix_length + config.minimum_tokens_per_mask:
			return { # Not enough space for even one meaningful operation
				"status": "fail", "objective": "SelectiveCopying", "input_ids": [], "label_ids": [],
				"unused_input_string": input_text, "unused_input_ids": encoded_input,
				"masked_count": 0, "original_length": 0,
			}
		
		# Limit processing to a segment of the input text that can plausibly fit
		# This `text_segment_to_process_len` is the length of the original text we'll draw spans from
		text_segment_to_process_len = min(n_total_original_tokens, max_n_tokens_for_content)


		np_rng = self.np_rng # Use the objective's RNG

		def sample_span_length(current_text_idx: int, max_available_text_len: int) -> int:
			raw_len = np_rng.poisson(config.tokens_per_mask) # Use config from instance
			raw_len = max(raw_len, config.minimum_tokens_per_mask)
			raw_len = min(raw_len, config.maximum_tokens_per_mask)
			limit = max_available_text_len - current_text_idx
			return max(1, min(raw_len, limit))

		for attempt_i in range(config.max_attempts):
			unshuffled_instructions_data = [] # Stores tuples of (original_placeholder_idx, prefix_toks, snippet_toks, suffix_toks)
			label_blocks_in_original_order = [] # Stores token lists for [result_toks, snippet_toks]
			
			placeholders_generated_count = 0
			total_copied_token_count = 0
			
			# Current index within the `encoded_input` array (up to `text_segment_to_process_len`)
			current_text_idx = 0 
			
			# Ensure first prefix can be drawn
			min_prefix_len = config.min_delimiter_prefix_length
			current_text_idx = min_prefix_len 

			while current_text_idx < text_segment_to_process_len:
				# Check if we can select a span (prefix + snippet + suffix)
				# Max length of a span we can select from remaining text
				max_possible_span_here = text_segment_to_process_len - current_text_idx 
				min_suffix_len = config.min_delimiter_suffix_length
				
				if max_possible_span_here < (config.minimum_tokens_per_mask + min_suffix_len):
					break # Not enough text left for a snippet and its suffix delimiter

				span_len_of_snippet = sample_span_length(current_text_idx, text_segment_to_process_len - min_suffix_len)

				# Decide if we select this span
				local_corruption_rate = np_rng.uniform(config.minimum_corruption_rate, config.maximum_corruption_rate)
				prob_to_select_span = local_corruption_rate # Higher rate means more likely to select
				
				if np_rng.uniform() < prob_to_select_span:
					# Determine actual delimiter lengths for this span
					actual_prefix_len = np_rng.integers(config.min_delimiter_prefix_length, min(current_text_idx, config.max_delimiter_prefix_length) + 1)
					actual_suffix_len = np_rng.integers(config.min_delimiter_suffix_length, min(text_segment_to_process_len - (current_text_idx + span_len_of_snippet), config.max_delimiter_suffix_length) + 1)

					prefix_delimiter_toks = encoded_input[current_text_idx - actual_prefix_len : current_text_idx]
					snippet_toks = encoded_input[current_text_idx : current_text_idx + span_len_of_snippet]
					suffix_delimiter_toks = encoded_input[current_text_idx + span_len_of_snippet : current_text_idx + span_len_of_snippet + actual_suffix_len]

					# Construct the "find" and "result" token sequences
					# These are tokenized on-the-fly because the placeholder index changes
					find_instr_start_toks = tokenizer.encode(f"\n\n[{config.start_token_template} {placeholders_generated_count}]\n")
					find_instr_sep_toks = tokenizer.encode(f"\n[{config.sep_token_template}]\n")
					find_instr_end_toks = tokenizer.encode(f"\n[/{config.start_token_template} {placeholders_generated_count}]")
					
					result_instr_start_toks = tokenizer.encode(f"\n\n[{config.result_token_template} {placeholders_generated_count}]\n" if placeholders_generated_count > 0 else f"[{config.result_token_template} {placeholders_generated_count}]\n")

					# Store data for this instruction/result pair
					unshuffled_instructions_data.append(
						(placeholders_generated_count, list(prefix_delimiter_toks), list(snippet_toks), list(suffix_delimiter_toks), 
						 list(find_instr_start_toks), list(find_instr_sep_toks), list(find_instr_end_toks))
					)
					label_blocks_in_original_order.append(list(result_instr_start_toks) + list(snippet_toks))
					
					total_copied_token_count += len(snippet_toks)
					placeholders_generated_count += 1
					current_text_idx += span_len_of_snippet + actual_suffix_len # Move past snippet and its suffix
					current_text_idx += np_rng.integers(config.min_delimiter_prefix_length // 2, config.min_delimiter_prefix_length) # Add a small gap before next potential prefix
				else:
					# If not selecting, advance by a small random amount to find a new potential start
					current_text_idx += np_rng.integers(1, config.minimum_tokens_per_mask + 1)
			
			# After trying to generate instructions from the text segment
			if placeholders_generated_count > 0:
				# Assemble the final input_ids and label_ids
				final_input_ids_list = list(self.tokenized_paradigm_prompt)
				final_label_ids_list = []

				# The context is the original text up to the point we processed
				# This ensures delimiters are part of the context if they were not part of a selected span's delimiter.
				context_content_toks = encoded_input[:text_segment_to_process_len] # Use the segment we actually processed

				# Determine order of instructions and context based on format_style
				instruction_blocks_tokenized = []
				
				# Shuffle instructions if needed (shuffles the (original_idx, prefix, snippet, suffix, find_start, find_sep, find_end) tuples)
				indices_to_iterate = np_rng.permutation(len(unshuffled_instructions_data)) if config.shuffle else range(len(unshuffled_instructions_data))

				for i, original_data_idx in enumerate(indices_to_iterate):
					_orig_pl_idx, prefix_toks, _snippet_toks, suffix_toks, find_start, find_sep, find_end = unshuffled_instructions_data[original_data_idx]
					
					# Instruction uses the current shuffled index `i` for its placeholder number
					current_find_instr_start = tokenizer.encode(f"\n\n[{config.start_token_template} {i}]\n")
					current_find_instr_sep = tokenizer.encode(f"\n[{config.sep_token_template}]\n")
					current_find_instr_end = tokenizer.encode(f"\n[/{config.start_token_template} {i}]")

					instruction_blocks_tokenized.extend(current_find_instr_start)
					instruction_blocks_tokenized.extend(prefix_toks)
					instruction_blocks_tokenized.extend(current_find_instr_sep)
					instruction_blocks_tokenized.extend(suffix_toks)
					instruction_blocks_tokenized.extend(current_find_instr_end)

				# Assemble label_ids based on the original order of selected snippets, but with new placeholder indices if shuffled
				if config.shuffle:
					for i, original_data_idx in enumerate(indices_to_iterate):
						_orig_pl_idx, _prefix_toks, snippet_toks, _suffix_toks, _fs, _fsep, _fe = unshuffled_instructions_data[original_data_idx]
						current_result_instr_start = tokenizer.encode(f"\n\n[{config.result_token_template} {i}]\n" if i > 0 or final_label_ids_list else f"[{config.result_token_template} {i}]\n")
						final_label_ids_list.extend(current_result_instr_start)
						final_label_ids_list.extend(snippet_toks)
				else: # No shuffle, use label_blocks_in_original_order
					for block in label_blocks_in_original_order:
						final_label_ids_list.extend(block)
				
				final_label_ids_list.extend(self.tokenized_done_token)


				# Assemble final input based on format style
				context_block_full = list(self.tokenized_context_token) + list(context_content_toks) + list(self.tokenized_close_context_token)
				if config.format_style == "query_context":
					final_input_ids_list.extend(instruction_blocks_tokenized)
					final_input_ids_list.extend(context_block_full)
				else: # context_query
					final_input_ids_list.extend(context_block_full)
					final_input_ids_list.extend(instruction_blocks_tokenized)
				
				# Check total length against remaining_space
				if len(final_input_ids_list) + len(final_label_ids_list) <= config.remaining_space:
					unused_input_ids_list = encoded_input[text_segment_to_process_len:]
					unused_input_str = tokenizer.decode(unused_input_ids_list)
					return {
						"status": "ok", "objective": "SelectiveCopying",
						"input_ids": np.array(final_input_ids_list, dtype=np.int32),
						"label_ids": np.array(final_label_ids_list, dtype=np.int32),
						"unused_input_string": unused_input_str,
						"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
						"masked_count": total_copied_token_count, # "masked" here means "selected for copying"
						"original_length": text_segment_to_process_len,
					}
				# If it doesn't fit, this attempt failed, loop to next attempt or fallback.
		
		# Fallback: If no placeholders inserted after all attempts, or if generated content was too long
		# Return a simplified version, e.g., just the prompt and context (or handle as error)
		# For now, let's indicate failure to generate a proper sample for this objective.
		# This could be changed to a pass-through of original text if desired.
		self.print(f"SelectiveCopying: Failed to insert any valid copy instructions after {config.max_attempts} attempts or due to length constraints.", verbosity_level=1)
		return {
			"status": "fail", "objective": "SelectiveCopying (generation failed)",
			"input_ids": np.array(self.tokenized_paradigm_prompt, dtype=np.int32), # Minimal input
			"label_ids": np.array([], dtype=np.int32),
			"unused_input_string": input_text, # All original text is unused in this failure case
			"unused_input_ids": np.array(encoded_input, dtype=np.int32),
			"masked_count": 0, "original_length": 0,
		}


# Example Test (can be run standalone if Tokenizer is available)
if __name__ == "__main__":
	try:
		from birdie_rl.modeling.tokenizer import Tokenizer 
	except ImportError:
		class Tokenizer: # Mock tokenizer
			def encode(self, t): return [ord(c) for c in t] if isinstance(t, str) else [[ord(c) for c in s] for s in t]
			def decode(self, ids): 
				if not ids: return ""
				if isinstance(ids[0], list): return ["".join([chr(i) for i in id_list if i >=0]) for id_list in ids]
				return "".join([chr(i) for i in ids if i >=0])

	tok = Tokenizer()
	cfg = SelectiveCopyingConfig(
		remaining_space=200, # Ample space for a couple of operations
		corruption_rate=0.8, # High probability to select spans
		tokens_per_mask=5,   # Avg snippet length
		min_delimiter_prefix_length=2, max_delimiter_prefix_length=5,
		min_delimiter_suffix_length=2, max_delimiter_suffix_length=5,
		shuffle=True,
		format_style="query_context", # or "context_query"
		tokenizer=tok # Pass tokenizer in config
	)
	obj = SelectiveCopyingObjective(cfg)
	
	test_text = "The quick brown fox jumps over the lazy dog. " * 3 + "A final sentence for context."
	result = obj(test_text)

	print("\n--- SelectiveCopying Test Output ---")
	print("Status:", result["status"])
	if result["status"] == "ok":
		print("Input IDs (len {}):".format(len(result["input_ids"])), result["input_ids"][:20], "...")
		print("Decoded Input:", tok.decode(result["input_ids"].tolist()))
		print("-" * 20)
		print("Label IDs (len {}):".format(len(result["label_ids"])), result["label_ids"][:20], "...")
		print("Decoded Label:", tok.decode(result["label_ids"].tolist()))
		print("-" * 20)
		print("Copied (Masked) Token Count:", result["masked_count"])
		print("Original Text Segment Length Used:", result["original_length"])
		print("Unused Input String:", result["unused_input_string"][:100] + "..." if result["unused_input_string"] else "None")
	else:
		print("Message:", result.get("message", "N/A"))
