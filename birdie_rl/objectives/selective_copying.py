"""
SelectiveCopying Objective:
- Optimizes by pre-tokenizing static delimiter strings.
- Dynamically adjusts context content length to fit remaining_space.
- Proactively checks token budget during instruction generation.
- Includes extensive debug printing.
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

# Static tokens (string form for config, will be tokenized in __init__)
COPY_TOKEN_STR    = "[COPY]" 
START_TOKEN_STR   = "find"   
RESULT_TOKEN_STR  = "res" 
CONTEXT_TOKEN_STR = "\n\n[ctx]\n"
CLOSE_CONTEXT_STR = "\n[/ctx]"
SEP_TOKEN_STR     = "sep"    
DONE_TOKEN_STR    = "\n\n[done]"


@dataclasses.dataclass
class SelectiveCopyingConfig(BaseObjectiveConfig):
	corruption_rate: float = 0.5 
	tokens_per_mask: int = 8     
	shuffle: bool = True         
	paradigm_prompt: str = "<|Selective Copying|>"
	max_attempts: int = 100      
	paradigm_end: str = ""       # Usually for labels, if needed (e.g. EOS)
	
	min_delimiter_prefix_length: int = 8
	max_delimiter_prefix_length: int = 64
	min_delimiter_suffix_length: int = 8
	max_delimiter_suffix_length: int = 64
	format_style: str = "query_context" 

	start_token_template: str = START_TOKEN_STR 
	result_token_template: str = RESULT_TOKEN_STR
	context_token: str = CONTEXT_TOKEN_STR
	close_context_token: str = CLOSE_CONTEXT_STR
	sep_token_template: str = SEP_TOKEN_STR
	done_token: str = DONE_TOKEN_STR

	min_context_content_length: int = 10 
	objective_verbosity: int = 0 # 0: none, 1: basic, 2: detailed, 3: extreme


	def __post_init__(self):
		self.minimum_corruption_rate = self.corruption_rate * 0.5
		self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
		self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
		self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


class SelectiveCopyingObjective(BaseObjective):
	"""
	SelectiveCopyingObjective. Pre-tokenizes static delimiter strings.
	Dynamically adjusts context content length and proactively checks budget.
	"""

	def __init__(self, config: SelectiveCopyingConfig) -> None:
		super().__init__(config)
		self.objective_name = "SelectiveCopying"
		if self.tokenizer:
			self.tokenized_paradigm_prompt = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_prompt)) if self.config.paradigm_prompt else []
			self.tokenized_paradigm_end = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_end)) if self.config.paradigm_end else []
			self.tokenized_context_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.context_token))
			self.tokenized_close_context_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.close_context_token))
			self.tokenized_done_token = self.safe_cast_to_list(self.tokenizer.encode(self.config.done_token))
		else:
			raise ValueError("Tokenizer must be provided to SelectiveCopyingObjective via config.")

	def _debug_print(self, message: str, level: int = 1):
		if hasattr(self.config, 'objective_verbosity') and self.config.objective_verbosity >= level:
			print(f"[{self.objective_name} DBG L{level}] {message}", flush=True)

	def build_input_and_labels(
		self, input_text: str, config: SelectiveCopyingConfig
	) -> Dict[str, Any]:
		self._debug_print(f"Starting build_input_and_labels. input_text len: {len(input_text)}, remaining_space: {config.remaining_space}", level=1)
		tokenizer = self.tokenizer
		encoded_input = tokenizer.encode(input_text)
		n_total_original_tokens = len(encoded_input)
		np_rng = self.np_rng
		self._debug_print(f"Encoded input len: {n_total_original_tokens}", level=2)

		initial_instr_result_buffer = 31 + 7 # Rough estimate for one operation's markers
		overhead_estimate = (
			len(self.tokenized_paradigm_prompt)
			+ len(self.tokenized_context_token)
			+ len(self.tokenized_close_context_token)
			+ len(self.tokenized_done_token) 
			+ initial_instr_result_buffer 
		)
		max_n_tokens_for_initial_segment = config.remaining_space - overhead_estimate
		self._debug_print(f"Initial overhead_estimate: {overhead_estimate}, max_n_tokens_for_initial_segment: {max_n_tokens_for_initial_segment}", level=2)
		
		min_segment_for_one_op = (
			config.min_delimiter_prefix_length 
			+ config.minimum_tokens_per_mask 
			+ config.min_delimiter_suffix_length
		)
		self._debug_print(f"min_segment_for_one_op: {min_segment_for_one_op}", level=2)

		if max_n_tokens_for_initial_segment < min_segment_for_one_op:
			self._debug_print("Failing: Not enough space for even one minimal operation after fixed overheads.", level=1)
			return {
				"status": "fail", "objective": self.objective_name, 
				"message": "Not enough space for even one minimal operation after fixed overheads.",
				"input_ids": [], "label_ids": [],
				"unused_input_string": input_text, "unused_input_ids": np.array(encoded_input, dtype=np.int32),
				"masked_count": 0, "original_length": 0,
			}
		
		text_segment_to_process_len = min(n_total_original_tokens, max_n_tokens_for_initial_segment)
		self._debug_print(f"text_segment_to_process_len: {text_segment_to_process_len}", level=2)
		if text_segment_to_process_len < min_segment_for_one_op:
			self._debug_print("Failing: Original text too short for one minimal operation within the calculated segment.", level=1)
			return {
				"status": "fail", "objective": self.objective_name, 
				"message": "Original text too short for one minimal operation within this segment.", # Clarified message
				"input_ids": [], "label_ids": [],
				"unused_input_string": input_text, "unused_input_ids": np.array(encoded_input, dtype=np.int32),
				"masked_count": 0, "original_length": 0,
			}

		def sample_span_length(current_text_idx: int, max_endpoint_for_snippet_start: int) -> int:
			max_len_here = max_endpoint_for_snippet_start - current_text_idx
			if max_len_here < config.minimum_tokens_per_mask:
				return 0 

			raw_len = np_rng.poisson(config.tokens_per_mask) 
			raw_len = max(raw_len, config.minimum_tokens_per_mask)
			raw_len = min(raw_len, config.maximum_tokens_per_mask)
			return max(config.minimum_tokens_per_mask, min(raw_len, max_len_here))


		for attempt_i in range(config.max_attempts):
			self._debug_print(f"Attempt {attempt_i + 1}/{config.max_attempts}", level=1)
			unshuffled_instructions_data = []
			label_blocks_in_original_order = []
			placeholders_generated_count = 0
			total_copied_token_count = 0
			current_text_idx = config.min_delimiter_prefix_length 

			accumulated_instruction_tokens_len = 0
			accumulated_label_content_tokens_len = 0

			fixed_overhead_for_input_assembly = len(self.tokenized_paradigm_prompt) + \
												len(self.tokenized_context_token) + \
												len(self.tokenized_close_context_token)
			fixed_overhead_for_label_assembly = len(self.tokenized_done_token) + \
												len(self.tokenized_paradigm_end)
			
			min_total_space_for_fixed_and_context = \
				fixed_overhead_for_input_assembly + \
				fixed_overhead_for_label_assembly + \
				config.min_context_content_length
			
			max_idx_for_snippet_start = text_segment_to_process_len - config.min_delimiter_suffix_length - config.minimum_tokens_per_mask
			self._debug_print(f"  Attempt {attempt_i+1}: max_idx_for_snippet_start: {max_idx_for_snippet_start}, current_text_idx initial: {current_text_idx}", level=3)
			
			loop_iteration_count = 0
			while current_text_idx <= max_idx_for_snippet_start :
				loop_iteration_count += 1
				self._debug_print(f"    Inner loop iter {loop_iteration_count}: current_text_idx: {current_text_idx}", level=3)
				span_len_of_snippet = sample_span_length(current_text_idx, text_segment_to_process_len - config.min_delimiter_suffix_length)
				self._debug_print(f"      span_len_of_snippet: {span_len_of_snippet}", level=3)
				if span_len_of_snippet == 0: 
					current_text_idx += 1 
					self._debug_print(f"      Snippet len 0, advancing current_text_idx to {current_text_idx}", level=3)
					continue

				local_corruption_rate = np_rng.uniform(config.minimum_corruption_rate, config.maximum_corruption_rate)
				rand_val_for_corruption = np_rng.uniform()
				self._debug_print(f"      local_corruption_rate: {local_corruption_rate:.3f}, rand_val: {rand_val_for_corruption:.3f}", level=3)

				if rand_val_for_corruption < local_corruption_rate:
					self._debug_print("      Corruption check PASSED. Attempting to select span.", level=3)
					actual_prefix_len = np_rng.integers(config.min_delimiter_prefix_length, min(current_text_idx, config.max_delimiter_prefix_length) + 1)
					space_after_snippet = text_segment_to_process_len - (current_text_idx + span_len_of_snippet)
					
					if space_after_snippet < config.min_delimiter_suffix_length:
						self._debug_print(f"      Not enough space for suffix ({space_after_snippet} < {config.min_delimiter_suffix_length}). Advancing current_text_idx.", level=3)
						current_text_idx += 1 
						continue
					actual_suffix_len = np_rng.integers(config.min_delimiter_suffix_length, min(space_after_snippet, config.max_delimiter_suffix_length) + 1)
					self._debug_print(f"      actual_prefix_len: {actual_prefix_len}, actual_suffix_len: {actual_suffix_len}", level=3)

					prefix_delimiter_toks = encoded_input[current_text_idx - actual_prefix_len : current_text_idx]
					snippet_toks = encoded_input[current_text_idx : current_text_idx + span_len_of_snippet]
					suffix_delimiter_toks = encoded_input[current_text_idx + span_len_of_snippet : current_text_idx + span_len_of_snippet + actual_suffix_len]

					temp_find_instr_start_toks = tokenizer.encode(f"\n\n[{config.start_token_template} {placeholders_generated_count}]\n")
					temp_find_instr_sep_toks = tokenizer.encode(f"\n[{config.sep_token_template}]\n")
					temp_find_instr_end_toks = tokenizer.encode(f"\n[/{config.start_token_template} {placeholders_generated_count}]")
					temp_result_instr_start_toks = tokenizer.encode(f"\n\n[{config.result_token_template} {placeholders_generated_count}]\n" if placeholders_generated_count > 0 or label_blocks_in_original_order else f"[{config.result_token_template} {placeholders_generated_count}]\n")
					
					current_instr_block_len_candidate = len(temp_find_instr_start_toks) + len(prefix_delimiter_toks) + \
														len(temp_find_instr_sep_toks) + len(suffix_delimiter_toks) + \
														len(temp_find_instr_end_toks)
					current_label_block_len_candidate = len(temp_result_instr_start_toks) + len(snippet_toks)
					self._debug_print(f"      Candidate instr_len: {current_instr_block_len_candidate}, label_len: {current_label_block_len_candidate}", level=3)

					prospective_total_instr_len = accumulated_instruction_tokens_len + current_instr_block_len_candidate
					prospective_total_label_len = accumulated_label_content_tokens_len + current_label_block_len_candidate
					
					estimated_total_len_if_added = min_total_space_for_fixed_and_context + \
												   prospective_total_instr_len + \
												   prospective_total_label_len
					self._debug_print(f"      Prospective total_instr: {prospective_total_instr_len}, total_label: {prospective_total_label_len}", level=3)
					self._debug_print(f"      min_total_space_for_fixed_and_context: {min_total_space_for_fixed_and_context}", level=3)
					self._debug_print(f"      estimated_total_len_if_added: {estimated_total_len_if_added} vs remaining_space: {config.remaining_space}", level=3)
													   
					if estimated_total_len_if_added > config.remaining_space:
						self._debug_print("      BREAKING inner loop: estimated_total_len_if_added > config.remaining_space", level=2)
						break 

					unshuffled_instructions_data.append(
						(placeholders_generated_count, list(prefix_delimiter_toks), list(snippet_toks), list(suffix_delimiter_toks), 
						 list(temp_find_instr_start_toks), list(temp_find_instr_sep_toks), list(temp_find_instr_end_toks))
					)
					label_blocks_in_original_order.append(list(temp_result_instr_start_toks) + list(snippet_toks))
					
					accumulated_instruction_tokens_len = prospective_total_instr_len
					accumulated_label_content_tokens_len = prospective_total_label_len
					
					total_copied_token_count += len(snippet_toks)
					placeholders_generated_count += 1
					self._debug_print(f"      Added placeholder {placeholders_generated_count}. Copied tokens: {len(snippet_toks)}. Total copied: {total_copied_token_count}", level=2)
					current_text_idx += span_len_of_snippet + actual_suffix_len 
					
					max_advance = max(1, config.min_delimiter_prefix_length // 2)
					current_text_idx += np_rng.integers(1, max_advance + 1) # Fixed ValueError here
					self._debug_print(f"      Advanced current_text_idx to {current_text_idx} after successful add.", level=3)
				else:
					self._debug_print("      Corruption check FAILED. Advancing current_text_idx.", level=3)
					current_text_idx += np_rng.integers(1, config.minimum_tokens_per_mask + 1)
			
			self._debug_print(f"  Attempt {attempt_i+1}: Inner loop finished. Placeholders generated: {placeholders_generated_count}", level=2)

			if placeholders_generated_count > 0:
				self._debug_print(f"  Attempt {attempt_i+1}: Processing {placeholders_generated_count} generated placeholders.", level=1)
				instruction_blocks_tokenized = []
				final_label_content_list = []   
				
				indices_to_iterate = np_rng.permutation(len(unshuffled_instructions_data)) if config.shuffle else range(len(unshuffled_instructions_data))
				self._debug_print(f"    Shuffle: {config.shuffle}. Indices to iterate: {list(indices_to_iterate)}", level=3)

				for i, original_data_idx in enumerate(indices_to_iterate):
					_orig_pl_idx, prefix_toks, snippet_toks_for_label, suffix_toks, _find_s, _find_sep, _find_e = unshuffled_instructions_data[original_data_idx]
					
					current_find_instr_start = tokenizer.encode(f"\n\n[{config.start_token_template} {i}]\n")
					current_find_instr_sep = tokenizer.encode(f"\n[{config.sep_token_template}]\n")
					current_find_instr_end = tokenizer.encode(f"\n[/{config.start_token_template} {i}]")
					instruction_blocks_tokenized.extend(current_find_instr_start + prefix_toks + current_find_instr_sep + suffix_toks + current_find_instr_end)

					current_result_instr_start = tokenizer.encode(f"\n\n[{config.result_token_template} {i}]\n" if i > 0 or final_label_content_list else f"[{config.result_token_template} {i}]\n")
					final_label_content_list.extend(current_result_instr_start + snippet_toks_for_label)
				
				self._debug_print(f"    Total instruction_blocks_tokenized len: {len(instruction_blocks_tokenized)}", level=3)
				self._debug_print(f"    Total final_label_content_list len: {len(final_label_content_list)}", level=3)

				space_for_paradigm_prompt = len(self.tokenized_paradigm_prompt)
				space_for_all_instructions = len(instruction_blocks_tokenized) 
				space_for_context_markers = len(self.tokenized_context_token) + len(self.tokenized_close_context_token)
				space_for_label_content = len(final_label_content_list) 
				space_for_label_fixed_end = len(self.tokenized_done_token) + len(self.tokenized_paradigm_end)

				input_non_context_len = space_for_paradigm_prompt + space_for_all_instructions + space_for_context_markers
				label_total_len = space_for_label_content + space_for_label_fixed_end
				available_for_actual_context_text = config.remaining_space - (input_non_context_len + label_total_len)
				self._debug_print(f"    space_for_paradigm_prompt: {space_for_paradigm_prompt}", level=3)
				self._debug_print(f"    space_for_all_instructions: {space_for_all_instructions}", level=3)
				self._debug_print(f"    space_for_context_markers: {space_for_context_markers}", level=3)
				self._debug_print(f"    space_for_label_content: {space_for_label_content}", level=3)
				self._debug_print(f"    space_for_label_fixed_end: {space_for_label_fixed_end}", level=3)
				self._debug_print(f"    input_non_context_len: {input_non_context_len}, label_total_len: {label_total_len}", level=3)
				self._debug_print(f"    available_for_actual_context_text: {available_for_actual_context_text} (min_context_content_length: {config.min_context_content_length})", level=2)


				if available_for_actual_context_text >= config.min_context_content_length:
					actual_context_content_len = min(text_segment_to_process_len, available_for_actual_context_text)
					self._debug_print(f"    actual_context_content_len: {actual_context_content_len}", level=2)
					
					if actual_context_content_len < config.min_context_content_length: # Check again after min with text_segment_to_process_len
						self._debug_print(f"    Context content too short ({actual_context_content_len} < {config.min_context_content_length}) after min with text_segment_to_process_len. Continuing to next attempt.", level=2)
						continue 

					context_content_toks = encoded_input[:actual_context_content_len]
					final_input_ids_list = list(self.tokenized_paradigm_prompt)
					context_block_full = list(self.tokenized_context_token) + list(context_content_toks) + list(self.tokenized_close_context_token)
					
					if config.format_style == "query_context":
						final_input_ids_list.extend(instruction_blocks_tokenized)
						final_input_ids_list.extend(context_block_full)
					else: 
						final_input_ids_list.extend(context_block_full)
						final_input_ids_list.extend(instruction_blocks_tokenized)
					
					final_label_ids_list = final_label_content_list + self.tokenized_done_token + self.tokenized_paradigm_end
					
					total_final_len = len(final_input_ids_list) + len(final_label_ids_list)
					self._debug_print(f"    Final assembly: input_len={len(final_input_ids_list)}, label_len={len(final_label_ids_list)}, total_final_len={total_final_len}, remaining_space={config.remaining_space}", level=2)

					if total_final_len <= config.remaining_space:
						unused_input_ids_list = encoded_input[actual_context_content_len:] 
						unused_input_str = tokenizer.decode(unused_input_ids_list)
						self._debug_print("    SUCCESS: Sample fits. Returning 'ok'.", level=1)
						return {
							"status": "ok", "objective": self.objective_name,
							"input_ids": np.array(final_input_ids_list, dtype=np.int32),
							"label_ids": np.array(final_label_ids_list, dtype=np.int32),
							"unused_input_string": unused_input_str,
							"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
							"masked_count": total_copied_token_count,
							"original_length": actual_context_content_len, 
						}
					else:
						self._debug_print(f"    Sample too long after final assembly ({total_final_len} > {config.remaining_space}). Continuing to next attempt.", level=2)
						continue 
				else: 
					self._debug_print(f"    Not enough space for min_context_content_length ({available_for_actual_context_text} < {config.min_context_content_length}). Continuing to next attempt.", level=2)
					continue 
		
		# Fallback if loop completes without returning
		# Using standard print for this important fallback message, ensuring it's always visible
		fallback_message = (
			f"[{self.objective_name}]\n"
			f"Failed to insert any valid copy instructions after {config.max_attempts} attempts.\n"
			f"Configured remaining_space: {config.remaining_space}, Input text original tokens: {n_total_original_tokens}.\n"
			f"Last text_segment_to_process_len: {text_segment_to_process_len}.\n"
			# placeholders_generated_count is from the last attempt, which might not be representative if it failed early
			# f"Last placeholders generated in an attempt (before final check): {placeholders_generated_count_in_last_successful_generation_loop if 'placeholders_generated_count_in_last_successful_generation_loop' in locals() else 'N/A'}.\n"
			f"Input text (first 100 chars of {len(input_text)}): '{input_text[:100]}...'"
		)
		print(fallback_message, flush=True)
		return {
			"status": "fail", "objective": f"{self.objective_name} (generation failed)",
			"message": "Failed to generate valid sample within token budget after all attempts.",
			"input_ids": np.array(self.tokenized_paradigm_prompt, dtype=np.int32),
			"label_ids": np.array([], dtype=np.int32),
			"unused_input_string": input_text, 
			"unused_input_ids": np.array(encoded_input, dtype=np.int32),
			"masked_count": 0, "original_length": 0,
		}

# Example Test (can be run standalone if Tokenizer is available)
if __name__ == "__main__":
	try:
		from birdie_rl.modeling.tokenizer import Tokenizer 
	except ImportError:
		class Tokenizer: 
			def encode(self, t): return [ord(c) for c in t] if isinstance(t, str) else [[ord(c) for c in s] for s in t]
			def decode(self, ids): 
				if not len(ids): return "" # Corrected as per user request
				if isinstance(ids, np.ndarray): ids = ids.tolist()
				if not len(ids): return "" # Check again after tolist
				is_batch = isinstance(ids[0], list) if ids and isinstance(ids, list) else False
				if is_batch:
					return ["".join([chr(i) for i in id_list if isinstance(i, int) and i >=0]) for id_list in ids]
				else: 
					return "".join([chr(i) for i in ids if isinstance(i, int) and i >=0])


	tok = Tokenizer()
	cfg_main_test = SelectiveCopyingConfig(
		remaining_space=200, 
		corruption_rate=0.8, 
		tokens_per_mask=5,   
		min_delimiter_prefix_length=2, max_delimiter_prefix_length=5,
		min_delimiter_suffix_length=2, max_delimiter_suffix_length=5,
		shuffle=True,
		format_style="query_context", 
		tokenizer=tok, 
		min_context_content_length=5,
		objective_verbosity=3 
	)
	obj_main_test = SelectiveCopyingObjective(cfg_main_test)
	
	test_text_main = "The quick brown fox jumps over the lazy dog. A second sentence for more context. And a third one to make sure it's long enough."
	print(f"Test text length (tokens): {len(tok.encode(test_text_main))}")
	result_main = obj_main_test(test_text_main)

	print("\n--- SelectiveCopying Test Output ---")
	print("Status:", result_main["status"])
	if result_main["status"] == "ok":
		print("Input IDs (len {}):".format(len(result_main["input_ids"])), result_main["input_ids"][:30], "...")
		print("Decoded Input:", tok.decode(result_main["input_ids"]))
		print("-" * 20)
		print("Label IDs (len {}):".format(len(result_main["label_ids"])), result_main["label_ids"][:30], "...")
		print("Decoded Label:", tok.decode(result_main["label_ids"]))
		print("-" * 20)
		print("Copied (Masked) Token Count:", result_main["masked_count"])
		print("Original Text Segment Length Used (for context):", result_main["original_length"])
		print("Unused Input String:", result_main["unused_input_string"][:100] + "..." if result_main["unused_input_string"] else "None")
	else:
		print("Message:", result_main.get("message", "N/A"))

	print("\n--- Test with very short text (EXPECTING BASE OBJECTIVE FAIL) ---")
	short_text = "Short example." # Length 14 with mock tokenizer
	cfg_short = SelectiveCopyingConfig(
		tokenizer=tok, 
		remaining_space=100, # Ample space
		# minimum_sequence_length is 32 by default from BaseObjectiveConfig
		min_context_content_length=5, 
		objective_verbosity=1 
	)
	obj_short = SelectiveCopyingObjective(cfg_short)
	result_short = obj_short(short_text) # This will call BaseObjective.run_checks first
	print("Status (short text):", result_short["status"])
	print("Message (short text):", result_short.get("message", "N/A"))


	print("\n--- Test with very short text (MODIFIED TO PASS BASE CHECK, EXPECTING OBJECTIVE FAIL OR PASS) ---")
	cfg_short_modified = SelectiveCopyingConfig(
		tokenizer=tok, 
		remaining_space=110, # Increased from 100 to 110
		minimum_sequence_length=10, # Override base config
		min_context_content_length=5, 
		min_delimiter_prefix_length=1, # Reduced
		min_delimiter_suffix_length=1, # Reduced
		tokens_per_mask=1,             # Reduced (so minimum_tokens_per_mask = 1)
		objective_verbosity=3 
	)
	obj_short_modified = SelectiveCopyingObjective(cfg_short_modified)
	result_short_modified = obj_short_modified(short_text) # Text "Short example." (len 14)
	print("Status (short text modified):", result_short_modified["status"])
	print("Message (short text modified):", result_short_modified.get("message", "N/A"))
	if result_short_modified["status"] == 'ok':
		print("Decoded Input (short text modified):", tok.decode(result_short_modified["input_ids"]))
		print("Decoded Label (short text modified):", tok.decode(result_short_modified["label_ids"]))


	print("\n--- Test with minimal remaining_space (EXPECTING OBJECTIVE FAIL) ---")
	cfg_tight = SelectiveCopyingConfig(
		remaining_space=60, 
		tokenizer=tok, 
		min_context_content_length=2, 
		tokens_per_mask=2, 
		min_delimiter_prefix_length=1, 
		min_delimiter_suffix_length=1, 
		objective_verbosity=1
	) 
	obj_tight = SelectiveCopyingObjective(cfg_tight)
	# Use a longer text that would normally be processable if space allowed
	result_tight = obj_tight("This is a slightly longer test text for the tight configuration to see if it can extract at least one small snippet.") 
	print("Status (tight space):", result_tight["status"])
	print("Message (tight space):", result_tight.get("message", "N/A"))

