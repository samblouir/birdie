import numpy as np
import copy
import heapq
import random

def debug_alignments(current_dict, sub_idx=None, accelerator=None, tokenizer=None):
    """
    If sub_idx is None, prints the detailed alignment for all items in the batch.
    Otherwise, prints for only the sub_idx-th item.
    Uses tokenizer to decode input and label IDs if provided.
    Stops printing for a batch item if remaining tokens are all padding.
    """
    if accelerator is None:
        _print = print
    else:
        _print = accelerator.print
    
    if "input_ids" not in current_dict or not hasattr(current_dict["input_ids"], 'shape'):
        _print("Warning: 'input_ids' not found or not a NumPy array in current_dict for debug_alignments.")
        return
    if "segment_ids" not in current_dict or not hasattr(current_dict['segment_ids'], 'shape'):
        _print("Warning: 'segment_ids' not found or not a NumPy array in current_dict for debug_alignments.")
        return
    if "attention_mask" not in current_dict or not hasattr(current_dict['attention_mask'], 'shape'): 
        _print("Warning: 'attention_mask' not found or not a NumPy array in current_dict for debug_alignments.")
        return

    batch_size, sequence_length = current_dict["input_ids"].shape

    if sub_idx is None:
        indices = range(batch_size)
    else:
        indices = [sub_idx]

    idx_w = 3
    id_w = 5 
    tok_w = 18 
    seg_w = 3
    mask_w = 4
    
    header = (
        f"{'idx':<{idx_w}} | "
        f"{'inp_id':<{id_w}} | {'inp_tok':<{tok_w}} | "
        f"{'lbl_id':<{id_w}} | {'lbl_tok':<{tok_w}} | "
        f"{'seg':<{seg_w}} | {'msk':<{mask_w}}"
    )

    for b_idx in indices:
        _print(f"--- Detailed Alignment for sub_idx={b_idx} (Batch element {b_idx + 1}/{batch_size}) ---")
        _print(header)
        _print("-" * len(header))
        has_printed_content = False
        content_indices = np.where(current_dict['segment_ids'][b_idx] > 0)[0]
        last_content_idx = content_indices[-1] if len(content_indices) > 0 else -1

        for tok_idx in range(sequence_length):
            if not (0 <= b_idx < current_dict['segment_ids'].shape[0] and \
                    0 <= tok_idx < current_dict['segment_ids'].shape[1]):
                _print(f"Warning: Index out of bounds for segment_ids at b_idx={b_idx}, tok_idx={tok_idx}")
                break 
            
            segment_id_val = current_dict['segment_ids'][b_idx][tok_idx]
            
            # Check if we are past all actual content
            if last_content_idx != -1 and tok_idx > last_content_idx:
                 # Optimization: if all subsequent segment IDs are 0, print padding message and break
                if not np.any(current_dict['segment_ids'][b_idx][tok_idx:] > 0):
                    _print(f"{'...':<{idx_w}} | {'...':<{id_w}} | {'<rest is padding>':<{tok_w}} | {'...':<{id_w}} | {'...':<{tok_w}} | {'...':<{seg_w}} | {'...':<{mask_w}}")
                    break

            if segment_id_val == 0: # This token is padding or part of an empty packer
                if not has_printed_content: # First line of a (potentially) empty packer
                     _print(f"{str(tok_idx):<{idx_w}} | {'0':<{id_w}} | {'<empty/pad>':<{tok_w}} | {'-100':<{id_w}} | {'<empty/pad>':<{tok_w}} | {str(segment_id_val):<{seg_w}} | {'0':<{mask_w}}")
                     has_printed_content = True 
                     if last_content_idx == -1: # Truly empty packer
                         break 
                elif last_content_idx != -1 : # Padding after some content
                     _print(f"{str(tok_idx):<{idx_w}} | {'...':<{id_w}} | {'<pad>':<{tok_w}} | {'...':<{id_w}} | {'<pad>':<{tok_w}} | {str(segment_id_val):<{seg_w}} | {'...':<{mask_w}}")
                continue # Move to next token if it was padding

            has_printed_content = True 

            input_id_val = current_dict['input_ids'][b_idx][tok_idx]
            label_id_val = current_dict['label_ids'][b_idx][tok_idx]
            attn_mask_val = current_dict['attention_mask'][b_idx][tok_idx]

            input_token_display = str(input_id_val)
            label_token_display = str(label_id_val) if label_id_val != -100 else "---"

            if tokenizer:
                try:
                    decoded_input_token = tokenizer.decode([input_id_val])
                    input_token_display = decoded_input_token[0] if isinstance(decoded_input_token, list) and decoded_input_token else str(decoded_input_token)
                    input_token_display = repr(input_token_display)
                except Exception:
                    input_token_display = f"ID:{input_id_val}" 

                if label_id_val != -100:
                    try:
                        decoded_label_token = tokenizer.decode([label_id_val])
                        label_token_display = decoded_label_token[0] if isinstance(decoded_label_token, list) and decoded_label_token else str(decoded_label_token)
                        label_token_display = repr(label_token_display)
                    except Exception:
                        label_token_display = f"ID:{label_id_val}"
            
            input_tok_print = (input_token_display[:tok_w-3] + '...') if len(input_token_display) > tok_w else input_token_display
            label_tok_print = (label_token_display[:tok_w-3] + '...') if len(label_token_display) > tok_w else label_token_display

            _print(
                f"{str(tok_idx):<{idx_w}} | "
                f"{str(input_id_val):<{id_w}} | {input_tok_print:<{tok_w}} | "
                f"{str(label_id_val):<{id_w}} | {label_tok_print:<{tok_w}} | "
                f"{str(segment_id_val):<{seg_w}} | {str(attn_mask_val):<{mask_w}}"
            )
        _print("-" * len(header))


class Packer:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = copy.deepcopy(config)
        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = int(self.config.get("sequence_length", 1024))
        self.start_generating_id = self.config.get("start_generating_id", 2)
        self.latent_token_id = self.config.get("latent_token_id", 1)
        self.max_samples_per_packer = self.config.get("max_samples_per_packer", float('inf')) 
        self.seeded_np_rng = np.random.default_rng(self.config.get("seed", None)) # Use config seed if provided
        self.reset(self.sequence_length)
        assert self.sequence_length >= self.minimum_sequence_length, f"Seq len {self.sequence_length} < min seq len {self.minimum_sequence_length}"

    def reset(self, sequence_length=None):
        target_sequence_length = sequence_length or self.sequence_length
        self.current_dict = {
            "input_ids": np.zeros((target_sequence_length,), dtype=np.int32),
            "attention_mask": np.zeros((target_sequence_length,), dtype=np.int32),
            "label_ids": np.full((target_sequence_length,), -100, dtype=np.int32),
            "segment_ids": np.zeros((target_sequence_length,), dtype=np.int32),
            "latent_ids": np.zeros((target_sequence_length,), dtype=np.int32),
            "position_ids": np.zeros((target_sequence_length,), dtype=np.int32),
            "cross_ratios": np.zeros((target_sequence_length,), dtype=np.float32),
        }
        self.remaining_space = target_sequence_length
        self.data_index = 0
        self.segment_counter = 0 # Tracks number of samples added
        self.seeded_np_rng = np.random.default_rng(self.config.get("seed", None)) # Use config seed if provided
        return self

    def get_remaining_space(self):
        return self.remaining_space

    def is_ready(self): # This definition of "ready" is based on sequence length for popping
        return self.remaining_space <= self.minimum_sequence_length

    def can_accept(self, input_ids, label_ids):
        # Check sample limit first
        if self.segment_counter >= self.max_samples_per_packer:
            return False
        # Then check other conditions
        if not (input_ids is not None and hasattr(input_ids, '__len__') and len(input_ids) > 0):
            return False
        if not (label_ids is not None and hasattr(label_ids, '__len__') and len(label_ids) > 0):
            return False
        final_length = len(input_ids) + len(label_ids)
        return final_length <= self.remaining_space

    def add(self, input_ids, label_ids):
        if not (input_ids is not None and hasattr(input_ids, '__len__') and len(input_ids) > 0):
            raise ValueError("input_ids must be a non-empty sequence for Packer.add.")
        if not (label_ids is not None and hasattr(label_ids, '__len__') and len(label_ids) > 0):
            raise ValueError("label_ids must be a non-empty sequence for Packer.add.")

        # Pre-condition check (should ideally be guaranteed by Batcher calling can_accept first)
        if self.segment_counter >= self.max_samples_per_packer:
            raise RuntimeError(f"Packer cannot accept more samples; limit of {self.max_samples_per_packer} reached.")

        N = len(input_ids)
        M = len(label_ids)
        final_length = N + M

        if not self.can_accept(input_ids, label_ids): 
            # This check now includes both sample limit and space
            raise ValueError(f"Cannot accept sample. Check space (len {final_length}, rem {self.remaining_space}) or sample limit (count {self.segment_counter}, max {self.max_samples_per_packer}).")


        final_input_ids = list(input_ids) + [self.start_generating_id] + list(label_ids[:-1])
        prefix_len = final_length - M
        final_label_ids = ([-100] * prefix_len) + list(label_ids)

        start_idx = self.data_index
        end_idx = start_idx + final_length

        self.current_dict["input_ids"][start_idx:end_idx] = final_input_ids
        self.current_dict["label_ids"][start_idx:end_idx] = final_label_ids
        self.segment_counter += 1 # Increment after successful addition preparation
        self.current_dict["segment_ids"][start_idx:end_idx] = self.segment_counter 
        self.current_dict["attention_mask"][start_idx : start_idx + N] = 1
        self.current_dict["attention_mask"][start_idx + N : end_idx] = 3

        self.data_index += final_length
        self.remaining_space -= final_length
        return self.is_ready()

    def can_accept_with_latent(self, input_ids, label_ids, number_of_latent_tokens):
        # Check sample limit first
        if self.segment_counter >= self.max_samples_per_packer:
            return False
        # Then check other conditions
        if not (input_ids is not None and hasattr(input_ids, '__len__') and len(input_ids) > 0):
            return False
        if not (label_ids is not None and hasattr(label_ids, '__len__') and len(label_ids) > 0):
            return False
        if not (isinstance(number_of_latent_tokens, int) and number_of_latent_tokens >= 0):
            return False
        final_length = len(input_ids) + number_of_latent_tokens + len(label_ids)
        return final_length <= self.remaining_space

    def add_with_latent(self, input_ids, label_ids, number_of_latent_tokens):
        if not (input_ids is not None and hasattr(input_ids, '__len__') and len(input_ids) > 0):
            raise ValueError("input_ids must be a non-empty sequence for Packer.add_with_latent.")
        if not (label_ids is not None and hasattr(label_ids, '__len__') and len(label_ids) > 0):
            raise ValueError("label_ids must be a non-empty sequence for Packer.add_with_latent.")
        if not (isinstance(number_of_latent_tokens, int) and number_of_latent_tokens >= 0):
            raise ValueError("number_of_latent_tokens must be a non-negative integer for Packer.add_with_latent.")

        if self.segment_counter >= self.max_samples_per_packer:
             raise RuntimeError(f"Packer cannot accept more samples; limit of {self.max_samples_per_packer} reached.")

        final_length = len(input_ids) + number_of_latent_tokens + len(label_ids)

        if not self.can_accept_with_latent(input_ids, label_ids, number_of_latent_tokens):
            raise ValueError(f"Cannot accept latent sample. Check space or sample limit.")

        latent_tokens = [self.latent_token_id] * number_of_latent_tokens
        final_input_ids = list(input_ids) + latent_tokens + [self.start_generating_id] + list(label_ids[:-1])
        prefix_len = final_length - len(label_ids)
        final_label_ids = ([-100] * prefix_len) + list(label_ids)

        start_idx = self.data_index
        end_idx = start_idx + final_length

        self.current_dict["input_ids"][start_idx:end_idx] = final_input_ids
        self.current_dict["label_ids"][start_idx:end_idx] = final_label_ids
        self.segment_counter += 1
        self.current_dict["segment_ids"][start_idx:end_idx] = self.segment_counter
        
        len_input = len(input_ids)
        len_latent = number_of_latent_tokens
        len_label = len(label_ids)

        prefix_start = start_idx
        prefix_end = start_idx + len_input
        latent_start = prefix_end
        latent_end = latent_start + len_latent
        label_start = latent_end # This is start of [start_gen_id] + label_ids[:-1]
        # label_end = end_idx # Correct

        self.current_dict["attention_mask"][prefix_start : prefix_end] = 1
        self.current_dict["attention_mask"][latent_start : latent_end] = 2
        self.current_dict["attention_mask"][label_start:end_idx] = 3 # Use end_idx

        latent_ratio = (len_latent / len_input)

        prefix_pos_start = 0
        prefix_pos_end = prefix_pos_start + max(len_input, len_input * latent_ratio)

        pos_label_start = prefix_pos_end
        pos_label_end = pos_label_start + max(len_label, len_label * latent_ratio)


        # random_label_length = self.seeded_np_rng.integers(32_768, 262_144) # Random length for label part
        # pos_label_start = pos_end
        # pos_label_end:int = pos_label_start + random_label_length # Random end for label part
        # pos_label_start:int = pos_end
        # pos_label_end:int = pos_label_start + random_label_length # Random end for label part


        # self.seeded_np_rng = np.random.default_rng(self.config.get("seed", None)) # Use config seed if provided


        # Ensure num for linspace is > 0 if the length is > 0
        if len_input > 0:
            self.current_dict['position_ids'][prefix_start : prefix_end] = np.linspace(prefix_pos_start, prefix_pos_end, num=len_input, endpoint=False, dtype=np.int32)
        if len_latent > 0:
            self.current_dict['position_ids'][latent_start : latent_end] = np.linspace(prefix_pos_start, prefix_pos_end, num=len_latent, endpoint=False, dtype=np.int32)

        # For label part, its length is len(label_ids), not (end_idx - label_start) if label_ids[:-1] is empty
        label_part_len = len(label_ids) # The actual length of the label segment including start_gen_id
        if label_part_len > 0:
            self.current_dict['position_ids'][label_start : end_idx] = np.linspace(pos_label_start, pos_label_end, num=label_part_len, endpoint=False, dtype=np.int32)
            # self.current_dict['position_ids'][label_start : end_idx] = pos_end*2
            # self.current_dict['position_ids'][label_start : end_idx] = np.linspace(pos_label_start, pos_label_end, num=label_part_len, endpoint=False, dtype=np.int32)
            # self.current_dict['position_ids'][label_start : end_idx] = np.arange(label_part_len, dtype=np.int32)*1000 + (524_288 + 262_144)



        self.data_index += final_length
        self.remaining_space -= final_length
        return self.is_ready()

class Batcher:
    def __init__(self, config=None):
        if config is None: config = {}
        self.config = copy.deepcopy(config) # This config will be passed to Packers
        self.batch_size = self.config.get("batch_size", 0)
        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 32)
        self.sequence_length = int(self.config.get("sequence_length", 0))
        # max_samples_per_packer will be read by Packer from its self.config
        self.packers = []
        self.pq = []
        assert self.sequence_length >= self.minimum_sequence_length, f"Seq len {self.sequence_length} < min seq len {self.minimum_sequence_length}"
        assert self.batch_size > 0, f"Batch size {self.batch_size} must be > 0"
        self.reset(self.sequence_length, self.batch_size)

    def reset(self, sequence_length=None, batch_size=None):
        target_sequence_length = sequence_length or self.sequence_length
        target_batch_size = batch_size or self.batch_size
        self.sequence_length = target_sequence_length
        self.config["sequence_length"] = target_sequence_length # Ensure Batcher's config is up-to-date
        self.batch_size = target_batch_size
        self.config["batch_size"] = target_batch_size

        if len(self.packers) != target_batch_size:
            # Pass the batcher's config to each new Packer
            self.packers = [Packer(config=self.config) for _ in range(target_batch_size)]
        else:
            for packer in self.packers: 
                # If packers are reused, ensure their config is updated if batcher's changed, then reset
                packer.config = copy.deepcopy(self.config) # Update packer's copy of config
                packer.max_samples_per_packer = packer.config.get("max_samples_per_packer", float('inf')) # Re-apply from new config
                packer.reset(target_sequence_length)
        
        self.pq = []
        for i, packer in enumerate(self.packers):
            if not hasattr(packer, 'current_dict') or packer.sequence_length != target_sequence_length:
                 packer.reset(target_sequence_length) 
            heapq.heappush(self.pq, (packer.get_remaining_space(), i, packer))
        return self

    def get_remaining_space(self, max_or_min="max"):
        if not self.pq: return 0 
        spaces = [entry[2].get_remaining_space() for entry in self.pq if entry[2] is not None]
        if not spaces: return 0 # If all packers were popped from PQ (e.g., all full by sample count)
        return max(spaces) if max_or_min == "max" else min(spaces)

    def is_ready(self): # "Ready" for popping based on sequence length criteria
        if not self.packers: return "not ready" 
        return "ready" if all(packer.is_ready() for packer in self.packers) else "not ready"
    
    def force_ready(self):
        new_pq = []
        for idx, packer_instance in enumerate(self.packers):
            packer_instance.remaining_space = 0 
            heapq.heappush(new_pq, (packer_instance.get_remaining_space(), idx, packer_instance))
        self.pq = new_pq
        return "ready"

    def get_sample_count(self): # Total samples across all packers
        return sum(packer.segment_counter for packer in self.packers)

    def _add_internal(self, add_method_type: str, sample_args: tuple, force_finish_pack: bool):
        buffer = []
        packer_found = False

        # Rebuild PQ with only packers that can potentially accept more samples
        # This ensures we don't try to pop from PQ if all packers are full by sample count
        current_available_packers_in_pq = []
        while self.pq:
            entry = heapq.heappop(self.pq)
            packer_to_check = entry[2]
            can_this_packer_ever_accept_more_samples = True
            if add_method_type == "add":
                # Check sample limit directly as can_accept includes it
                 if packer_to_check.segment_counter >= packer_to_check.max_samples_per_packer:
                     can_this_packer_ever_accept_more_samples = False
            elif add_method_type == "add_with_latent":
                 if packer_to_check.segment_counter >= packer_to_check.max_samples_per_packer:
                     can_this_packer_ever_accept_more_samples = False
            
            if can_this_packer_ever_accept_more_samples:
                 current_available_packers_in_pq.append(entry)
            else: # If full by sample count, keep it aside (won't be added to for this call)
                 buffer.append(entry) # Add to buffer to restore later if not used

        if not current_available_packers_in_pq:
            # Restore all packers to PQ if none were available by sample count
            while buffer: heapq.heappush(self.pq, buffer.pop(0)) # FIFO for buffer restore
            return "full" # No packer can accept due to sample limits or other reasons

        # Use the filtered list for attempting to add
        temp_pq_for_add = current_available_packers_in_pq
        heapq.heapify(temp_pq_for_add) # Make it a valid heap again

        while temp_pq_for_add: # Iterate through packers that are not yet full by sample count
            remaining_space_val, idx, packer = heapq.heappop(temp_pq_for_add)
            
            can_accept_current = False
            actual_add_fn = None
            
            if add_method_type == "add":
                if len(sample_args) == 2:
                    can_accept_current = packer.can_accept(sample_args[0], sample_args[1]) # This now checks sample limit too
                    actual_add_fn = packer.add
            elif add_method_type == "add_with_latent":
                if len(sample_args) == 3:
                    can_accept_current = packer.can_accept_with_latent(sample_args[0], sample_args[1], sample_args[2])
                    actual_add_fn = packer.add_with_latent
            
            if can_accept_current:
                actual_add_fn(*sample_args) 
                current_packer_remaining_space_for_pq = 0 if force_finish_pack else packer.get_remaining_space()
                heapq.heappush(self.pq, (current_packer_remaining_space_for_pq, idx, packer)) # Add back to main PQ
                packer_found = True
                break # Found a packer
            else:
                # This packer couldn't accept (e.g., due to length), put it in main buffer
                buffer.append((remaining_space_val, idx, packer))
        
        # Restore remaining from temp_pq_for_add and buffer to main_pq
        while temp_pq_for_add: heapq.heappush(self.pq, heapq.heappop(temp_pq_for_add))
        while buffer: heapq.heappush(self.pq, buffer.pop(0)) # FIFO for buffer restore

        return self.is_ready() if packer_found else "full"


    def can_accept(self, input_ids, label_ids):
        # Check if *any* packer (considering its current state and sample limit) can accept
        return any(p.can_accept(input_ids, label_ids) for p in self.packers)


    def add(self, input_ids, label_ids, force_finish_pack=False):
        return self._add_internal(add_method_type="add", 
                                  sample_args=(input_ids, label_ids), 
                                  force_finish_pack=force_finish_pack)

    def can_accept_with_latent(self, input_ids, label_ids, number_of_latent_tokens):
        return any(p.can_accept_with_latent(input_ids, label_ids, number_of_latent_tokens) for p in self.packers)


    def add_with_latent(self, input_ids, label_ids, number_of_latent_tokens, force_finish_pack=False):
        return self._add_internal(add_method_type="add_with_latent", 
                                  sample_args=(input_ids, label_ids, number_of_latent_tokens), 
                                  force_finish_pack=force_finish_pack)

    def pop(self, peek=False):
        if not self.packers:
            dummy_packer_config = self.config.copy()
            dummy_packer_config["sequence_length"] = self.sequence_length
            dummy_packer = Packer(config=dummy_packer_config)
            keys_to_stack = dummy_packer.current_dict.keys()
            stacked_dict = {}
            for key in keys_to_stack:
                dtype = dummy_packer.current_dict[key].dtype
                shape = (self.batch_size, self.sequence_length)
                if key == "label_ids":
                    stacked_dict[key] = np.full(shape, -100, dtype=dtype)
                else:
                    stacked_dict[key] = np.zeros(shape, dtype=dtype)
        else:
            keys_to_stack = self.packers[0].current_dict.keys()
            stacked_dict = {
                key: np.stack([p.current_dict[key] for p in self.packers])
                for key in keys_to_stack
            }

        if not peek: self.reset()
        return stacked_dict

if __name__ == "__main__":
    print("=== Test Suite for Packer/Batcher (with max_samples_per_packer) ===\n")
    
    # --- Config for tests ---
    base_test_config = {
        "sequence_length": 128, 
        "minimum_sequence_length": 4, # Low for testing "ready by length" easily
        "start_generating_id": 258, 
        "latent_token_id": 259,   
    }

    class DebugTokenizer: # Simple tokenizer for testing
        def __init__(self):
            self.vocab = {i: chr(65+i) for i in range(26)} # A-Z for 0-25
            self.vocab[base_test_config["start_generating_id"]] = "<S>" 
            self.vocab[base_test_config["latent_token_id"]] = "<L>"   
            for i in range(100, 126): self.vocab[i] = f"l{i-100}" # labels like l0, l1
        def decode(self, token_ids: list[int]) -> list[str]: 
            if not token_ids: return [""] 
            return [self.vocab.get(tid, f"ID:{tid}") for tid in token_ids]

    tokenizer = DebugTokenizer()

    # --- Test 1: max_samples_per_packer = 1, batch_size = 2 ---
    print("--- Test: max_samples_per_packer = 1, batch_size = 2 ---")
    config_s1_b2 = {**base_test_config, "batch_size": 2, "max_samples_per_packer": 1}
    batcher_s1_b2 = Batcher(config=config_s1_b2)
    
    sample1_in, sample1_lb = [0,1], [100,101] # Len 2+2=4
    sample2_in, sample2_lb = [2,3], [102,103] # Len 2+2=4
    sample3_in, sample3_lb = [4,5], [104,105] # Len 2+2=4

    print(f"Packer 0 max_samples: {batcher_s1_b2.packers[0].max_samples_per_packer}, Packer 1 max_samples: {batcher_s1_b2.packers[1].max_samples_per_packer}")

    print("\nAdding sample 1...")
    status = batcher_s1_b2.add(sample1_in, sample1_lb)
    print(f"Status: {status}, Samples in batcher: {batcher_s1_b2.get_sample_count()}")
    print(f"Packer 0 samples: {batcher_s1_b2.packers[0].segment_counter}, Packer 1 samples: {batcher_s1_b2.packers[1].segment_counter}")


    print("\nAdding sample 2...")
    status = batcher_s1_b2.add(sample2_in, sample2_lb)
    print(f"Status: {status}, Samples in batcher: {batcher_s1_b2.get_sample_count()}")
    print(f"Packer 0 samples: {batcher_s1_b2.packers[0].segment_counter}, Packer 1 samples: {batcher_s1_b2.packers[1].segment_counter}")

    print("\nAdding sample 3 (should be 'full' due to max_samples_per_packer)...")
    status = batcher_s1_b2.add(sample3_in, sample3_lb)
    print(f"Status: {status}, Samples in batcher: {batcher_s1_b2.get_sample_count()}") # Should still be 2

    print(f"\nBatcher is_ready (by length): {batcher_s1_b2.is_ready()}") # Likely not ready by length
    print("Forcing batcher ready...")
    batcher_s1_b2.force_ready()
    print(f"Batcher is_ready after force: {batcher_s1_b2.is_ready()}")
    
    popped_batch_s1_b2 = batcher_s1_b2.pop(peek=False)
    print("\nPopped batch details:")
    debug_alignments(popped_batch_s1_b2, tokenizer=tokenizer)

    # --- Test 2: max_samples_per_packer = 2, batch_size = 1 ---
    #    Packer should take 2 samples then be full by sample count.
    print("\n\n--- Test: max_samples_per_packer = 2, batch_size = 1 ---")
    config_s2_b1 = {**base_test_config, "batch_size": 1, "max_samples_per_packer": 2}
    batcher_s2_b1 = Batcher(config=config_s2_b1)

    print(f"Packer 0 max_samples: {batcher_s2_b1.packers[0].max_samples_per_packer}")

    sample4_in, sample4_lb = [6,7,8], [106,107,108] # Len 3+3=6
    sample5_in, sample5_lb = [9,10,11], [109,110,111] # Len 3+3=6
    sample6_in, sample6_lb = [12,13,14], [112,113,114] # Len 3+3=6

    print("\nAdding sample 4...")
    status = batcher_s2_b1.add(sample4_in, sample4_lb)
    print(f"Status: {status}, Samples in packer 0: {batcher_s2_b1.packers[0].segment_counter}")

    print("\nAdding sample 5...")
    status = batcher_s2_b1.add(sample5_in, sample5_lb)
    print(f"Status: {status}, Samples in packer 0: {batcher_s2_b1.packers[0].segment_counter}") # Should be 2

    print("\nAdding sample 6 (should be 'full')...")
    status = batcher_s2_b1.add(sample6_in, sample6_lb)
    print(f"Status: {status}, Samples in packer 0: {batcher_s2_b1.packers[0].segment_counter}") # Should still be 2

    print(f"\nBatcher is_ready (by length): {batcher_s2_b1.is_ready()}")
    if batcher_s2_b1.is_ready() != "ready":
        print("Forcing batcher ready...")
        batcher_s2_b1.force_ready()
    
    popped_batch_s2_b1 = batcher_s2_b1.pop(peek=False)
    print("\nPopped batch details:")
    debug_alignments(popped_batch_s2_b1, tokenizer=tokenizer)
    
    print("\n=== End of All Tests ===")
