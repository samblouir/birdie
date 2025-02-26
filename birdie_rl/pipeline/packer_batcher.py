import numpy as np
import copy
import heapq


def debug_alignments(current_dict, sub_idx=0):
    for idx in range(len(current_dict["input_ids"][sub_idx])):
        print(
            f"idx: {idx}, "
            f"input_ids: {current_dict['input_ids'][sub_idx][idx]}, "
            f"label_ids: {current_dict['label_ids'][sub_idx][idx]}, "
            f"segment_ids: {current_dict['segment_ids'][sub_idx][idx]}, "
            f"attention_mask: {current_dict['attention_mask'][sub_idx][idx]}"
        )


class Packer:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = copy.deepcopy(config)
        del config

        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = int(self.config.get("sequence_length", 1024))
        self.start_generating_paradigm = self.config.get("start_generating_paradigm", "\n<|assistant|>\n")
        self.tokenized_start_generating_paradigm = self.config.get("tokenizer", None).encode(self.start_generating_paradigm)
        self.reset(self.sequence_length)
        assert self.sequence_length >= self.minimum_sequence_length

    def reset(self, sequence_length=None):
        target_sequence_length = sequence_length or self.sequence_length

        self.current_dict = {
            "input_ids": np.zeros((target_sequence_length,), dtype=np.int32),
            "attention_mask": np.zeros((target_sequence_length,), dtype=np.int32),
            "label_ids": np.zeros((target_sequence_length,), dtype=np.int32) - 100,
            "segment_ids": np.zeros((target_sequence_length,), dtype=np.int32),
        }
        self.remaining_space = target_sequence_length
        self.sequence_length = target_sequence_length
        self.data_index = 0
        self.segment_counter = 0

        return self

    def get_remaining_space(self):
        # Reserve space for the paradigm tokens.
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        return max(0, self.remaining_space - paradigm_len)

    def is_ready(self):
        return self.remaining_space <= self.minimum_sequence_length

    def can_accept(self, input_ids, label_ids):
        """
        We add tokens in a 'teacher forcing' style:
        - input_ids (N tokens)
        - label_ids (N tokens)
        But effectively, it's N + (N-1) = 2N - 1 tokens, plus the paradigm tokens.
        """
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1
        return total_length_to_add <= self.remaining_space

    def add(self, input_ids, label_ids, loss_mask=None):
        # Calculate the total number of tokens to add (including paradigm tokens)
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1

        if not self.can_accept(input_ids, label_ids):
            raise ValueError(
                f"Insufficient space to add {total_length_to_add:,} tokens.  "
                f"remaining_space: {self.remaining_space:,}"
            )
        
        # Append paradigm tokens to input_ids
        input_ids = np.concatenate([input_ids, self.tokenized_start_generating_paradigm])
        self.segment_counter += 1
        
        input_start = self.data_index
        input_end = input_start + len(input_ids)  # len(input_ids) now includes paradigm tokens

        label_start = input_end - 1
        label_end = label_start + len(label_ids)

        # The teacher-forcing input region is the label sequence minus its last token
        input_teacher_forcing_start = input_end
        input_teacher_forcing_end = label_end

        # Sanity check:
        assert (input_teacher_forcing_end - input_start) == total_length_to_add

        # Fill in input_ids and attention_mask for the original input region
        self.current_dict["input_ids"][input_start:input_end] = input_ids
        self.current_dict["attention_mask"][input_start:input_end] = 1

        # Teacher-forcing inputs (the label sequence minus the last token)
        self.current_dict["input_ids"][input_teacher_forcing_start:input_teacher_forcing_end] = label_ids[:-1]

        # Fill in label_ids
        self.current_dict["label_ids"][label_start:label_end] = label_ids

        # Fill in segment_ids
        self.current_dict["segment_ids"][input_start:input_teacher_forcing_end] = self.segment_counter

        # Update indices / remaining_space by the full total tokens added
        self.data_index += total_length_to_add
        self.remaining_space -= total_length_to_add

        return self.is_ready()


class Batcher:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = copy.deepcopy(config)
        del config

        self.batch_size = self.config.get("batch_size", 8)
        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = int(self.config.get("sequence_length", 1024))

        # Store packers both in a list (for indexing) and in a min-heap (priority queue).
        self.packers = []
        self.pq = []  # priority queue
        self.reset(self.sequence_length, self.batch_size)

    def reset(self, sequence_length=None, batch_size=None):
        target_sequence_length = sequence_length or self.sequence_length
        target_batch_size = batch_size or self.batch_size

        self.sequence_length = target_sequence_length
        self.config["sequence_length"] = target_sequence_length
        self.batch_size = target_batch_size
        self.config["batch_size"] = target_batch_size

        self.packers = []
        self.pq = []

        for i in range(self.batch_size):
            packer = Packer(config=self.config)
            self.packers.append(packer)
            # Push (remaining_space, index, packer) into the priority queue
            heapq.heappush(self.pq, (packer.get_remaining_space(), i, packer))

        return self

    def get_remaining_space(self, max_or_min="max"):
        """
        Return the max or min remaining space across all packers, depending on the argument.
        """
        if max_or_min == "max":
            return max(entry[2].get_remaining_space() for entry in self.pq)
        else:
            return min(entry[2].get_remaining_space() for entry in self.pq)

    def is_ready(self):
        # All packers are "ready" if each is at or below the minimum sequence length remaining.
        if all(entry[2].is_ready() for entry in self.pq):
            return "ready"
        else:
            return "not ready"

    def can_accept(self, input_ids, label_ids):
        # If any packer can accept, we return True.
        return any(entry[2].can_accept(input_ids, label_ids) for entry in self.pq)

    def add(self, input_ids, label_ids, loss_mask=None, force_finish_pack=False):
        """
        Prioritize packers with the least remaining space that can still accept the input.
        We pop from the min-heap; if the top packer can accept, we use it.
        Otherwise, we temporarily store packers until we find one that can accept.
        """
        buffer = []
        packer_found = False

        # Pop from the heap, checking if the packer can accept the tokens.
        while self.pq:
            remaining_space, idx, packer = heapq.heappop(self.pq)

            if packer.can_accept(input_ids, label_ids):
                packer.add(input_ids, label_ids, loss_mask=loss_mask)
                # Push it back with its updated remaining space.
                new_remaining_space = 0 if force_finish_pack else packer.get_remaining_space()
                heapq.heappush(self.pq, (new_remaining_space, idx, packer))
                packer_found = True
                break
            else:
                # Temporarily store this packer.
                buffer.append((remaining_space, idx, packer))

        # Push all temporarily removed packers back onto the heap.
        while buffer:
            heapq.heappush(self.pq, buffer.pop())

        if not packer_found:
            return "full"

        return self.is_ready()

    def pop(self, peek=False):
        """
        Combine the current_dicts from all packers into a single batched dictionary.
        The order is determined by the original packer index.
        """
        sorted_packers = sorted(self.pq, key=lambda x: x[1])
        keys_to_stack = sorted_packers[0][2].current_dict.keys()
        stacked_dict = {
            key: np.stack([p[2].current_dict[key] for p in sorted_packers])
            for key in keys_to_stack
        }

        if not peek:
            self.reset()

        return stacked_dict

    def get_sample_count(self):
        running_total = 0
        for entry in self.pq:
            running_total += entry[2].segment_counter
        return running_total
