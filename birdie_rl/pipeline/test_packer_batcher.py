# test_packer_batcher.py

import unittest
import numpy as np
# Ensure this import path is correct if test_utils is in the same directory or a discoverable path
# from ..tests.test_utils import align # If test_utils is in parent/tests
# For simplicity, assuming align is defined or imported correctly elsewhere, or not strictly needed for this refactor focus.

# Updated import to use packer_batcher2
from birdie_rl.pipeline.packer_batcher2 import Packer, Batcher

# Dummy align function if test_utils.align is not available or for isolated testing
def align(*arrs, names=None):
	if names is None:
		names = [f"arr{i}" for i in range(len(arrs))]
	print(f"*" * 60)
	headers = ',    '.join(names)
	print(f'\t    idx, ' + headers)
	for idx in range(len(arrs[0])):
		to_print_parts = [f"{idx:>3d}"]
		for arr_idx, arr in enumerate(arrs):
			val = arr[idx]
			try:
				val_str = f"{val:>5}" # Attempt to format as number
			except TypeError:
				val_str = str(val) # Fallback to string
			to_print_parts.append(f"{names[arr_idx]}: {val_str}")
		print('\t' + ", ".join(to_print_parts))
	print(f"*" * 60)


class TestPackerBatcher(unittest.TestCase):

	def test_packer_multiple_sequences(self):
		# Config for the new Batcher/Packer
		packer_config = {
			"sequence_length": 16,
			"minimum_sequence_length": 4, # From old packer for consistency in test logic
			"start_generating_id": 99, # Example ID for <S>
			"latent_token_id": 98,     # Example ID for <L>
			"max_samples_per_packer": float('inf'), # Default
			"seed": 0
		}
		# Batcher in packer_batcher2.py takes a config for itself, which it also passes to its Packers.
		# For this test, we are testing the Batcher's ability to manage one Packer (batch_size=1).
		batcher_config = {
			"batch_size": 1, # Testing a single packer scenario via Batcher
			**packer_config # Packer will use these settings from Batcher's config
		}
		batcher = Batcher(config=batcher_config)

		# Sample 1: input_ids = [1, 2, 3], label_ids = [4, 5]
		# N=3, M=2. Packer.add logic:
		# final_input_ids = [1, 2, 3, 99, 4] (len 5)
		# final_label_ids = [-100, -100, -100, 4, 5] (len 5)
		# Segment ID will be 1 for these 5 tokens.
		batcher.add(np.array([1, 2, 3]), np.array([4, 5]))

		# Sample 2: input_ids = [6, 7], label_ids = [8, 9, 10]
		# N=2, M=3. Packer.add logic:
		# final_input_ids = [6, 7, 99, 8, 9] (len 5)
		# final_label_ids = [-100, -100, 8, 9, 10] (len 5)
		# Segment ID will be 2 for these 5 tokens.
		batcher.add(np.array([6, 7]), np.array([8, 9, 10]))

		# The packer (Batcher with batch_size=1) now contains these two segments.
		# Total length = 5 (from sample1) + 5 (from sample2) = 10.
		# Remaining space = 16 - 10 = 6. This is > minimum_sequence_length (4), so not "ready" by length.
		# To pop, we might need to force it or add more until ready.
		# For testing, let's assume we force pop or it becomes ready.
		# The test logic in packer_batcher.py implies popping when the Batcher is ready.
		# Let's make it ready by filling more or using force_ready.
		# Current data_index = 10. Remaining space = 6.
		# If we add another sample that fits, e.g., N=1, M=1 (total 2 tokens):
		# input_ids = [11], label_ids = [12]
		# final_input_ids = [11, 99] (len 2)
		# final_label_ids = [-100, 12] (len 2)
		# This would make data_index = 12, remaining_space = 4. Now it's ready.
		batcher.add(np.array([11]), np.array([12]))


		self.assertEqual(batcher.is_ready(), "ready") # Should be ready now
		packed_data = batcher.pop() # Pop the single batch item

		# packed_data is a dict where each value is a NumPy array of shape (batch_size, sequence_length)
		# Here batch_size is 1.
		self.assertIsNotNone(packed_data)
		self.assertEqual(packed_data['input_ids'].shape, (1, 16))

		# Expected combined sequence in the single packer
		# Sample 1: [1, 2, 3, 99, 4]
		# Sample 2: [6, 7, 99, 8, 9]
		# Sample 3: [11, 99]
		# Padded to 16
		expected_input_ids_flat = [1, 2, 3, 99, 4, 6, 7, 99, 8, 9, 11, 99, 0, 0, 0, 0]
		expected_label_ids_flat = [-100, -100, -100, 4, 5, -100, -100, 8, 9, 10, -100, 12, -100, -100, -100, -100]
		expected_segment_ids_flat = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0]
		# Attention mask logic from packer_batcher2.Packer.add:
		# input_ids part: 1, start_generating_id + label_ids[:-1] part: 3 (for teacher forcing)
		# Sample 1: input [1,2,3] (mask 1), [99,4] (mask 3)
		# Sample 2: input [6,7] (mask 1), [99,8,9] (mask 3)
		# Sample 3: input [11] (mask 1), [99] (mask 3)
		expected_attention_mask_flat = [1,1,1, 3,3,  1,1, 3,3,3,  1, 3,  0,0,0,0]


		print("\nPacked Data from Batcher (Batch Size 1):")
		for key, value_array in packed_data.items():
			print(f"  {key}: {value_array[0].tolist()}") # Print the first (only) item in the batch

		np.testing.assert_array_equal(packed_data['input_ids'][0], expected_input_ids_flat)
		np.testing.assert_array_equal(packed_data['label_ids'][0], expected_label_ids_flat)
		np.testing.assert_array_equal(packed_data['segment_ids'][0], expected_segment_ids_flat)
		np.testing.assert_array_equal(packed_data['attention_mask'][0], expected_attention_mask_flat)


if __name__ == '__main__':
	unittest.main()
