# test_packer_batcher.py

import unittest
import numpy as np
from packer_batcher import Packer, Batcher




def align(*arrs, names=None):
	names = [*names]
	headers = ',    '.join(names)
	print(f"*" * 60,)
	print('\t   ' + headers)
	for idx in range(len(arrs[0])):
		to_print = {
			"idx": f"{idx:>3d}",
		}
		for arr_idx, arr in enumerate(arrs):
			to_print[names[arr_idx]] = arrs[arr_idx][idx]
		def sc(x):
			try:
				x = f"{x:>3d}"
			except:
				pass
			return x
		to_print_str = ", ".join([f"{k}: {sc(v)}" for k, v in to_print.items()])
		to_print_str = '\t' + to_print_str
		print(to_print_str)
	print(f"*" * 60,)



class TestPackerBatcher(unittest.TestCase):

	# def test_packer_single_sequence(self):
	#     packer = Packer(sequence_length=10)
	#     input_ids = [1, 2, 3]
	#     label_ids = [4, 5]
	#     packer.add(input_ids, label_ids)
	#     self.assertTrue(packer.is_ready())
	#     packed_data = packer.pop()
	#     np.testing.assert_array_equal(packed_data['input_ids'][:4], [1, 2, 3, 4])
	#     np.testing.assert_array_equal(packed_data['label_ids'][:4], [-100, -100, 4, 5])
	#     np.testing.assert_array_equal(packed_data['attention_mask'][:4], [1, 1, 1, 0])
	#     np.testing.assert_array_equal(packed_data['segment_ids'][:4], [1, 1, 1, 1])
	#     self.assertEqual(packer.current_length, 0)
	#     self.assertFalse(packer.is_ready())


	def test_packer_multiple_sequences(self):
		packer = Batcher({
			"sequence_length":16,
		}) # Increased sequence length to accommodate both sequences
		packer.add([1, 2, 3], [4, 5])
		packer.add([6, 7], [8, 9, 10])

		# self.assertTrue(packer.is_ready())
		packed_data = packer.pop()

		expected_input_ids = [1, 2, 3, 4, 6, 7, 8, 9, 0,]
		expected_label_ids = [-100, -100, 4, 5, -100, 8,9, 10]
		expected_segment_ids = [1, 1, 1, 1, 2, 2, 2, 2, 0,]
		expected_attention_mask = [1, 1, 1, 0, 1, 1,]

		# pad with 0's
		expected_input_ids += [0] * (16 - len(expected_input_ids))
		expected_label_ids += [-100] * (16 - len(expected_label_ids))
		expected_segment_ids += [0] * (16 - len(expected_segment_ids))
		expected_attention_mask += [0] * (16 - len(expected_attention_mask))

		for expected_input_ids_idx, (_expected_input_ids) in enumerate(expected_input_ids):
			print(f"  expected_input_ids[{expected_input_ids_idx}]: {_expected_input_ids}")
		print(f"*" * 60,)
			
		for expected_label_ids_idx, (_expected_label_ids) in enumerate(expected_label_ids):
			print(f"  expected_label_ids[{expected_label_ids_idx}]: {_expected_label_ids}")
			
		assert(len(expected_label_ids) == len(expected_input_ids))
		# expected_attention_mask = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		# expected_segment_ids = [1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
		
		for packed_data_idx, (key, value) in enumerate(packed_data.items()):
			value = value[:1]
			print(f"  packed_data[{key}]: {value.tolist()}")

		input_ids = packed_data['input_ids'][0]
		label_ids = packed_data['label_ids'][0]

		align(
			input_ids, expected_input_ids,
			names=["aid", "eid"],
		)
		
		align(
			label_ids, expected_label_ids,
			names=["ali", "eli"],
		)

		align(
			packed_data['attention_mask'][0], expected_attention_mask,
			names=["aam", "eam"],
		)

		align(
			packed_data['segment_ids'][0], expected_segment_ids,
			names=["asi", "esi"],
		)


		# exit()
			

		np.testing.assert_array_equal(packed_data['input_ids'][0][:7], expected_input_ids[:7])
		np.testing.assert_array_equal(packed_data['label_ids'][0][:7], expected_label_ids[:7])
		# np.testing.assert_array_equal(packed_data['attention_mask'][:7], expected_attention_mask[:7])
		# np.testing.assert_array_equal(packed_data['segment_ids'][:7], expected_segment_ids[:7])

		# Check padding part (from length 7 to 15, sequence_length)
		# np.testing.assert_array_equal(packed_data['input_ids'][7:], [0] * 8)
		# np.testing.assert_array_equal(packed_data['label_ids'][7:], [-100] * 8)
		# np.testing.assert_array_equal(packed_data['attention_mask'][7:], [0] * 8)
		# np.testing.assert_array_equal(packed_data['segment_ids'][7:], [0] * 8)


	# def test_packer_padding(self): # This test may need adjustment based on new packing logic
	#     packer = Packer(sequence_length=10)
	#     input_ids = [1, 2, 3]
	#     label_ids = [4, 5, 6] # Label IDs same length for this test - not really relevant now
	#     packer.add(input_ids, label_ids)
	#     packed_data = packer.pop()
	#     self.assertEqual(len(packed_data['input_ids']), 10)
	#     self.assertEqual(len(packed_data['label_ids']), 10)
	#     self.assertEqual(len(packed_data['attention_mask']), 10)
	#     self.assertEqual(len(packed_data['segment_ids']), 10)
	#     np.testing.assert_array_equal(packed_data['input_ids'][:4], [1, 2, 3, 4]) # packed input ids is input + first label. Here label is [4, 5, 6], so first is 4.
	#     np.testing.assert_array_equal(packed_data['input_ids'][4:], [0, 0, 0, 0, 0, 0])
	#     np.testing.assert_array_equal(packed_data['label_ids'][:4], [-100, -100, 4, 5])
	#     np.testing.assert_array_equal(packed_data['label_ids'][4:], [-100, -100, -100, -100, -100, -100])
	#     np.testing.assert_array_equal(packed_data['attention_mask'][:4], [1, 1, 1, 0])
	#     np.testing.assert_array_equal(packed_data['attention_mask'][4:], [0, 0, 0, 0, 0, 0])
	#     np.testing.assert_array_equal(packed_data['segment_ids'][:4], [1, 1, 1, 1])
	#     np.testing.assert_array_equal(packed_data['segment_ids'][4:], [0, 0, 0, 0, 0, 0])



	# def test_batcher_add_pop(self): # May need adjustment based on new packing
	#     config = {'batch_size': 2, 'sequence_length': 10}
	#     batcher = Batcher(config)
	#     batcher.add([1, 2, 3], [4, 5])
	#     batcher.add([7, 8], [9])
	#     batch_data = batcher.pop()
	#     self.assertIsNotNone(batch_data)
	#     self.assertEqual(len(batch_data['input_ids']), 2) # Batch size 2
	#     np.testing.assert_array_equal(batch_data['input_ids'][0][:4], [1, 2, 3, 4]) # Sequence 1 packed input
	#     np.testing.assert_array_equal(batch_data['input_ids'][0][4:], [0,0,0,0,0,0])
	#     np.testing.assert_array_equal(batch_data['input_ids'][1][:3], [7, 8, 9]) # Sequence 2 packed input
	#     np.testing.assert_array_equal(batch_data['input_ids'][1][3:], [0,0,0,0,0,0,0])

	#     np.testing.assert_array_equal(batch_data['label_ids'][0][:4], [-100, -100, 4, 5]) # Sequence 1 packed labels
	#     np.testing.assert_array_equal(batch_data['label_ids'][0][4:], [-100, -100, -100, -100, -100, -100])
	#     np.testing.assert_array_equal(batch_data['label_ids'][1][:3], [-100, -100, 9]) # Sequence 2 packed labels
	#     np.testing.assert_array_equal(batch_data['label_ids'][1][3:], [-100, -100, -100, -100, -100, -100, -100])

	#     np.testing.assert_array_equal(batch_data['attention_mask'][0][:4], [1, 1, 1, 0]) # Sequence 1 attention mask
	#     np.testing.assert_array_equal(batch_data['attention_mask'][0][4:], [0, 0, 0, 0, 0, 0])
	#     np.testing.assert_array_equal(batch_data['attention_mask'][1][:3], [1, 1, 0]) # Sequence 2 attention mask
	#     np.testing.assert_array_equal(batch_data['attention_mask'][1][3:], [0, 0, 0, 0, 0, 0, 0])

	#     np.testing.assert_array_equal(batch_data['segment_ids'][0][:4], [1, 1, 1, 1]) # Sequence 1 segment ids
	#     np.testing.assert_array_equal(batch_data['segment_ids'][0][4:], [0, 0, 0, 0, 0, 0])
	#     np.testing.assert_array_equal(batch_data['segment_ids'][1][:3], [2, 2, 2]) # Sequence 2 segment ids.
	#     np.testing.assert_array_equal(batch_data['segment_ids'][1][3:], [0, 0, 0, 0, 0, 0, 0])


	# def test_batcher_full_batch_pop(self): # May need adjustment based on new packing
	#     config = {'batch_size': 2, 'sequence_length': 5, 'minimum_sequence_length': 4}
	#     batcher = Batcher(config)
	#     batcher.add([1, 2, 3, 4], [5, 6, 7]) # Packer 0 becomes ready - packed length 4+1=5
	#     batcher.add([9, 10, 11, 12], [13, 14, 15]) # Packer 1 becomes ready - packed length 4+1=5
	#     batch_data = batcher.pop()
	#     self.assertIsNotNone(batch_data)
	#     self.assertEqual(len(batch_data['input_ids']), 2) # Batch size 2


	# def test_batcher_add_more_than_batch_size(self): # May need adjustment
	#     config = {'batch_size': 2, 'sequence_length': 10}
	#     batcher = Batcher(config)
	#     batcher.add([1, 2, 3], [4, 5]) # Packer 0 - len 4
	#     batcher.add([7, 8], [9]) # Packer 1 - len 3
	#     batcher.add([11, 12, 13], [14, 15]) # Try to add to packer 0 again
	#     batch_data = batcher.pop()
	#     self.assertIsNotNone(batch_data)
	#     self.assertEqual(len(batch_data['input_ids']), 2) # Batch size 2


	# def test_batcher_reset(self):
	#     config = {'batch_size': 2, 'sequence_length': 10}
	#     batcher = Batcher(config)
	#     batcher.add([1, 2, 3], [4, 5, 6])
	#     batcher.reset()
	#     batch_data = batcher.pop()
	#     self.assertIsNone(batch_data) # After reset, pop should return None


if __name__ == '__main__':
	unittest.main()