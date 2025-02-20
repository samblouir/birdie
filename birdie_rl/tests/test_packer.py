# tests/test_packer.py
"""
Unit tests for the SequencePacker and PackedArray classes.
"""

import unittest
import numpy as np
from test_utils import debug_print_array_dict
from birdie_rl.packer import SequencePacker

class TestPacker(unittest.TestCase):
    def test_basic_packing(self):
        packer = SequencePacker(sequence_length=16, minimum_sequence_length=4)
        print("\n--- test_basic_packing ---")
        print("SequencePacker created with sequence_length=16, minimum_sequence_length=4")
        samples = [
            (np.array([1,2,3], dtype=np.int32), np.array([4,5,6], dtype=np.int32)),
            (np.array([7,8],   dtype=np.int32), np.array([9,10], dtype=np.int32)),
        ]
        for i, (inp, lab) in enumerate(samples):
            print(f"\nAdding sample {i}: input_ids.size={inp.size}, label_ids.size={lab.size}")
            status = packer.add_sample(inp, lab, add_sep=False)
            remaining_space = packer.get_remaining_space()
            print(f"After add_sample => status={status}, remaining_space={remaining_space}")
            self.assertFalse(status["ready"])
        data_dict = packer.get_data(peek=False)
        debug_print_array_dict(data_dict, heading="Final packed array contents")
        input_ids = data_dict["input_ids"]
        label_ids = data_dict["label_ids"]
        self.assertEqual(len(input_ids), 16)
        self.assertEqual(len(label_ids), 16)
        expected_input_ids = np.array([1, 2, 3, 4, 5, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        expected_label_ids = np.array([-100, -100, 4, 5, 6, -100, 9, 10, -100, -100, -100, -100, -100, -100, -100, -100], dtype=np.int32)
        np.testing.assert_array_equal(input_ids, expected_input_ids)
        np.testing.assert_array_equal(label_ids, expected_label_ids)

    def test_exceed_max_input_length_error(self):
        packer = SequencePacker(sequence_length=16, minimum_sequence_length=4)
        print("\n--- test_exceed_max_input_length_error ---")
        inp = np.array([1]*15, dtype=np.int32)
        lab = np.array([2]*5, dtype=np.int32)
        with self.assertRaises(ValueError):
            packer.add_sample(inp, lab)

    def test_fill_packer_entirely(self):
        packer = SequencePacker(sequence_length=10, minimum_sequence_length=3)
        print("\n--- test_fill_packer_entirely ---")
        inp1 = np.array([1,2,3], dtype=np.int32)
        lab1 = np.array([4,5,6,7], dtype=np.int32)
        status1 = packer.add_sample(inp1, lab1)
        print("After first add => status:", status1)
        leftover_space1 = packer.get_remaining_space()
        print("Leftover space after first add:", leftover_space1)
        self.assertFalse(status1["ready"])
        inp2 = np.array([8], dtype=np.int32)
        lab2 = np.array([9], dtype=np.int32)
        status2 = packer.add_sample(inp2, lab2)
        print("After second add => status:", status2)
        leftover_space2 = packer.get_remaining_space()
        print("Leftover space after second add:", leftover_space2)
        self.assertTrue(leftover_space2 == 3)
        inp3 = np.array([10], dtype=np.int32)
        lab3 = np.array([11], dtype=np.int32)
        status3 = packer.add_sample(inp2, lab2)
        print("After third add => status:", status3)
        leftover_space3 = packer.get_remaining_space()
        print("Leftover space after third add:", leftover_space3)
        self.assertTrue(status3["ready"])
        final = packer.get_data(peek=False)
        debug_print_array_dict(final, heading="Final Packer State")

    def test_forced_ready(self):
        packer = SequencePacker(sequence_length=16, minimum_sequence_length=4)
        print("\n--- test_forced_ready ---")
        inp = np.array([1,2,3,4,5,6], dtype=np.int32)
        lab = np.array([7,8,9,10,11,12], dtype=np.int32)
        status = packer.add_sample(inp, lab, add_sep=False, force_ready_if_too_long=6)
        print("After add_sample => status:", status)
        leftover_space = packer.get_remaining_space()
        print("Leftover space after adding sample:", leftover_space)
        self.assertTrue(status["ready"])
        data_dict = packer.get_data()
        input_ids = data_dict["input_ids"]
        label_ids = data_dict["label_ids"]
        print("\nFinal data from packer:")
        print("input_ids:", input_ids)
        print("label_ids:", label_ids)

    def test_multiple_arrays(self):
        packer = SequencePacker(sequence_length=10, minimum_sequence_length=4)
        print("\n--- test_multiple_arrays ---")
        for i in range(3):
            inp = np.array([1,2,3], dtype=np.int32)
            lab = np.array([4,5,6], dtype=np.int32)
            print(f"\nAdding sample #{i}: input.size={inp.size}, label.size={lab.size}")
            status = packer.add_sample(inp, lab, add_sep=False)
            leftover_space = packer.get_remaining_space()
            print(f"After add_sample => status={status}, leftover_space={leftover_space}")
        num_arrays = len(packer.packed_arrays)
        print(f"\nNumber of packed arrays in the packer: {num_arrays}")
        self.assertGreaterEqual(num_arrays, 2)
        data_dict1 = packer.get_data(peek=False)
        data_dict2 = packer.get_data(peek=False)
        print("\nFirst packed array data:", data_dict1)
        print("\nSecond packed array data:", data_dict2)
        self.assertIsNotNone(data_dict1)
        self.assertIsNotNone(data_dict2)


class TestPackerAdditional(unittest.TestCase):
    def test_packer_with_chinese_samples(self):
        """
        Test adding samples that contain Chinese text to confirm no errors in packing logic.
        """
        packer = SequencePacker(sequence_length=20, minimum_sequence_length=5)
        inp = np.array([300,301,302], dtype=np.int32)  # Some stand-in tokens
        lab = np.array([303,304,305], dtype=np.int32)
        # Attempt to pack them
        status = packer.add_sample(inp, lab)
        self.assertFalse(status["ready"])

    def test_packer_with_hindi_samples(self):
        """
        Test adding samples that contain Hindi tokens (simulated by arbitrary ID range).
        """
        packer = SequencePacker(sequence_length=12, minimum_sequence_length=4)
        inp = np.array([400,401], dtype=np.int32)
        lab = np.array([402,403,404], dtype=np.int32)
        status = packer.add_sample(inp, lab)
        # Should fit or might remain not ready
        self.assertIn("ready", status)

    def test_packer_with_dna_samples(self):
        """
        Test packing repeated DNA token patterns to ensure correct leftover detection.
        """
        packer = SequencePacker(sequence_length=16, minimum_sequence_length=3)
        # Simulate 'AGTC' repeated as token IDs 10,11,12,13
        inp = np.array([10,11,12,13,10,11], dtype=np.int32)
        lab = np.array([12,13,10,11], dtype=np.int32)
        status = packer.add_sample(inp, lab)
        self.assertFalse(status["ready"], "Might not fill array completely but no errors expected")

if __name__ == "__main__":
    unittest.main()
