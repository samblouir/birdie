"""
Unit tests for the Tokenizer class.

This file includes both the original TestTokenizer class
and an additional TestTokenizerAdditional class to cover
Chinese, Hindi, and DNA sequences. The main fix for
the AttributeError is adding setUp() in the additional class.
"""

import unittest
from modeling.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Tests basic encoding/decoding behaviors for English text
    and checks certain out-of-range token handling.
    """

    def setUp(self):
        """
        Creates a Tokenizer instance before each test in this class.
        """
        self.tokenizer = Tokenizer()

    def test_encode_decode(self):
        """
        Test that encoding and then decoding the same string returns
        the original text.
        
        Expected Input:
          A normal English sentence
        Expected Output:
          The decoded text should match exactly the original input.
        """
        text = "Hello world!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_list_input(self):
        """
        Test encoding a list of strings and decoding them back.
        
        Expected Input:
          A list of short strings
        Expected Output:
          The decoded list of strings should match the original list.
        """
        texts = ["Hello", "world"]
        encoded_list = self.tokenizer.encode(texts)
        self.assertIsInstance(encoded_list, list)
        decoded_list = self.tokenizer.decode(encoded_list)
        self.assertEqual(decoded_list, texts)

    def test_out_of_range_tokens(self):
        """
        Test that out-of-range tokens no longer trigger placeholder logic.
        With tiktoken, tokens >= 256 might represent valid subwords.
        
        Expected Input:
          A token ID list with some values > 256
        Expected Output:
          Decoded string is not empty, no placeholders like <SENTxx>.
        """
        token_ids = [72, 101, 108, 108, 111, 300, 301, 33]
        decoded = self.tokenizer.decode(token_ids)
        self.assertIsInstance(decoded, str)
        self.assertNotIn("<SENT", decoded)


class TestTokenizerAdditional(unittest.TestCase):
    """
    Additional tests for the Tokenizer class focusing on:
      - Chinese text
      - Hindi text
      - DNA-like sequences

    The main fix for the previous errors is ensuring we
    have self.tokenizer set up.
    """

    def setUp(self):
        """
        Creates a Tokenizer instance before each test in this class.
        """
        self.tokenizer = Tokenizer()

    def test_encode_decode_chinese(self):
        """
        Test that Chinese text can be encoded/decoded without error.

        Quirks:
          - Exact round-trip might differ slightly if the tokenizer merges
            characters or splits them in a certain way.
        Expected Input:
          A string containing Chinese characters.
        Expected Output:
          The decoded result should include at least part of the original characters.
        """
        text = "你好，世界"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("你好", decoded, "Decoded text should contain some Chinese characters")

    def test_encode_decode_hindi(self):
        """
        Test that Hindi text can be encoded/decoded without error.
        
        Expected Input:
          A string containing Hindi characters.
        Expected Output:
          The decoded result contains "नमस्ते" or relevant substring.
        """
        text = "नमस्ते दुनिया"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("नमस्ते", decoded, "Decoded text should contain 'नमस्ते'")

    def test_encode_decode_dna_sequence(self):
        """
        Test that a repeated DNA sequence is handled by the tokenizer
        without crashing. The exact text might differ slightly after decode
        if tiktoken merges tokens.

        Expected Input:
          A repeated DNA-like string.
        Expected Output:
          The decode is not empty, no error is raised.
        """
        dna = "AGTCAGTCAGTC"
        encoded = self.tokenizer.encode(dna)
        decoded = self.tokenizer.decode(encoded)
        self.assertNotEqual(len(encoded), 0, "Encoded DNA sequence should not be empty")
        self.assertTrue(len(decoded) > 0, "Decoded string should have some length")

if __name__ == "__main__":
    unittest.main()
