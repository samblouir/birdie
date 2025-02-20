"""
Utility functions for objectives.

Example: slicing text into used and leftover portions based on remaining_space.
"""

import numpy as np
from typing import Any, Dict
import hashlib


def slice_text_by_remaining_space(
	text: str, tokenizer: Any, remaining_space: int
) -> Dict[str, Any]:
	"""
	Splits the input text into 'used' portion and leftover, according to remaining_space.

	Args:
		text: The input text string.
		tokenizer: A tokenizer instance (with encode/decode).
		remaining_space: How many tokens we can use. If negative, use all.

	Returns:
		A dict containing:
		  - used_text
		  - used_tokens
		  - unused_text
		  - unused_tokens
	"""
	all_tokens = tokenizer.encode(text)
	token_count = len(all_tokens)

	# remaining_space -= 16

	if remaining_space <= 0:
		return {
			# "used_text": text,
			# "used_tokens": np.array(all_tokens, dtype=np.int32),
			# "unused_text": "",
			# "unused_tokens": np.array([], dtype=np.int32),
			"used_text": "",
			"used_tokens": np.array([], dtype=np.int32),
			"unused_text": text,
			"unused_tokens": np.array(all_tokens, dtype=np.int32),
		}

	if token_count <= remaining_space:
		return {
			"used_text": text,
			"used_tokens": np.array(all_tokens, dtype=np.int32),
			"unused_text": "",
			"unused_tokens": np.array([], dtype=np.int32),
		}
	else:
		used_tokens = all_tokens[:remaining_space]
		leftover_tokens = all_tokens[remaining_space:]
		used_text = tokenizer.decode(used_tokens)
		leftover_text = tokenizer.decode(leftover_tokens)
		return {
			"used_text": used_text,
			"used_tokens": np.array(used_tokens, dtype=np.int32),
			"unused_text": leftover_text,
			"unused_tokens": np.array(leftover_tokens, dtype=np.int32),
		}

	
def sha_hash(x):
	x = str(x)
	return hashlib.sha256(x.encode()).hexdigest()

if __name__ == "__main__":

	class MockTokenizer:
		def encode(self, txt: str) -> list:
			return [ord(c) for c in txt]

		def decode(self, t_ids: list) -> str:
			return "".join([chr(x) for x in t_ids])

	tok = MockTokenizer()
	txt = "Hello World"

	result_neg = slice_text_by_remaining_space(txt, tok, -1)
	print("remaining_space=-1 =>", result_neg)

	result_partial = slice_text_by_remaining_space(txt, tok, 5)
	print("remaining_space=5 =>", result_partial)
