# tests/test_utils.py
"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from birdie_rl.modeling.tokenizer import Tokenizer
tokenizer = Tokenizer()

def debug_print_array_dict(data_dict, heading="Debug output"):
	"""
	Prints a dictionary of arrays in a column-aligned table, including an 'idx' column.
	Assumes all arrays have the same length.
	"""
	arrays = {k: np.array(v) for k, v in data_dict.items()}
	if not arrays:
		print(f"{heading}\n\n(No arrays to display)")
		return
	
	if "input_ids" in data_dict:
		input_ids = data_dict["input_ids"]
		decoded_input_ids = [tokenizer.decode([inp]).replace("\n", "\\n") for inp in input_ids]
		arrays["decoded_input_ids"] = decoded_input_ids
	
	if "label_ids" in data_dict:
		label_ids = data_dict["label_ids"]
		if len(label_ids) > 0:
			label_ids = [int(x) if x != -100 else 222 for x in label_ids]
			decoded_label_ids = [tokenizer.decode([lbl]).replace("\n", "\\n") for lbl in label_ids]
			arrays["decoded_label_ids"] = decoded_label_ids

	# tp = []
	# columns = sorted(arrays.keys())
	# n = len(arrays[columns[0]])
	# columns = ["idx"] + columns
	# widths = {}
	# for col in columns:
	# 	if col == "idx":
	# 		max_val_len = len(str(n-1))
	# 		widths[col] = max(max_val_len, len(col))
	# 	else:
	# 		col_header_len = len(col)
	# 		col_values = arrays[col]
	# 		max_val_len = max(len(str(x)) for x in col_values)
	# 		widths[col] = max(col_header_len, max_val_len)
	# tp.append(f"{heading}\n")
	# header_parts = []
	# for col in columns:
	# 	header_parts.append(f"{col:>{widths[col]}}")
	# header_str = "  ".join(header_parts)
	# tp.append(header_str)
	# for i in range(n):
	# 	row_parts = []
	# 	for col in columns:
	# 		if col == "idx":
	# 			row_parts.append(f"{i:>{widths['idx']}}")
	# 		else:
	# 			val = arrays[col][i]
	# 			row_parts.append(f"{val:>{widths[col]}}")
	# 	row_str = "  ".join(row_parts)
	# 	tp.append(row_str)
	# tp.append('\n')
	# print(flush=True, end='')
	# print('\n'.join(tp), flush=True)
	

	if "input_ids" in arrays: print(f"  decoded_input_ids: \"\"\"{''.join(decoded_input_ids)}\"\"\"")
	if "label_ids" in arrays and (len(label_ids) > 0): print(f"  decoded_label_ids: \"\"\"{''.join(decoded_label_ids)}\"\"\"")


# def debug_print_array_dict(data_dict, heading="Debug output"):
# 	"""
# 	Prints a dictionary of arrays in a column-aligned table, including an 'idx' column.
# 	Assumes all arrays have the same length.
# 	"""

# 	tp = []
# 	arrays = {k: np.array(v) for k, v in data_dict.items()}

	
# 	if "input_ids" in data_dict:
# 		input_ids = data_dict["input_ids"]
# 		decoded_input_ids = [tokenizer.decode([inp]) for inp in input_ids]
# 		arrays["decoded_input_ids"] = decoded_input_ids
	
# 	if "label_ids" in data_dict:
# 		label_ids = data_dict["label_ids"]
# 		if len(label_ids) > 0:
# 			label_ids = [int(x) if x != -100 else 12 for x in label_ids]
# 			decoded_label_ids = [tokenizer.decode([lbl]) for lbl in label_ids]
# 			arrays["decoded_label_ids"] = decoded_label_ids

# 	if not arrays:
# 		print(f"{heading}\n\n(No arrays to display)")
# 		return
	
# 	if "input_ids" in data_dict:
# 		input_ids = data_dict["input_ids"]
# 		decoded_input_ids = tokenizer.decode(input_ids)
# 		arrays["decoded_input_ids"] = decoded_input_ids
	
# 	if "label_ids" in data_dict:
# 		label_ids = data_dict["label_ids"]
# 		if len(label_ids) > 0:
# 			label_ids = [int(x) if x != -100 else 12 for x in label_ids]
# 			decoded_label_ids = tokenizer.decode(label_ids)
# 			arrays["decoded_label_ids"] = decoded_label_ids

# 	columns = sorted(arrays.keys())
# 	n = len(arrays[columns[0]])
# 	columns = ["idx"] + columns
# 	widths = {}
# 	for col in columns:
# 		if col == "idx":
# 			max_val_len = len(str(n-1))
# 			widths[col] = max(max_val_len, len(col))
# 		else:
# 			col_header_len = len(col)
# 			col_values = arrays[col]
# 			max_val_len = max(len(str(x)) for x in col_values)
# 			widths[col] = max(col_header_len, max_val_len)
# 	tp.append(f"{heading}\n")
# 	header_parts = []
# 	for col in columns:
# 		header_parts.append(f"{col:>{widths[col]}}")
# 	header_str = "  ".join(header_parts)
# 	tp.append(header_str)
# 	for i in range(n):
# 		row_parts = []
# 		for col in columns:
# 			if col == "idx":
# 				row_parts.append(f"{i:>{widths['idx']}}")
# 			else:
# 				val = arrays[col][i]
# 				row_parts.append(f"{val:>{widths[col]}}")
# 		row_str = "  ".join(row_parts)
# 		tp.append(row_str)
# 	tp.append('\n')
	

# 	if "input_ids" in arrays: tp.append(f"  decoded_input_ids: \"\"\"{decoded_input_ids}\"\"\"")
# 	if "label_ids" in arrays and (len(label_ids) > 0): tp.append(f"  decoded_label_ids: \"\"\"{decoded_label_ids}\"\"\"")

# 	print('\n'.join(tp), flush=True)



def test_debug_print_array_dict_empty(capfd):
	"""
	Test debug_print_array_dict with an empty dictionary.
	We use capfd (pytest fixture) to capture printed output.
	"""
	debug_print_array_dict({})
	out, _ = capfd.readouterr()
	assert "(No arrays to display)" in out

def test_debug_print_array_dict_nonempty(capfd):
	"""
	Test debug_print_array_dict with a simple data dictionary.
	"""
	data = {
		"input_ids": np.array([1,2,3]),
		"label_ids": np.array([-100,4,5])
	}
	debug_print_array_dict(data, heading="Test heading")
	out, _ = capfd.readouterr()
	assert "Test heading" in out
	# We expect columns like idx, input_ids, label_ids
	assert "idx" in out
	assert "input_ids" in out
	assert "label_ids" in out
