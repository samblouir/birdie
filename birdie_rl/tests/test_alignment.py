"""
test_alignment.py

A simple test suite that checks whether each objective's output
has 'input_ids' and 'label_ids' that are aligned enough to be valid.
We ensure that:
 - Both arrays are non-empty
 - They do not differ in length by an extreme factor
"""

import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from birdie_rl.load_objective import load_objective

@pytest.mark.parametrize("obj_name, overrides", [
    ("autoencoding",           {"corruption_rate": 0.5}),
    ("copying",                {"paradigm": "[TEST_COPY]"}),
    ("deshuffling",            {"percentage_of_tokens_to_shuffle": 0.5}),
    ("infilling",              {"corruption_rate": 0.5}),
    ("next_token_prediction",  {"paradigm": "[NTP]"}),
    ("prefix_language_modeling", {"prefix_fraction": 0.5}),
    ("selective_copying",      {"formatting_type": "query_context"}),
])
def test_input_label_alignment(obj_name, overrides):
    """
    For each known objective, load it with some overrides,
    pass in a sample text, and verify that:
     1) status == "ok"
     2) input_ids and label_ids are not empty
     3) their lengths do not differ by a large ratio (like 1:3 or more)
    """
    # Create a Tokenizer
    tokenizer = Tokenizer()
    
    # Load the objective with given overrides
    objective = load_objective(obj_name, overrides)
    objective.set_tokenizer(tokenizer)
    
    # Define a sample text
    sample_text = f"Alignment check for {obj_name} with overrides {overrides}."
    
    # Call the objective
    result = objective(sample_text)
    
    # Check the status
    assert result["status"] == "ok", f"Expected status=ok, got {result['status']} for objective={obj_name}"
    
    # Get lengths of input_ids and label_ids
    inp_len = len(result["input_ids"])
    lbl_len = len(result["label_ids"])
    
    # Both must be > 0
    assert inp_len > 0, f"Objective {obj_name} => input_ids is empty!"
    assert lbl_len > 0, f"Objective {obj_name} => label_ids is empty!"
    
    # Check ratio isn't extreme. Some objectives produce slightly different sizes,
    # but typically not extremely large differences.
    ratio = max(inp_len / lbl_len, lbl_len / inp_len)
    max_ratio = 3.0  # Arbitrary threshold
    assert ratio < max_ratio, (
        f"For {obj_name}, input_ids length={inp_len} vs. label_ids length={lbl_len} => ratio {ratio:.2f} too high"
    )
