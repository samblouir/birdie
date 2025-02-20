import pytest
from birdie_rl.load_objective import load_objective
from modeling.tokenizer import Tokenizer

def test_load_objective_valid():
    """Verify we can load a known objective with minimal overrides."""
    obj = load_objective("copying", {"paradigm": "[COPY]"})
    assert obj.config.paradigm == "[COPY]"
    assert obj.config.objective == ""  # default
    assert obj.tokenizer is not None, "Tokenizer should be auto-created."

def test_load_objective_with_tokenizer():
    """Check that we can supply our own tokenizer."""
    tok = Tokenizer()
    obj = load_objective("autoencoding", {"tokenizer": tok, "corruption_rate": 0.2})
    assert obj.tokenizer == tok, "User-supplied tokenizer must be set."
    assert obj.config.corruption_rate == 0.2

def test_load_objective_unknown():
    """Ensure we raise ValueError for unknown objectives."""
    with pytest.raises(ValueError) as excinfo:
        load_objective("unknown_objective")
    assert "Unknown objective" in str(excinfo.value)

def test_load_objective_chinese_override():
    """
    Check that we can load an objective (infilling) with a Chinese text override.
    """
    tok = Tokenizer()
    obj = load_objective("infilling", {
        "tokenizer": tok,
        "corruption_rate": 0.5,
        "infilling_prefix": "[mask_中文"
    })
    assert obj.tokenizer == tok
    assert obj.config.infilling_prefix.startswith("[mask_中文")

def test_load_objective_hindi_override():
    """
    Load an objective with a Hindi text-based override to ensure acceptance of non-Latin characters.
    """
    tok = Tokenizer()
    obj = load_objective("autoencoding", {
        "tokenizer": tok,
        "mask_prefix": "[मास्क_"
    })
    assert obj.config.mask_prefix == "[मास्क_"

def test_load_objective_dna_override():
    """
    Load an objective with a custom prefix referencing 'DNA' to ensure no issues with unusual tokens.
    """
    tok = Tokenizer()
    obj = load_objective("prefix_language_modeling", {
        "tokenizer": tok,
        "objective": "DNA-Testing"
    })
    assert obj.config.objective == "DNA-Testing"
