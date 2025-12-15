"""
Top-level package exports.

Birdie depends on PyTorch; on some systems a broken/mismatched CUDA install can
make `import torch` fail (e.g. missing shared libs). We keep `birdie_rl`
importable in that case so users can still work with the text/objective/pipeline
utilities, and raise a helpful error only when `Birdie` is accessed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Birdie"]

def __getattr__(name: str) -> Any:  # pragma: no cover
	if name != "Birdie":
		raise AttributeError(name)
	try:
		from .birdie_reward_model import Birdie
	except Exception as exc:  # noqa: BLE001 - want to catch CUDA/torch import failures too
		raise ImportError(
			"Failed to import `Birdie` (PyTorch/CUDA runtime issue). "
			"If you recently changed GPUs/drivers, reinstall a matching PyTorch build, "
			"or install a CPU-only torch wheel for data-pipeline debugging."
		) from exc
	return Birdie
