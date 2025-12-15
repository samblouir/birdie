"""
Exports the Birdie class so it can be imported conveniently from this package.

USAGE:
	from birdie_rl.birdie_reward_model import Birdie
	# Construct Birdie with your config
	birdie = Birdie(config)
	
	for step_idx in range(steps):
		
		# If it's time to evaluate, measure the validation losses
		if birdie.time_for_eval(step_idx):
			for (objective_name, batch) in birdie.measure_validation_losses():
				# Calculate the loss
				loss = model(**batch)
				birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step_idx)
				
		# Get the next training sample
		batch = birdie.get_next_training_sample()
		model = train_step(model, batch)
"""

from __future__ import annotations

from typing import Any

__all__ = ["Birdie"]

def __getattr__(name: str) -> Any:  # pragma: no cover
	if name != "Birdie":
		raise AttributeError(name)
	try:
		from .birdie import Birdie
	except Exception as exc:  # noqa: BLE001 - want to catch torch/datasets import failures too
		raise ImportError(
			"Failed to import `Birdie` (optional dependencies/runtime issue). "
			"This does not affect importing submodules like `agent_bird` for synthetic runs."
		) from exc
	return Birdie
