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

# Import the Birdie class from birdie.py so it is accessible at the package level.
from .birdie import Birdie

# Restricts __all__ variable to simplify a code running "from birdie_reward_model import *"
__all__ = ["Birdie"]
