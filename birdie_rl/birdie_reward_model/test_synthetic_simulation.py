import unittest

import numpy as np

from birdie_rl.birdie_reward_model.synthetic_simulation import SyntheticEnvConfig, run_synthetic_agent


class TestSyntheticSimulation(unittest.TestCase):
	def test_reproducible_actions_per_seed(self):
		env_cfg = SyntheticEnvConfig(
			num_objectives=4,
			init_loss=1.0,
			min_loss=1e-6,
			noise_std=0.0,
			min_improvement_rate=0.001,
			max_improvement_rate=0.01,
		)
		agent_kwargs = dict(
			grok_iterations=1,
			agent_num_actions_to_try=32,
			hidden_dims=[32, 32],
			model_max_seq_len=32,
			use_torch_compile=False,
			disable_tqdm=True,
			deterministic=True,
		)

		res1 = run_synthetic_agent(
			seed=123,
			steps=8,
			device="cpu",
			agent_kwargs=agent_kwargs,
			env_cfg=env_cfg,
			training=True,
			log_every=0,
		)
		res2 = run_synthetic_agent(
			seed=123,
			steps=8,
			device="cpu",
			agent_kwargs=agent_kwargs,
			env_cfg=env_cfg,
			training=True,
			log_every=0,
		)

		self.assertTrue(np.array_equal(res1["actions"], res2["actions"]))

