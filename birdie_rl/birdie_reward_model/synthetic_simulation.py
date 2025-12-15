"""
synthetic_simulation.py

Run AgentBird on fully-synthetic objective losses/actions to:
- sanity check the reward model path (import + forward + update)
- benchmark throughput (updates/sec, samples/sec)
- validate reproducibility under seeding

This module intentionally avoids importing the full Birdie dataloader/pipeline.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from birdie_rl.birdie_reward_model.agent_bird import AgentBird


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	if deterministic:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		try:
			torch.use_deterministic_algorithms(True)
		except Exception:
			pass


def _to_numpy_f32(x) -> np.ndarray:
	if isinstance(x, np.ndarray):
		return x.astype(np.float32, copy=False)
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy().astype(np.float32, copy=False)
	return np.asarray(x, dtype=np.float32)


@dataclass(frozen=True)
class SyntheticEnvConfig:
	num_objectives: int
	min_loss: float = 1e-6
	init_loss: float = 1.0
	noise_std: float = 0.0
	min_improvement_rate: float = 0.0005
	max_improvement_rate: float = 0.01


def synthetic_env_step(
	*,
	old_loss: np.ndarray,
	action: np.ndarray,
	base_rates: np.ndarray,
	rng: np.random.Generator,
	cfg: SyntheticEnvConfig,
) -> np.ndarray:
	"""
Simple synthetic dynamics:
	new_loss = old_loss * (1 - base_rates * action) + N(0, noise_std)
	"""
	improvement = base_rates * action
	if cfg.noise_std > 0:
		noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=old_loss.shape).astype(np.float32)
	else:
		noise = 0.0

	new_loss = old_loss * (1.0 - improvement) + noise
	new_loss = np.maximum(new_loss, cfg.min_loss)
	return new_loss.astype(np.float32, copy=False)


def run_synthetic_agent(
	*,
	seed: int,
	steps: int,
	device: str,
	agent_kwargs: dict,
	env_cfg: SyntheticEnvConfig,
	training: bool,
	log_every: int,
) -> dict:
	seed_everything(seed, deterministic=bool(agent_kwargs.get("deterministic", False)))

	rng = np.random.default_rng(seed + 1)
	base_rates = rng.uniform(env_cfg.min_improvement_rate, env_cfg.max_improvement_rate, size=env_cfg.num_objectives).astype(np.float32)

	agent = AgentBird(
		device=device,
		num_objectives=env_cfg.num_objectives,
		reward_signal_dims=env_cfg.num_objectives,
		np_rng_seed=seed,
		**agent_kwargs,
	)

	current_loss = np.full(env_cfg.num_objectives, env_cfg.init_loss, dtype=np.float32)
	action = np.full(env_cfg.num_objectives, 1.0 / env_cfg.num_objectives, dtype=np.float32)

	actions = []
	update_s = 0.0
	sample_s = 0.0

	t0_total = time.perf_counter()
	for step in range(steps):
		agent.step_counter = step
		old_loss = current_loss
		new_loss = synthetic_env_step(old_loss=old_loss, action=action, base_rates=base_rates, rng=rng, cfg=env_cfg)

		t0 = time.perf_counter()
		_ = agent.update(new_loss_vector=new_loss, old_loss_vector=old_loss, action_taken=action)
		t1 = time.perf_counter()
		out = agent.sample(new_loss, training=training)
		t2 = time.perf_counter()

		update_s += (t1 - t0)
		sample_s += (t2 - t1)

		action = _to_numpy_f32(out["action"])
		current_loss = new_loss
		actions.append(action)

		if log_every > 0 and (step == 0 or (step + 1) % log_every == 0):
			print(
				f"[step {step + 1:>6}] explored={bool(out['explored'])} "
				f"loss_mean={float(current_loss.mean()):.6f} "
				f"update_ms={(t1 - t0) * 1e3:.2f} sample_ms={(t2 - t1) * 1e3:.2f}"
			)

	t_total = time.perf_counter() - t0_total
	actions_arr = np.stack(actions, axis=0) if actions else np.zeros((0, env_cfg.num_objectives), dtype=np.float32)

	return {
		"steps": int(steps),
		"device": str(device),
		"num_objectives": int(env_cfg.num_objectives),
		"actions": actions_arr,
		"throughput_steps_per_sec": (float(steps) / t_total) if t_total > 0 else 0.0,
		"throughput_updates_per_sec": (float(steps) / update_s) if update_s > 0 else 0.0,
		"throughput_samples_per_sec": (float(steps) / sample_s) if sample_s > 0 else 0.0,
		"timing_total_s": float(t_total),
		"timing_update_s": float(update_s),
		"timing_sample_s": float(sample_s),
	}


def _parse_hidden_dims(arg: str) -> list[int]:
	if not arg:
		return []
	return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main() -> None:
	parser = argparse.ArgumentParser(description="Synthetic AgentBird simulation + throughput + reproducibility")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--steps", type=int, default=64)
	parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
	parser.add_argument("--num-objectives", type=int, default=8)
	parser.add_argument("--training", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--log-every", type=int, default=0)

	# Agent knobs
	parser.add_argument("--grok-iterations", type=int, default=1)
	parser.add_argument("--agent-num-actions-to-try", type=int, default=256)
	parser.add_argument("--hidden-dims", type=str, default="64,64")
	parser.add_argument("--model-max-seq-len", type=int, default=64)
	parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=True)

	# Synthetic env knobs
	parser.add_argument("--init-loss", type=float, default=1.0)
	parser.add_argument("--min-loss", type=float, default=1e-6)
	parser.add_argument("--noise-std", type=float, default=0.0)
	parser.add_argument("--min-improvement-rate", type=float, default=0.0005)
	parser.add_argument("--max-improvement-rate", type=float, default=0.01)

	args = parser.parse_args()

	env_cfg = SyntheticEnvConfig(
		num_objectives=int(args.num_objectives),
		min_loss=float(args.min_loss),
		init_loss=float(args.init_loss),
		noise_std=float(args.noise_std),
		min_improvement_rate=float(args.min_improvement_rate),
		max_improvement_rate=float(args.max_improvement_rate),
	)

	agent_kwargs = dict(
		grok_iterations=int(args.grok_iterations),
		agent_num_actions_to_try=int(args.agent_num_actions_to_try),
		hidden_dims=_parse_hidden_dims(args.hidden_dims),
		model_max_seq_len=int(args.model_max_seq_len),
		use_torch_compile=bool(args.compile),
		disable_tqdm=bool(args.disable_tqdm),
		deterministic=bool(args.deterministic),
	)

	res = run_synthetic_agent(
		seed=int(args.seed),
		steps=int(args.steps),
		device=str(args.device),
		agent_kwargs=agent_kwargs,
		env_cfg=env_cfg,
		training=bool(args.training),
		log_every=int(args.log_every),
	)

	print(
		"throughput: "
		f"{res['throughput_steps_per_sec']:.2f} steps/s "
		f"({res['throughput_updates_per_sec']:.2f} updates/s, {res['throughput_samples_per_sec']:.2f} samples/s)"
	)


if __name__ == "__main__":
	main()

