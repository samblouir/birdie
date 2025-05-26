from datasets import load_dataset, IterableDataset
import numpy as np
import torch
import os
import time
import traceback
import sys


def reward_fn(
    action_taken=None,
    old_loss=None,
    new_loss=None,
    old_step_idx=None,
    new_step_idx=None,
):
    delta_loss = new_loss - old_loss
    rv = delta_loss / (old_loss + 1e-8)
    n = (new_loss * old_loss).sqrt() * rv.pow(3) * torch.e
    reward = -100 * torch.tanh(n) * torch.e
    reward = torch.where(
        torch.isnan(reward),
        torch.tensor(0.0, dtype=reward.dtype, device=reward.device),
        reward,
    )
    reward = torch.clamp(reward, -1.0, 1.0)
    return reward


def data_generator(split, worker_id, num_workers, rng_seed=0):
    pid = os.getpid()
    # print(f"[data_generator (utils.py) Worker {worker_id} PID {pid}] Initializing with HuggingFace TinyStories for split: {split}.", flush=True)

    try:
        ds_split = load_dataset(
            "roneneldan/TinyStories", split=split, trust_remote_code=True
        )
        ds_split = ds_split.shard(num_shards=num_workers, index=worker_id)

        list_idx = 0
        while True:
            item_to_yield = ds_split[list_idx]
            yield item_to_yield
            list_idx = (list_idx + 1) % len(ds_split)

    except Exception as e:
        print(
            f"[data_generator (utils.py) Worker {worker_id} PID {pid}] CRITICAL ERROR in data_generator for split '{split}': {e}",
            flush=True,
        )  # Keep error
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        while True:
            yield {"text": f"ERROR_IN_DATA_GENERATOR_W{worker_id}_S{split}"}
            time.sleep(1)
