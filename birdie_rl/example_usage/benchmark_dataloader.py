"""
This is based off of example.py and adds profiling for the main process
`pipeline/worker.py` can have its profiling enabled by setting `profile:bool=True` in its `run()`. function
"""

import os

import time

import cProfile
import pstats
import io

from birdie_rl import Birdie
from datasets import load_dataset
from tqdm import tqdm
import accelerate
import numpy as np
import tiktoken
import torch
from ul2_config import dummy_config, ul2_config, ul2_decoder_config
import utils


# --- Configuration ---
MAX_TIME_FOR_BATCH_FETCH_SECONDS = 300.0
PROFILING_NUM_STEPS = 200


accelerator = accelerate.Accelerator()
config = {
    #
    "reward_fn": utils.reward_fn,
    "ds": utils.data_generator,
    "objectives": ul2_config,
    "tokenizer": tiktoken.get_encoding("o200k_base"),
    #
    "batch_size": 8,
    "sequence_length": 2048,
    "num_workers": 1,
    "steps_between_evaluations": 32,
    "num_steps": PROFILING_NUM_STEPS,
    "accelerator": accelerator,
    #
    "num_validation_batches": 2,
}

# --- Birdie Initialization ---


print(
    "Initializing Birdie (this may take some time if num_validation_batches is high)..."
)
birdie = Birdie(config)
print("Birdie initialized.")


initial_action = birdie.get_current_action()
print(f"Initial Birdie action: {initial_action}")

# --- Setup for Training Loop ---
seeded_np_rng = np.random.default_rng(0)
progress_bar = tqdm(total=PROFILING_NUM_STEPS, desc="Training (Profiling)", unit="step")
total_tokens_processed = 0

# --- Profiling Setup ---
profiler = cProfile.Profile()


# --- Main Training Function (to be profiled) ---
def main_training_loop():
    global total_tokens_processed
    global training_start_time

    print(f"\nStarting training for {PROFILING_NUM_STEPS} steps (for profiling)...")
    for step_idx in range(PROFILING_NUM_STEPS):
        # --- Hang Detection & Batch Fetching ---
        batch_fetch_start_time = time.time()
        train_batch = birdie.get_next_training_sample()
        batch_fetch_duration = time.time() - batch_fetch_start_time
        if batch_fetch_duration > MAX_TIME_FOR_BATCH_FETCH_SECONDS:
            progress_bar.close()
            raise TimeoutError(
                f"Fetching training batch for step {step_idx} took {batch_fetch_duration:.2f} seconds, "
                f"exceeding the timeout of {MAX_TIME_FOR_BATCH_FETCH_SECONDS:.2f} seconds. "
                "Potential hang detected in birdie.get_next_training_sample()."
            )

        # --- Token & Throughput Calculation ---
        if "input_ids" in train_batch and hasattr(train_batch["input_ids"], "numel"):
            num_tokens_in_batch = train_batch["input_ids"].numel()
            total_tokens_processed += num_tokens_in_batch
        else:
            print(
                "Warning: 'input_ids' not found or not a tensor, cannot calculate token count for this batch."
            )
            num_tokens_in_batch = 0

        elapsed_time_training = time.time() - training_start_time
        tokens_per_second = (
            total_tokens_processed / elapsed_time_training
            if elapsed_time_training > 0
            else 0
        )

        progress_bar.update(1)
        progress_bar.set_postfix(tokens_p_sec=f"{tokens_per_second:.2f}", refresh=True)

        # --- Optional: Display Batch Statistics ---
        show_batch_stats = False
        if (
            show_batch_stats
            and "input_ids" in train_batch
            and "label_ids" in train_batch
        ):
            used_iids = torch.where(train_batch["input_ids"] == 0, 0, 1).sum().item()
            wasted_iids = torch.where(train_batch["input_ids"] == 0, 1, 0).sum().item()
            max_input_ids = (train_batch["input_ids"]).max().item()
            max_label_ids = (train_batch["label_ids"]).max().item()
            packer_efficiency = (
                1 - (wasted_iids / (wasted_iids + used_iids))
                if (wasted_iids + used_iids) > 0
                else 0
            )
            _results = [
                f"Step {step_idx}:",
                f"  max_input_ids: {max_input_ids}",
                f"  max_label_ids: {max_label_ids}",
                f"  wasted_iids: {wasted_iids}",
                f"  used_iids: {used_iids}",
                f"  packer_efficiency: {packer_efficiency:.2%}",
            ]
            tqdm.write("\n".join(_results))

        # --- Evaluation Phase ---
        if birdie.time_for_eval(step_idx):
            tqdm.write(f"\n--- Step {step_idx}: Running Evaluation ---")
            for objective_name, batch in birdie.measure_validation_losses():
                simulated_loss = seeded_np_rng.random()
                birdie.log_validation_loss(
                    key=objective_name, loss=simulated_loss, step_idx=step_idx
                )
                tqdm.write(
                    f"  Objective '{objective_name}': Simulated validation loss = {simulated_loss:.4f}"
                )
            (status_dict, status_str) = birdie.get_verbose_action()
            tqdm.write("Current Birdie Action (Status):")
            tqdm.write(status_str)
            tqdm.write("--- Evaluation Complete ---")


# --- Run Training with Profiling ---
try:
    print(f"  Awaiting first batch before starting profiling...", flush=True)
    first_batch_start_time = time.time()
    train_batch = birdie.get_next_training_sample()
    first_batch_duration = time.time() - first_batch_start_time
    print(f"  First batch fetched in {first_batch_duration:.2f} seconds.", flush=True)
    training_start_time = time.time()
    print("Starting profiled training loop...")
    profiler.enable()
    main_training_loop()  # Call the main loop
    profiler.disable()
    print("Profiled training loop finished.")


except KeyboardInterrupt:
    print("\nTraining interrupted by user. Disabling profiler.")
    profiler.disable()


finally:
    progress_bar.close()

    # --- Print Profiling Stats ---
    print("\n--- Profiling Results ---")
    s = io.StringIO()

    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats(30)
    print(s.getvalue())

    # --- Cleanup and Exit ---
    print("\n" * 3, end="")
    print("All training steps completed (or interrupted).")
    total_training_duration = time.time() - training_start_time
    print(f"Total training time: {total_training_duration:.2f} seconds.")
    print(f"Total tokens processed: {total_tokens_processed}")
    if total_training_duration > 0:
        print(
            f"Average tokens per second: {total_tokens_processed / total_training_duration:.2f}"
        )

    print("Closing Birdie and freeing associated resources...")
    birdie.close()
    print("Birdie closed.")

    print("Script finished. Exiting.")
