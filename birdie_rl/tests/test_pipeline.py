"""
test_pipeline.py

End-to-end tests for each objective via MainController + Worker. 
We decode the final input_ids/label_ids, print the results, and also
call debug_print_array_dict(...) to display numeric arrays in detail.

Objectives covered:
  - Copying
  - Next Token Prediction
  - Autoencoding
  - Infilling
  - Deshuffling
  - Prefix Language Modeling
  - Selective Copying

Features for readability:
  - ShortWorker (subclass of Worker) that only generates a few samples (reduces leftover).
  - single_sample_gen(...) yields the same text repeatedly (for repeatable tests).
  - safe_decode(...) ensures negative token IDs won't crash Tiktoken.
  - shorten_text(...) truncates very long decoded strings in logs.
  - debug_print_array_dict(...) from tests/test_utils prints numeric arrays nicely.
"""

import queue
import threading
import numpy as np
from typing import List

# If "test_utils" is in the same tests/ folder, you can do:
from tests.test_utils import debug_print_array_dict

from datasets import load_dataset
from pipeline.main_controller import MainController
from pipeline.worker import Worker
from modeling.tokenizer import Tokenizer

########################################
# HELPER: SINGLE-SAMPLE GENERATOR
########################################

def single_sample_gen(sample_text: str):
    """
    Returns a generator that always yields the same 'sample_text'.
    This ensures we produce a known, repeatable input for pipeline tests.
    """
    while True:
        yield sample_text

########################################
# HELPER: SAFE DECODE
########################################

def safe_decode(tokenizer: Tokenizer, token_ids_array) -> str:
    """
    Decodes a list/array of token IDs, filtering out negative IDs (like -100)
    by converting them to 999 (or removing them). This prevents Tiktoken errors.

    Args:
        tokenizer: The Tokenizer instance (Tiktoken-based).
        token_ids_array: A list or NumPy array of token IDs.

    Returns:
        A decoded string, with out-of-range IDs replaced or removed.
    """
    if isinstance(token_ids_array, np.ndarray):
        token_ids = token_ids_array.tolist()
    else:
        token_ids = token_ids_array
    # Replace negative token IDs with 999 (arbitrary)
    filtered = [tid if tid >= 0 else 999 for tid in token_ids]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)

########################################
# HELPER: TRUNCATE VERY LONG STRINGS
########################################

def shorten_text(text: str, max_len: int = 120) -> str:
    """
    If 'text' is longer than max_len, truncate it and append '(...)' for readability.
    """
    if len(text) > max_len:
        return text[:max_len] + " (...)"
    return text

########################################
# CUSTOM MAIN CONTROLLER THAT STORES BATCHES
########################################

class TestMainController(MainController):
    """
    A specialized MainController that saves the collected batches
    in a list, so we can retrieve them in our tests.
    """
    def __init__(self, tasks_queue, results_queue, objectives_config, max_batches=1):
        super().__init__(tasks_queue, results_queue, objectives_config, max_batches)
        self._collected_batches = []

    def run(self):
        # Step 1) Push instructions
        instructions = {"objectives": self.objectives_config}
        print("\n[TestMainController] Sending objective distribution to worker.")
        self.tasks_queue.put(instructions)

        # Step 2) Collect up to max_batches
        batches_collected = 0
        while batches_collected < self.max_batches:
            try:
                result = self.results_queue.get(timeout=5.0)
            except queue.Empty:
                print("[TestMainController] Timeout waiting for batch. Continuing.")
                continue

            self._collected_batches.append(result)
            batches_collected += 1
            print(f"[TestMainController] Received batch #{batches_collected} from worker {result['worker_id']}.")

        # Step 3) Send sentinel to stop the worker
        print("[TestMainController] Done collecting; sending sentinel to worker.\n")
        self.tasks_queue.put(None)
        self.is_running = False

########################################
# SHORTWORKER: Produce fewer sub-batches
########################################

class ShortWorker(Worker):
    """
    A Worker subclass that only produces a small number of samples
    to avoid large repeated leftover text in logs.
    """
    def _produce_samples(self):
        """
        Produce a small number of samples (e.g. 3) instead of 50
        in the original Worker code.
        """
        for _ in range(3):
            # Step 1) choose text from leftover or data_iter
            if self.leftover_ids.size > 0:
                text_sample = self.leftover_text
                self.leftover_text = ""
                self.leftover_ids = np.array([], dtype=np.int32)
            else:
                try:
                    text_sample = next(self.data_iter)
                except StopIteration:
                    self.initialize_data_iterator()
                    continue

            # Step 2) pick an objective randomly based on cumprobs
            rnd = np.random.rand()
            idx = 0
            while idx < len(self.cumprobs) and rnd > self.cumprobs[idx]:
                idx += 1
            if idx == len(self.cumprobs):
                idx = len(self.cumprobs) - 1

            obj_info = self.objectives_info[idx]
            objective_name = obj_info["name"]
            config_overrides = obj_info.get("config_overrides", {})

            # Step 3) Load + apply objective
            from birdie_rl.load_objective import load_objective
            objective = load_objective(objective_name, config_overrides)
            result = objective(text_sample)
            if result["status"] != "ok":
                continue

            input_ids = result["input_ids"]
            label_ids = result["label_ids"]
            leftover_str_new = result.get("unused_input_string", "")
            leftover_ids_new = result.get("unused_input_ids", np.array([], dtype=np.int32))

            # Step 4) store leftover if any
            if leftover_ids_new.size > 0:
                self.leftover_text = leftover_str_new
                self.leftover_ids = leftover_ids_new

            # Step 5) try to add sample to packer
            try:
                status_info = self.packer.add_sample(input_ids, label_ids)
            except ValueError:
                # If we can't add, pop the current sub-batch + start a new packer
                self._pop_and_store_in_batch(objective_name)
                self.packer = self._create_new_packer()
                self.packer.add_sample(input_ids, label_ids)
                status_info = {"ready": False}

            # Step 6) If the packer is "ready," pop => sub-batch
            if status_info["ready"]:
                self._pop_and_store_in_batch(objective_name)

            # Step 7) If we have enough sub-batches in current_batch, send it
            if len(self.current_batch) >= self.batch_size:
                self._finalize_and_send_batch()

    def _create_new_packer(self):
        """
        Creates a new SequencePacker with the same settings
        as the original worker was initialized with.
        """
        from birdie_rl.packer import SequencePacker
        return SequencePacker(
            sequence_length=self.sequence_length,
            minimum_sequence_length=self.min_seq_len_for_packing
        )

########################################
# HELPER: RUN A SINGLE-OBJECTIVE TEST
########################################

def run_single_test_modified(objective_name, config_overrides, test_text, batch_size=1, max_batches=1):
    """
    Creates a pipeline with a single objective, uses ShortWorker,
    returns the collected batch data from the main controller.

    Args:
        objective_name: e.g. 'copying'
        config_overrides: dict with config for that objective
        test_text: The text sample to feed
        batch_size: sub-batches we gather before sending a batch
        max_batches: total number of final batches to collect
    """
    # 1) Build single-objective config
    objectives_config = [
        {
            "name": objective_name,
            "prob": 1.0,
            "config_overrides": config_overrides
        }
    ]

    # 2) Create queues
    tasks_q = queue.Queue()
    results_q = queue.Queue()

    # 3) MainController
    main_ctrl = TestMainController(
        tasks_queue=tasks_q,
        results_queue=results_q,
        objectives_config=objectives_config,
        max_batches=max_batches
    )

    # 4) Worker
    worker = ShortWorker(
        worker_id=0,
        tasks_queue=tasks_q,
        results_queue=results_q,
        data_generator=lambda: single_sample_gen(test_text),
        sequence_length=256,
        min_seq_len_for_packing=32,
        batch_size=batch_size
    )

    # 5) Start threads
    main_t = threading.Thread(target=main_ctrl.run, daemon=True)
    worker_t = threading.Thread(target=worker.run, daemon=True)
    main_t.start()
    worker_t.start()
    main_t.join()
    worker_t.join()

    # 6) Return all collected batch data
    return main_ctrl._collected_batches

########################################
# TEST DEFINITIONS (calls debug_print_array_dict)
########################################

def test_copying():
    """
    Test the 'copying' objective. Label should match the original text,
    while input has a '[COPY_PROMPT] ' prefix.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: Copying ===")
    print("*" * 60 + "\n")
    test_text = "Hello world from copy test."
    overrides = {
        "remaining_space": 256,
        "paradigm": "[COPY_PROMPT] "
    }
    all_batches = run_single_test_modified("copying", overrides, test_text, batch_size=1)

    # Expect 1 batch => 1 sub-batch
    if not all_batches:
        print("No batches received! Something went wrong.")
        return

    batch = all_batches[0]
    if not batch["batch_items"]:
        print("Empty sub-batch! Check worker logic.")
        return

    sub = batch["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: copying arrays]")

    # Then do a quick decode
    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (copying) completed.\n")


def test_next_token_prediction():
    """
    Test the 'next_token_prediction' objective. Label is original text,
    input has '[NTP_PROMPT]' prefix.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: NextTokenPrediction ===")
    print("*" * 60 + "\n")
    test_text = "Hello NTP test."
    overrides = {
        "remaining_space": 256,
        "paradigm": "[NTP_PROMPT] "
    }
    all_batches = run_single_test_modified("next_token_prediction", overrides, test_text)
    if not all_batches:
        print("No batches received!")
        return

    batch = all_batches[0]
    if not batch["batch_items"]:
        print("Empty sub-batch!")
        return
    sub = batch["batch_items"][0]

    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: next_token_prediction arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (next_token_prediction) completed.\n")


def test_autoencoding():
    """
    'autoencoding' with moderate corruption_rate => placeholders in input,
    original text in label.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: Autoencoding ===")
    print("*" * 60 + "\n")
    test_text = "Hello world from autoencoding test."
    overrides = {
        "remaining_space": 256,
        "corruption_rate": 0.3,
        "tokens_per_mask": 2
    }
    all_batches = run_single_test_modified("autoencoding", overrides, test_text)
    if not all_batches:
        print("No batches returned!")
        return
    sub = all_batches[0]["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: autoencoding arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (autoencoding) completed.\n")


def test_infilling():
    """
    'infilling' => placeholders [mask_0], etc. in both input and label.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: Infilling ===")
    print("*" * 60 + "\n")
    test_text = "Hello from infilling. Let's see if it masks properly."
    overrides = {
        "remaining_space": 256,
        "corruption_rate": 0.5,
        "tokens_per_mask": 2,
        "infilling_prefix": "[mask_"
    }
    all_batches = run_single_test_modified("infilling", overrides, test_text)
    if not all_batches:
        return
    sub = all_batches[0]["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: infilling arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (infilling) completed.\n")


def test_deshuffling():
    """
    'deshuffling' => partial shuffle. The label is the original order,
    input has the tokens partially shuffled.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: Deshuffling ===")
    print("*" * 60 + "\n")
    test_text = "One two three four five deshuffling test."
    overrides = {
        "remaining_space": 256,
        "percentage_of_tokens_to_shuffle": 0.5
    }
    all_batches = run_single_test_modified("deshuffling", overrides, test_text)
    if not all_batches:
        return
    sub = all_batches[0]["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: deshuffling arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (deshuffling) completed.\n")


def test_prefix_language_modeling():
    """
    'prefix_language_modeling' => about 60% prefix, 40% suffix.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: PrefixLanguageModeling ===")
    print("*" * 60 + "\n")
    test_text = "Prefix LM test to see how we handle partial prefix."
    overrides = {
        "remaining_space": 256,
        "prefix_fraction": 0.6
    }
    all_batches = run_single_test_modified("prefix_language_modeling", overrides, test_text)
    if not all_batches:
        return
    sub = all_batches[0]["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: prefix_lm arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (prefix_language_modeling) completed.\n")


def test_selective_copying():
    """
    'selective_copying' => with query_context format. 
    We'll see [FUNC_x] [COPY] blocks, [CONTEXT], and [RESULT_x] in label.
    """
    print("\n" + ("*" * 60))
    print("=== TEST: SelectiveCopying ===")
    print("*" * 60 + "\n")
    test_text = "Selective copying test with spans."
    overrides = {
        "remaining_space": 256,
        "formatting_type": "query_context",
        "min_span_length": 2,
        "max_span_length": 5,
        "min_num_spans": 1,
        "max_num_spans": 2
    }
    all_batches = run_single_test_modified("selective_copying", overrides, test_text)
    if not all_batches:
        return
    sub = all_batches[0]["batch_items"][0]
    tokenizer = Tokenizer()

    # Print numeric arrays
    print("\n===== Packed Data Arrays =====")
    debug_print_array_dict(sub["packed_data"], heading="[Debug: selective_copying arrays]")

    dec_inp = shorten_text(tokenizer.decode(sub["packed_data"]["input_ids"]))
    dec_lbl = shorten_text(safe_decode(tokenizer, sub["packed_data"]["label_ids"]))

    print("\nDecoded input:", dec_inp)
    print("Decoded label:", dec_lbl)
    print("TEST (selective_copying) completed.\n")

########################################
# MAIN
########################################

def main():
    """
    Calls each test in sequence, printing the results.
    """
    test_copying()
    test_next_token_prediction()
    test_autoencoding()
    test_infilling()
    test_deshuffling()
    test_prefix_language_modeling()
    test_selective_copying()
    print("All tests done.")

if __name__ == "__main__":
    main()
