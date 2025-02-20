# Birdie: Reward-driven Automated Curricula

**Birdie** was published at **EMNLP 2024**!  
Check out our paper on arXiv: [arXiv:2411.01030](https://arxiv.org/abs/2411.01030).

Birdie RL is an open-source framework designed to automate **multi-objective** training using a **reward-driven** curriculum.

With dynamically mixes of training tasks -- including selective copying, next token prediction, autoencoding, infilling, copying, and prefix-LM -- Birdie automatically attempts to optimize model learning according to a **reward model** that tracks per-objective loss improvements, conditioned on the entire history.

This codebase is designed to be hackable, allowing for new reward functions.
Currently, only causal-only or prefix-LM **state space models** and Transformers decoder-only models are best supported.
Birdie also features **sequence packing** for efficient batching.


## Usage

Below is a quick start for integrating Birdie RL in your training loop:

```python
from birdie_rl import Birdie
from example_usage.ul2_config import ul2_config
import tiktoken
import accelerate

# Configuration
config = {
    "batch_size": 8,
    "sequence_length": 2048,
    "num_workers": 16,
    "steps_between_evaluations": 32,
    "num_steps": 4096,
    "accelerator": accelerate.Accelerator(),
    "tokenizer": tiktoken.get_encoding("o200k_base"),
    "objectives": ul2_config,
    "ds": your_data_generator_function,  # Provide your dataset or generator
    "reward_fn": your_reward_function,   # Define your custom reward logic
}

# Initialize Birdie
birdie = Birdie(config)

# Training Loop
for step in range(config["num_steps"]):
    # Periodic evaluation
    if birdie.time_for_eval(step):
        for (objective_name, batch) in birdie.measure_validation_losses():
            loss = model(**batch)  # Example model inference
            birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step)

    # Fetch the next training sample from Birdie
    sample = birdie.get_next_training_sample()
    # Pass 'sample' to your model and do a training step
```

You can find more detailed examples in:
- **`example_usage/example.py`** for a minimal script
- **`example_usage/ul2_config.py`** for UL2-style objectives
- **`example_usage/utils.py`** for custom reward functions and data generator demos

---


<div align="center">
  <a href="https://github.com/samblouir/birdie/blob/main/birdie_emnlp_2024_poster.jpg?raw=true"><img src="https://github.com/samblouir/birdie/blob/main/birdie_emnlp_2024_poster.jpg?raw=true" alt="Birdie EMNLP 2024 Poster" width="800" /></a>
</div>

---

## Features & Highlights

- **Automated Multi-Objective Training**  
  This all-in-one pipeline easily adds an automated curriculum with multi-objective training, including **autoencoding**, **deshuffling**, **infilling**, **copying**, etc. all with customizable parameters.

- **Character-level noising functions**
   By default, Birdie's noise functions work on the character-level for text. Long inputs are automatically sliced into suitable chunks to fit into your desired maximum sequence length.

- **Reward-Driven Curriculum**  
  Birdie uses a Transformer reward model to adaptively select objectives, optimizing training based on sub-loss improvements, historical objective mixture rates, and any other factors.

- **Efficient Data Pipeline**  
  Integrates multi-worker processing and **sequence packing** to reduce wasted compute, boosting effective tokens per second throughput during training.

- **Huggingface Accelerate Support**
   Birdie is compatible with Huggingface's Accelerate library, allowing for easy scaling to multiple GPUs or TPUs. Birdie currently supports model parallel setups for the dataloader. JAX compatibility to be added soon.

- **Modular Architecture**  
  Birdie is designed to be hackable. Easily add new objectives, custom reward functions, and other pipeline components to experiment with different strategies.

- **Paper**  
   Birdie was published at EMNLP 2024, where it brought SSMs and Transformer models to state-of-the-art performance on several tasks, compared to standard next token prediction training.


---

## Installation

### Simplest approach
   ```bash
   pip install git+https://github.com/samblouir/birdie.git
   ```

   Please see "example_usage/example.py" for an example of how to use Birdie with your Torch (or, with minimal modifications, JAX) training loop.

### In-depth approach

#### Prerequisites
- Python 3.8+
- Git

#### Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/samblouir/birdie.git
   cd birdie-rl
   ```

2. **Install Dependencies**  
   Birdie RL relies on `numpy`, `torch`, `datasets`, and `accelerate`. Install them via:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, manually `pip install numpy torch datasets accelerate`.)*

3. **Verify Setup**  
   Test everything with a sample script:
   ```bash
   python example_usage/example.py
   ```

---

## Codebase Overview

**Directory Structure** (simplified):
```
birdie_rl/
  birdie_reward_model/
    birdie.py           # Main Birdie class
    agent_bird.py       # RL agent logic
    reward_model.py     # Simplified API for the reward model
    rotary.py           # Rotary positional encoding utilities
    gated_ssm_agent.py  # Default Transformer
  example_usage/
    example.py          # Minimal usage script
    ul2_config.py       # UL2-inspired objectives
    utils.py            # Shows functions Birdie needsreward fn, data gen, etc.
  objectives/
    base.py              # BaseObjective class. Shows how to add objectives.
    selective_copying.py # A new structured-deshuffling objective introduced in Birdie
    autoencoding.py      # BART-style autoencoding, with deshuffling support
    infilling.py        
    copying.py          
    deshuffling.py      
    next_token_prediction.py
    prefix_language_modeling.py
  pipeline/
    main_controller.py  # Objective distribution & worker coordination
    packer_batcher.py   # Sequence packing logic
    worker.py           # Worker processes to transform data
    pipeline_generator.py
  load_objective.py      # Registry for objective loading
  ...
```

**Key Components**:
- **`birdie_reward_model/`**  
  Hosts the RL agent (`agent_bird.py`), the main Birdie orchestrator (`birdie.py`), and optional gating/MLP code.  
- **`objectives/`**  
  Houses all self-supervised tasks (infilling, copying, etc.) derived from `BaseObjective`.
- **`pipeline/`**  
  Manages multi-process data generation, sequence packing, and distributing tasks among workers.

---

## Contributing

We **strongly welcome** contributions! Whether it’s a new objective, a fresh reward function, or bug fixes, we appreciate your help in making Birdie RL better.

Please feel free just post in discussion.
Please open issues for feature requests or bug reports.

Alternatively, you can fork the repository and submit a pull request with your changes. Here’s a quick guide:
1. Fork the repository  
2. Create a branch (`git checkout -b feature/awesome-update`)  
3. Commit your changes (`git commit -m "Add something awesome"`)  
4. Push & open a Pull Request  


---

## License & Citation

Birdie RL is released under the **Apache License 2.0**. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.
The intent is to be as permissible as possible for any kind of usage.

If you use (or build on) Birdie RL in your work, kindly cite our **EMNLP 2024** paper:

```bibtex
@inproceedings{blouir-etal-2024-birdie,
    title = "Birdie: Advancing State Space Language Modeling with Dynamic Mixtures of Training Objectives",
    author = "Blouir, Sam and Smith, Jimmy T.H. and Anastasopoulos, Antonios and Shehu, Amarda",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.541/",
    doi = "10.18653/v1/2024.emnlp-main.541",
    pages = "9679--9705",
}
```



