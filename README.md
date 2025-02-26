# Birdie: Reward-driven Automated Curricula

**Birdie** was published at **EMNLP 2024**!

Please check out our paper on arXiv: [arXiv:2411.01030](https://arxiv.org/abs/2411.01030).

Birdie RL is an open-source framework designed to automate **multi-objective** training using a **reward-driven** curriculum.

Using dynamic mixtures of training tasks -- including selective copying, next token prediction, autoencoding, infilling, copying, and prefix-LM -- Birdie automatically attempts to optimize model learning according to a **reward model** that tracks per-objective loss improvements, conditioned on the entire history.

This codebase is designed to be hackable, allowing for swappable reward functions and objectives.
Currently, decoder-only and causal or prefix-LM **state space models** and **Transformers** are supported.
Birdie also features **sequence packing** for efficient training.

For full performance benefits, **it is strongly recommended to use a prefix-LM SSM or Transformer with Birdie.** Please see "example_usage/base_model.py" for an example of a prefix-LM Transformer in PyTorch.
Birdie benefits both causal and bidirectional models on multi-Phone number retrieval, but most strongly improved SQuAD v2 performance when coupled with a bidirectional, prefix-LM model.

### Installation
   ```bash
   # For a standard installation
   pip install git+https://github.com/samblouir/birdie.git

   # To upgrade to the latest version
   pip install git+https://github.com/samblouir/birdie.git --upgrade --no-deps

   # To re-install and get the latest version
   pip install git+https://github.com/samblouir/birdie.git --force-reinstall --no-deps
   ```

---

<div align="center">
  <a href="https://github.com/samblouir/birdie/blob/main/birdie_emnlp_2024_poster.jpg?raw=true"><img src="https://github.com/samblouir/birdie/blob/main/birdie_emnlp_2024_poster.jpg?raw=true" alt="Birdie EMNLP 2024 Poster" width="800" /></a>
</div>

---

# Usage

Below is a quick start for integrating Birdie RL in your training loop.
There are two primary components needed to use Birdie: adding a few lines to your training loop, and preparing dataloader and grabber functions.

You can find usage examples in:
- **`birdie_dna`** *COMING SOON* for a complete working example with a domain specific pre-training objective configuration, unique dataset, and tokenizer.
- **`example_usage/base_model.py`** for bidirectional, prefix-LM Transformer that use's Birdie's objectives.
- **`example_usage/example.py`** for a minimal working example with a dummy model.
- **`example_usage/ul2_config.py`** to see how to define objectives (Using UL2's objectives).
- **`example_usage/utils.py`** to see how to structure a custom reward function, as well as a data generator.
function.

## 1) Add Birdie to your training loop
```
# Training Loop
for step_idx in range(config["num_steps"]):

    # Periodic evaluations (set in the config: "steps_between_evaluations")
    if birdie.time_for_eval(step_idx):
        model.eval()
        for (objective_name, batch) in birdie.measure_validation_losses():
            loss = model(**batch)
            birdie.log_validation_loss(key=objective_name, loss=loss, step_idx=step_idx)
         model.train()

    # Fetch the next training batch from Birdie. It is of a fixed-shape, defined by (batch, sequence_length) in the config..
    batch = birdie.get_next_training_sample()
    loss = model(**batch)
    optimizer.zero_grad()

    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## 2) Configuration

### Create an instance of Birdie

Define a config and create an instance of Birdie.
*Additional configuration settings are documented in birdie_rl/birdie_reward_model/birdie.py Birdie.__init__().*

```python
from birdie_rl import Birdie
from birdie_rl.example_usage.ul2_config import ul2_config
import tiktoken
import accelerate

# Configuration


config = {
    # This is the batch size that Birdie will use.
    "batch_size": 8,

    # This is the sequence length shape that Birdie will pack your inputs into.
    # Padding tokens (0's) will be added to the rightside.                          
    "sequence_length": 2048,

    # This controls the number of training dataset dataloader workers (per process)
    # If you have added more intense objectives or are being bottlenecked by the dataloader, feel free to increase this number.
    "num_workers": 8,                                 

    # Birdie will calculate new objective sampling ratios every `steps_between_evaluations` steps
    "steps_between_evaluations": 1024,       

    # This is used by the reward model - there are more parameters that can be set
    #  the default is to - with cosine decay -  explore ratios during the first half of training and exploit during the second half.
    "num_steps": 32_768,        

    # Accelerate is currently required in this version of the code.
    "accelerator": accelerate.Accelerator(),     

    # Any tokenizer will work that uses .encode() and .decode()
    "tokenizer": tiktoken.get_encoding("o200k_base"), 

    # This uses the objectives from UL2, and lets Birdie adjust them during training. Pass in no objectives to use the default (Birdie) objectives.
    "objectives": ul2_config,                       

    # If desired, define a new custom reward function, if you like. Please see example_usage/utils.py's reward_fn() for an example.
    "reward_fn": your_reward_function,        

    # Provide your dataset fn here (See section 3 below)
    "ds": data_generator_fn,                   

    # Define how to extract text from your dataset in whichever way you want. (See section 3 belowbelow)
    "text_grabber_fn": text_grabber_fn,

    # Adds a separator between the prefix and suffix regions.
    "start_generating_paradigm": "\n<|assistant|>\n", # This is also the default

}

# Initialize Birdie
birdie = Birdie(config)
```



### Preparing your dataloader functions

The data_generator_fn and text_grabber_fn's are critical!

It should return an iterable object for a given split, worker_id, num_workers, and rng_seed.
This will allow your code to work across anywhere from one to multiple machines.
You can also do whatever you like in data_generator_fn, including loading entirely different datasets than what you are training on.

### 4) Data generator function using HuggingFace's datasets:
```python
def huggingface_data_generator_fn(split, worker_id, num_workers, rng_seed=0):
	"""
	The data_generator function will be called by each dataloading worker.
	This currently only data parallel training, where each accelerator has its own copy of the model.

	This function should return a generator for a given
	  - split (e.g., "train", "validation", "test")
	  - shard defined by by worker_id and num_workers
	  - shuffle data using rng_seed
	"""

	# Load the TinyStories dataset from Hugging Face
	ds = load_dataset("roneneldan/TinyStories", split=split)

	# Shuffle the dataset for randomness
	ds = ds.shuffle(rng_seed)

	# Shard the dataset among multiple workers
	ds = ds.shard(num_shards=num_workers, index=worker_id)

	# Return the prepared dataset
	return ds
```

#### Data generator function from a list:
```python
def data_generator_fn(split, worker_id, num_workers, rng_seed=0):
    """
    The data_generator function will be called by each dataloading worker.
    This currently only data parallel training, where each accelerator has its own copy of the model.

    This function should return a generator for a given
    - split (e.g., "train", "validation", "test")
    - shards it by worker_id and num_workers
    - shuffles the data using rng_seed
    """

    ds = dataloader.prepare_dataset_as_list()

    # Load the TinyStories dataset from Hugging Face
    if split == "train":
      ds = ds["train"]
    elif split == "validation":
      ds = ds['validation']

    # Shuffle the dataset for randomness
    seeded_np_rng = np.random.default_rng(rng_seed)
    seeded_np_rng.shuffle(ds)

    # Shard the dataset among multiple workers
    ds = ds[worker_id::num_workers]

    # Return the prepared dataset
    return ds
  ```


#### Important: Element grabber function
   If each element of ds_train looks like this:
   ```python
   {
     "entry": {
                 "text": "This is a story about a cat.",
              },
     "source": "www.facts.com",
   }
   ```
  
   Then we can make a text_grabber_fn like this to tell the dataloader how to extract the text from each element.
  ```  
  def text_grabber_fn(x):
    return x["entry"]["text"]
  ```

  Then, we pass it into Birdie's config as "text_grabber_fn": text_grabber_fn

  For the above HuggingFace example using TinyStories, we want to use this text_grabber_fn:

  ```python
     def text_grabber_fn(x):
        return x["text"]
  ```

   






## Additional important usage notes:

Birdie's code assumes your model accepts the following keyword arguments:
- `input_ids` (torch.Long): The input token IDs in a shape of (batch_size, sequence_length)
- `label_ids` (torch.Long): The target token IDs in a shape of (batch_size, sequence_length). This is used for calculating the loss.
- `attention_mask` (torch.Long): The attention mask in a shape of (batch_size, sequence_length). Indices with 1 are areas allowed to have bidirectional Attention. Indices with 0 should be modeled causally.
- `segment_ids` (torch.Long): The segment IDs in a shape of (batch_size, sequence_length). This is used for models that support segment embeddings.

---


## Features & Highlights

- **Automated Multi-Objective Training**  
  This all-in-one pipeline easily adds an automated curriculum with multi-objective training, including **autoencoding**, **deshuffling**, **infilling**, **copying**, etc. all with customizable parameters.

- **Character-level noising functions**
   By default, Birdie's deshuffling function works at the character-level for text.

- **Reward-Driven Curriculum**  
  Birdie uses a Transformer reward model to adaptively select objectives, optimizing training based on sub-loss improvements, historical objective mixture rates, and any other factors.

- **Efficient Data Pipeline**  
  Integrates multi-worker processing and **sequence packing** to reduce wasted compute, boosting effective tokens per second throughput during training. Long inputs are automatically sliced into chunks to fit into your desired maximum sequence length across batches.

- **Huggingface Accelerate Support**
   Birdie is compatible with Huggingface's Accelerate library, allowing for easy scaling to multiple GPUs or TPUs. Birdie currently supports model parallel setups for the dataloader. JAX compatibility to be added soon.

- **Modular Architecture**  
  Birdie is designed to be hackable. Easily add new objectives, custom reward functions, and other pipeline components to experiment with different strategies.

- **Paper**  
   Birdie was published at EMNLP 2024, where it we saw strong benefits versus standard next token prediction training on several NLP comprehension and retrieval tasks.


---

## Installation

### Simplest install approach

First, install birdie.
   ```bash
   pip install git+https://github.com/samblouir/birdie.git
   ```

Then see "example_usage/example.py" for an example of how to use Birdie with your Torch (or, with minimal modifications, JAX) training loop.

## Dataloader Debugging

Data processing issues?
in *birdie_rl/pipeline/worker.py*, uncomment the print line in this function:
```python
  	def print(self, *args, **kwargs):
      """
      Helper method to print with worker info.
      """
      # print(*args, **kwargs) ## Uncomment this to enable worker debug printing
      pass
```

### In-depth Installation Instructions


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



