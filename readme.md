# Birdie

Paper accepted to EMNLP 2024! Come see us on Tuesday, November 12th, 2:00 - 3:30 PM.

The latest version of the paper is available at: https://arxiv.org/abs/2411.01030.


![Image of Birdie's EMNLP Poster](https://github.com/samblouir/birdie/blob/main/birdie_emnlp_2024_poster.jpg?raw=true)

# Birdie
Planned integrations with HuggingFace and EasyDeL for ease-of-use.

**The codebase is being refactored and will be updated soon. Currently, it is bring-your-own-model.**

## Existing Features:
Example usage:
```python

import tokenizer as huggingface_tokenizer
from loss_fns import log_softmax_cross_entropy


model = "Your Model"
tokenizer = huggingface_tokenizer.get_tokenizer("Your Tokenizer")



from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", streaming=True,)


from birdie.all_in_one import Birdie
config = {
	"tokenizer": tokenizer,
	"pretraining_objectives": "birdie",

	"total_steps": 10_000,
	"steps_between_evaluations": 500,

	"batch_size": 8,
	"max_sequence_length": 4096,

	"dataset": dataset,
}
birdie = Birdie(config)

for step_idx in range(10_000):

	if birdie.time_for_eval():
		model.eval()
		for x in birdie.get_validation_samples():
			preds = model(x['input_ids'])
			loss = log_softmax_cross_entropy(preds, x['labels'], x['loss_mask'])
			birdie.update(loss)

		validation_losses = birdie.get_validation_losses()
	
	x = birdie.get_next_training_sample()

	preds = model(x['input_ids'])
	loss = log_softmax_cross_entropy(preds, x['labels'], x['loss_mask'])
	loss.backward()
	optimizer.step()



```

## Demos

### Pre-training objectives
To see the training objectives being applied to a sample story, run:
```python
python configs/birdie_config.py
```
### Agent Bird
To see Birdie train on a synthetic model and dataset, run:
```python
python birdie/all_in_one.py
```
This will print the actions and losses chosen by Birdie as it learns to choose objectives in the best-performing objective class (class 5)

## Planned features coming soon:
### Train

This trains model using Birdie, Next Token Prediction, UL2, or Fixed Ratio Mixtures.

Available models include:
- Gated SSM
- Hawk
- Transformer

### Story Infilling

This script runs the models on our new infilling dataset.

### Multi-phone number retrieval

This will create and cache a phone number dataset using the specified hyperparameters.

The following commandline options are available:
- How many phone numbers to retrieve simultaneously
  - Note: A quantity of 32 phone numbers means finding and retrieving 320 total digits.
- The minimum and maximum number of names in the phonebook.
- Format the phonebook using {question-context-question}, {question-context}, or {context-question}.
- For evaluation:
  - The quantity of evaluation phonebooks
  - The minimum and maximum number of entries in phonebooks

### SQuAD V2 Question-Answering

This evaluates models on SQuAD V2, using all available answerable questions.


# JAX Adapter for EleutherAI LM Harness
This adapter allows for the use of JAX (and other framework) models with the EleutherAI LM Harness. It is designed originally for max-likelihood/multiple-choice tasks, such as ARC and MMLU, but will be extended to support generative tasks.

## Overview

The system is divided into three main components:

1. **api.py**
   - You just need to fill-in two functions:
     - Model loader: Given a model_tag or other arguments, load and prepare a model for usage
     - Model caller: Accepts batched input_ids (shape: (batch, sequence_length)) and returns the logits from your model in a 3D shape (batch, length, vocab_size).
     - Tokenizer: Tokenizes from a string or list of strings. An example is given in the api.py file itself.
  
2. **birdie_adapter.py (Add this to your EleutherAI LM Harness installation)**:
   - You can install this by copying `birdie_adapter.py` to `lm-evaluation-harness/lm_eval/models/birdie_adapter.py`.
    - Receives data and arguments from the EleutherLM Harness.
    - Forwards the request to the "server.py".
    - Receives the response and performs some minimal processing.
    - Returns the response to the EleutherLM Harness.
   
3. **server.py**:
    - Receives requests from "birdie_adapter.py"
    - Loads and runs the requested model on the inputs.
    - Sends the output back to "birdie_adapter.py", which  EleutherLM Harness via "birdie_adapter.py".


## Notes

- "server.py" should be recursively called in a while-loop (explained in section "Running the Server" below) due JAX's official method for resetting allocated GPU VRAM causing segmentation faults. As a workaround, the server crudely unloads models by exiting entirely and restarting.

## Usage

### Installation

- [Download the EleutherAI LM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
- Copy in the birdie_adapter.py file:
```bash
cp birdie_adapter.py lm-evaluation-harness/lm_eval/models/birdie_adapter.py
```

- (Probably) re-install JAX, since Torch may have changed your NVIDIA packages...:

[Please see JAX's installation instructions on their Github:](https://github.com/jax-ml/jax?tab=readme-ov-file#installation)
| Platform        | Instructions                                                                                                    |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| CPU             | `pip install -U jax`                                                                                            |
| NVIDIA GPU      | `pip install -U "jax[cuda12]"`                                                                                  |
| Google TPU      | `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`                 |
| AMD GPU (Linux) | Use [Docker](https://hub.docker.com/r/rocm/jax-community/tags), [pre-built wheels](https://github.com/ROCm/jax/releases), or [build from source](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus). |
| Mac GPU         | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |

See [the documentation](https://jax.readthedocs.io/en/latest/installation.html)
for information on alternative installation strategies. These include compiling
from source, installing with Docker, using other versions of CUDA, a
community-supported conda build, and answers to some frequently-asked questions.


- Move `birdie_adapter.py` to `lm-evaluation-harness/lm_eval/models/`.

### Running the Server

- Start the server using
  ```bash
  # This loop is for
  while True; do
     python host_main.py
     echo "Restarting server..."
     sleep 5s
  done
  ```
It listens for requests from the EleutherLM Harness and handles model loading and predictions accordingly.

### Starting Tasks
Here is a real example of how I used this harness in Birdie
```bash

## Variables to set:
# Comma-seperated list of tasks to run
tasks="boolq"
model_tag="attention_trained_using_birdie_14B" # This is passed to API.py, which you must update to load in your desired model.
kwargs="port=5000,max_sequence_length=65536" # comma-seperated values that you can pass in. These will make it all the way to load_model() and load_tokenizer() in API.py!

# Index of the GPU(s) to use, on your machine
gpu_id=0

# Number of fewshot examples to use (Note: The Eleuther harness doesn't support this for all tasks, and may not tell you that few shot examples weren't added...)
num_fewshot=0
results_dir="/home/sam/birdie_eleuther_results"
cache_dir="/home/sam/birdie_eleuther_cache"
## End of variables to set


# Json output path. This will save your results to a json file
# NOTE: If you do too many tasks at once, this may create a file that is too long for your OS!
output_path=${results_dir}/${tasks}_fewshot:${num_fewshot}.json

# Ensures the results directory exists
mkdir -p ${results_dir}/${model_tag}

# Place your model args using comma-seperated values
model_args="--model_args model_tag=${model_tag}" 
cache_args="--device cuda:${gpu_id} --use_cache ${cache_dir} 
fewshot_args="--num_fewshot ${num_fewshot}"

# Runs it! This will start Eleuther's harness, and thanks to your custom model, the requests will be forwarded to our server running in host_main.py, which will load the model, tokenize the inputs, and return the final losses to the harness.
python /home/sam/lm-evaluation-harness/lm_eval/__main__.py --model birdie ${model_args} ${fewshot_args} --tasks ${tasks}  --output_path ${output_args} ${cache_args}

# That's all!
```


### Overall Harness workflow

1. Launch the server.
2. The server receives a request from the EleutherLM Harness.
3. The request contains the desired model tag and the inputs.
4. If the model has not been loaded:
    - Load the model.
    - Store the model tag.
5. If a different model is requested:
    - Exit the program.
    - An outside script automatically restarts the server to load the new model.
6. Make a prediction using the loaded model.
7. Return the prediction to the EleutherLM Harness via the connection with birdie_adapter.py
