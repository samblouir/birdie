import jax
import jax.numpy as jnp
import typing


## Please fill in prepare_model and prepare_tokenizer
## These are currently dummy functions that return a dummy model and tokenizer.

def prepare_model(**kwargs):

	model_tag = kwargs.get("model_tag", "dummy_model")

	import api
	model = api.Model(**kwargs)

	return model

def prepare_tokenizer(**kwargs):
	
	class Tokenizer():
		
		def __init__(self, **kwargs):
			self.kwargs = kwargs
			# Sets the beginning of sequence ID
			self.bos_id = int(kwargs.get("bos_id", 0))
		
		def __call__(self, sequence: str) -> list[int]:
			# Should return a list of input_ids
			return [self.bos_id] + list(map(ord, sequence))

	return Tokenizer()



class ModelWrapper():

	def __init__(self, **kwargs):
		self.kwargs = kwargs

		# You can access these kwargs like so:
		self.model_tag = self.kwargs['model_tag']

		# or, more riskily:
		self.__dict__.update(kwargs)

		## Prepare your model here
		# self.model = ...
		self.model = prepare_model(**kwargs)

		# Prepare your tokenizer here.
		# self.tokenizer = ...
		self.tokenizer = prepare_tokenizer(**kwargs)


	def tokenize(self, inputs: list[str], labels: list[str], delimiter='\n\n',):
		# -> dict(
		# 	inputs: list of [numpy or jax array]:
		# 	labels: list of [numpy or jax array]:
		# ):
		'''
		This takes in two lists: inputs and labels.
		Each list is list of strings.
		This function should return a dictionary with two keys: inputs and labels.


		This function should tokenize lists of strings.
		If you are using a standard, causal, and decoder-only Transformer, you can use something like the following code:
		'''
		tokenized_inputs = []
		tokenized_labels = []
		for input, label in zip(inputs, labels):
			sequence = (input + delimiter + label)
			tokenized_sequence = self.tokenizer(sequence)
			tokenized_input = tokenized_sequence['input_ids'][:-1]
			tokenized_label = tokenized_sequence['input_ids'][1:]
			tokenized_inputs.append(tokenized_input)
			tokenized_labels.append(tokenized_label)

		return {'inputs': tokenized_inputs, 'labels': tokenized_labels}


	def predict(batch: dict):# -> list[jnp.float32]:
		"""
		This function processes one batch at a time, and should return a list or array of scalars representing the loss.

		For example
			batch['inputs'].shape: (batch_size, sequence_length)
			batch['labels'].shape: (batch_size, sequence_length)

			logits = model(
						input_ids=batch['input_ids'],
						label_ids=batch['label_ids'],
					)
		"""