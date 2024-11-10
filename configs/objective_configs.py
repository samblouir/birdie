import dataclasses
import typing
import numpy as np
from birdie_objectives import _span

def remap(x, in_min=None, in_max=None, out_min=-1.0, out_max=1.0):
	if in_min is None:
		in_min = np.min(x)
	if in_max is None:
		in_max = np.max(x)
	div = (in_max - in_min) + 1e-8
	return (x - in_min) * (out_max - out_min) / div + out_min


def default_tokenizer(*args, **kwargs,) -> typing.NoReturn:
    raise NotImplementedError("\"tokenizer\" field not set, but was used.")



@dataclasses.dataclass
class ObjectiveConfig:
	objective: str = ""
	kwargs: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)
	paradigm: str = ""
	tokenizer: typing.Optional[typing.Callable] = default_tokenizer

	def set_tokenizer(self, tokenizer):
		self.tokenizer = tokenizer



@dataclasses.dataclass
class NextTokenPredictionConfig(ObjectiveConfig):
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)
	use_begin_generating_paradigm:int = 0

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Next Token Prediction'

	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)

	def __call__(self, input_ids, **kwargs,):
		'''
		'''
		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
		input_ids = np.int32(input_ids)
		if self.use_begin_generating_paradigm:
			label_ids = input_ids
			input_ids = np.concatenate([[self.special_token_ids.get('start_generating_id', 1),], input_ids[:-1],])

		else:
			label_ids = input_ids[1:]
			input_ids = input_ids[:-1]

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
		)
	



@dataclasses.dataclass
class PrefixLanguageModelingConfig(ObjectiveConfig):
	prefix_frac: float = 0.75
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	minimum_prefix_length:int = 4
	maximum_prefix_length:int = -1
	minimum_suffix_length:int = 4
	maximum_suffix_length:int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)
	paradigm_style:str = dataclasses.field(default_factory=str)

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Prefix Language Modeling'
		
		self.np_rng = np.random.default_rng(self.rng_seed)

	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)

		if len(input_ids) < (self.minimum_prefix_length + self.minimum_suffix_length):
			return dict(
				status="error",
				message=f"The input sequence is not long enough to split. (len(input_ids): {len(input_ids)}, minimum_prefix_length: {self.minimum_prefix_length}, minimum_suffix_length: {self.minimum_suffix_length})",
			)
		
		return dict(
			status="ok",
		)

	def __call__(self, input_ids, **kwargs,):
		'''
		'''
		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
		random_split_point_float = self.np_rng.random(1, dtype=np.float32)[0]

		if ((1 - self.prefix_frac) < self.prefix_frac):
			max_deviation = (1 - self.prefix_frac)
		else:
			max_deviation = self.prefix_frac

		start_idx = (len(input_ids) * self.prefix_frac) - (len(input_ids) * max_deviation)
		end_idx = (len(input_ids) * self.prefix_frac) + (len(input_ids) * max_deviation)

		if (start_idx < self.minimum_prefix_length):
			start_idx = self.minimum_prefix_length

		if (len(input_ids) - end_idx < self.minimum_suffix_length):
			end_idx = (len(input_ids) - self.minimum_suffix_length)

		random_split_point_float = remap(random_split_point_float, 0.0, 1.0, start_idx, end_idx)
		random_split_point_int = np.floor(random_split_point_float).astype(np.int32)
		
		input_ids = np.int32(input_ids)
		
		label_ids = input_ids[random_split_point_int:]
		input_ids = input_ids[:random_split_point_int]

		input_ids = np.concatenate([input_ids, [int(self.special_token_ids.get('start_generating_id', 1)),],])
		label_ids = np.int32(label_ids)

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
		)





@dataclasses.dataclass
class CopyingConfig(ObjectiveConfig):
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Copying'
		self.np_rng = np.random.default_rng(self.rng_seed)

	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)

	def __call__(self, input_ids, **kwargs,):
		'''
		'''

		remaining_space = int(kwargs.get("remaining_space", 0))
		unused_input_ids = None
		if len(input_ids) < remaining_space//2 - 8:
			unused_input_ids = input_ids[remaining_space//2-8:]
			input_ids = input_ids[:remaining_space//2-8]

		if self.maximum_sequence_length < len(input_ids):
			if unused_input_ids is not None:
				unused_input_ids = np.concatenate([unused_input_ids, input_ids[self.maximum_sequence_length:],])
			else:
				unused_input_ids = input_ids[self.maximum_sequence_length:]
			input_ids = input_ids[:self.maximum_sequence_length]



		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
		input_ids = np.concatenate([input_ids, [self.special_token_ids.get('start_generating_id', 1),],],)
		label_ids = input_ids[:-1]

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
			unused_input_ids=unused_input_ids
		)



@dataclasses.dataclass
class DeshufflingConfig(ObjectiveConfig):
	percentage_of_tokens_to_shuffle: float = 1.0
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Deshuffling'
		self.np_rng = np.random.default_rng(self.rng_seed)


	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)
	
	def __call__(self, input_ids, **kwargs,):
		'''
		'''

		remaining_space = int(kwargs.get("remaining_space", 0))
		unused_input_ids = None
		if len(input_ids) < remaining_space//2 - 8:
			unused_input_ids = input_ids[remaining_space//2-8:]
			input_ids = input_ids[:remaining_space//2-8]

		if self.maximum_sequence_length < len(input_ids):
			if unused_input_ids is not None:
				unused_input_ids = np.concatenate([unused_input_ids, input_ids[self.maximum_sequence_length:],])
			else:
				unused_input_ids = input_ids[self.maximum_sequence_length:]
			input_ids = input_ids[:self.maximum_sequence_length]

		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
		label_ids = np.int32(input_ids)

		indices = np.arange(len(input_ids))
		self.np_rng.shuffle(indices)

		# Determine the number of indices to shuffle
		num_indices_to_shuffle = int(len(input_ids) * self.percentage_of_tokens_to_shuffle)
		shuffle_indices = indices[:num_indices_to_shuffle]

		# Shuffle only the specified tokens
		input_ids = label_ids.copy()
		input_ids[shuffle_indices] = label_ids[shuffle_indices[self.np_rng.permutation(num_indices_to_shuffle)]]


		input_ids = np.concatenate([input_ids, [self.special_token_ids.get('start_generating_id', 1),],],)

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
			unused_input_ids=unused_input_ids
		)



@dataclasses.dataclass
class SelectiveCopyingConfig(ObjectiveConfig):
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	minimum_span_length: int = 8
	maximum_span_length: int = 16
	num_spans: int = 1
	formatting_type: str = dataclasses.field(default_factory=str)
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)

	mean_start_token_length: int = 4
	mean_end_token_length: int = 2

	paradigm_function_str: str = dataclasses.field(default_factory=str)
	paradigm_context_str: str = dataclasses.field(default_factory=str)
	paradigm_start_span_str: str = dataclasses.field(default_factory=str)
	paradigm_end_span_str: str = dataclasses.field(default_factory=str)
	paradigm_done_str: str = dataclasses.field(default_factory=str)
	paradigm_sep_str: str = dataclasses.field(default_factory=str)

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Selective Copying'
		self.np_rng = np.random.default_rng(self.rng_seed)
		self.possible_span_lengths = np.arange(self.minimum_span_length, self.maximum_span_length + 1)
		self.possible_start_token_lenmgths = np.arange(1, self.mean_start_token_length*2 + 1)
		self.possible_end_token_lengths = np.arange(1, self.mean_end_token_length*2 + 1)

		self.span_finding_attempts = max(32, self.num_spans*2,)
		self.repeat_attempts = 10

	def set_tokenizer(self, tokenizer):
		super_return_value = super().set_tokenizer(tokenizer)
		self.tokenized_function_paradigm = self.tokenizer(self.paradigm_function_str)
		self.tokenized_start_paradigm = self.tokenizer(self.paradigm_start_span_str)
		self.tokenized_end_paradigm = self.tokenizer(self.paradigm_end_span_str)
		self.tokenized_context_paradigm = self.tokenizer(self.paradigm_context_str)
		self.tokenized_done_paradigm = self.tokenizer(self.paradigm_done_str)
		self.tokenized_sep_paradigm = self.tokenizer(self.paradigm_sep_str)
		return super_return_value



	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)
	
	def __call__(self, input_ids, attempts=None, unused_input_ids=None, **kwargs,):
		if attempts is None:
			attempts = self.repeat_attempts

		
		if self.maximum_sequence_length < len(input_ids):
			unused_input_ids = input_ids[self.maximum_sequence_length:]
			input_ids = input_ids[:self.maximum_sequence_length]


		span_lengths = self.np_rng.choice(self.possible_span_lengths, size=self.num_spans, replace=True)
		start_token_lengths = self.np_rng.choice(self.possible_start_token_lenmgths, size=self.num_spans, replace=True)
		end_token_lengths = self.np_rng.choice(self.possible_end_token_lengths, size=self.num_spans, replace=True)
		total_num_aux_tokens = (np.sum(span_lengths) + np.sum(start_token_lengths) + np.sum(end_token_lengths))
		total_num_tokens = (len(input_ids) + total_num_aux_tokens)

		remaining_space = int(kwargs.get("remaining_space", 0))


		if (remaining_space < total_num_tokens):
			# slice into unused_input_ids, if valid
			diff = (total_num_tokens - remaining_space)
			if unused_input_ids is not None:
				if (len(unused_input_ids) >= diff):
					input_ids = unused_input_ids[:diff]
					unused_input_ids = unused_input_ids[diff:]
				else:
					return dict(
						status="error",
						message=f"The remaining space is not large enough to accomodate the selected spans. (remaining_space: {remaining_space}, total_num_tokens: {total_num_tokens})",
					)

		if (remaining_space < total_num_tokens):
			if (attempts > 0):
				return self(input_ids, attempts=(attempts - 1), unused_input_ids=unused_input_ids, **kwargs,)
			else:
				return dict(
					status="error",
					message=f"The remaining space is not large enough to accomodate the selected spans. (remaining_space: {remaining_space}, total_num_tokens: {total_num_tokens})",
				)


		# run checks
		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status


		
		if (len(input_ids) < total_num_aux_tokens):
			return dict(
				status="error",
				message=f"The input sequence is not long enough to accomodate the selected span lengths without overlapping. (len(input_ids): {len(input_ids)}, total_num_aux_tokens: {total_num_aux_tokens})",
			)
		
		
		# find "self.num_spans" contiguous spans of length "span_lengths[idx]"
		# starting from "start_span" and ending at "end_span".
		# The spans cannot overlap.

		shortest_start_token_length_idx = np.argmin(start_token_lengths)
		shortest_start_token_length = start_token_lengths[shortest_start_token_length_idx]

		shortest_end_token_length_idx = np.argmin(end_token_lengths)
		shortest_end_token_length = end_token_lengths[shortest_end_token_length_idx]

		shortest_span_length_idx = np.argmin(span_lengths)
		shortest_span_length = span_lengths[shortest_span_length_idx]

		possible_span_start_locations = np.arange(shortest_start_token_length, (len(input_ids) - shortest_end_token_length - shortest_span_length + 1))
		used_locations = np.zeros(len(input_ids), dtype=np.int32)

		found_spans = []
		

		for idx in range(self.span_finding_attempts):
			idx = (idx % self.num_spans)

			start_span = self.np_rng.choice(possible_span_start_locations, size=1, replace=False)[0]

			span_length = span_lengths[idx]
			start_length = start_token_lengths[idx]
			end_length = end_token_lengths[idx]

			end_span = (start_span + span_length + start_length + end_length)

			# check if the new span overlaps with any of the existing spans
			if np.any(used_locations[start_span:end_span]):
				continue

			# mark the new span as used
			used_locations[start_span : end_span] = 1

			# add the new span to the list of found spans
			selected_super_span = input_ids[start_span : end_span]

			start_tokens = selected_super_span[:start_length]
			end_tokens = selected_super_span[-end_length:]
			selected_span = selected_super_span[start_length : -end_length]

			found_spans.append(
				dict(
					start_tokens=start_tokens,
					end_tokens=end_tokens,
					selected_span=selected_span,
				)
			)

			if (len(found_spans) == self.num_spans):
				break

		if (len(found_spans) != self.num_spans):

			if (attempts > 0):
				return self(input_ids, attempts=(attempts - 1), unused_input_ids=unused_input_ids, **kwargs,)
			else:
				return dict(
					status="error",
					message=f"Could not find enough non-overlapping spans. (len(found_spans): {len(found_spans)}, self.num_spans: {self.num_spans})",
				)
		
		constructed_instructions = [
			self.tokenized_function_paradigm,
		]
		constructed_label = []

		for found_spans_idx, (_found_spans) in enumerate(found_spans):
			start_tokens = _found_spans['start_tokens']
			end_tokens = _found_spans['end_tokens']
			selected_span = _found_spans['selected_span']

			constructed_label.extend([selected_span, self.tokenized_sep_paradigm])
			constructed_instructions.extend([self.tokenized_start_paradigm, start_tokens, self.tokenized_end_paradigm, end_tokens,])

		constructed_instructions = np.concatenate(constructed_instructions)

		constructed_context = np.concatenate([self.tokenized_context_paradigm, input_ids])
			

		constructed_label = constructed_label[:-1] # remove the extra separator
		constructed_label.append(self.tokenized_done_paradigm)
		constructed_label = np.concatenate(constructed_label)

		if self.formatting_type == "query_context":
			input_ids = np.concatenate([constructed_instructions, constructed_context, [self.special_token_ids.get('start_generating_id', 1),],],)

		elif self.formatting_type == "context_query":
			input_ids = np.concatenate([constructed_context, constructed_instructions, [self.special_token_ids.get('start_generating_id', 1),],],)
		
		else:
			raise NotImplementedError(f"Unknown formatting_type: {self.formatting_type}")

		label_ids = constructed_label

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
			unused_input_ids=unused_input_ids,
		)		


		









@dataclasses.dataclass
class InfillingConfig(ObjectiveConfig):
	corruption_rate: float = 0.15
	mean_span_width: int = 3
	mask_token_ids: typing.List[int] = dataclasses.field(default_factory=list)
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Infilling'
		self.np_rng = np.random.default_rng(self.rng_seed)

		## Patched to use the API from the original paper code.
		## Will refactor soon.
		self.corruption_config = _span.SpanCorruptionConfig(
			mean_span_width=self.mean_span_width,
			mean_corruption_percentage=self.corruption_rate,
			use_autoencoding_objective=False,
		)

	
	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)
	
	def __call__(self, input_ids, **kwargs,):
		'''
		'''

		remaining_space = int(kwargs.get("remaining_space", 0))

		
		unused_input_ids = None
		if len(input_ids) < remaining_space:
			unused_input_ids = input_ids[remaining_space-128:]
			input_ids = input_ids[:remaining_space-128]

		if self.maximum_sequence_length < len(input_ids):
			if unused_input_ids is not None:
				unused_input_ids = np.concatenate([unused_input_ids, input_ids[self.maximum_sequence_length:],])
			else:
				unused_input_ids = input_ids[self.maximum_sequence_length:]
			input_ids = input_ids[:self.maximum_sequence_length]


		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
			
		
		corruption_instance = _span.SpanCorruptionInstance(
			inputs=input_ids,
			max_allowed_length=int(kwargs.get("remaining_space", 0)),
			rng_seed=self.np_rng.integers(0, 2**32, size=1)[0],
			corruption_config=self.corruption_config,
		)
		
		result = _span.find_valid_result(corruption_instance = corruption_instance)

		input_ids = result['inputs']
		label_ids = result['labels']

		input_ids = np.concatenate([input_ids, [self.special_token_ids.get('start_generating_id', 1),],],)

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
			unused_input_ids=unused_input_ids
		)






@dataclasses.dataclass
class AutoencodingConfig(ObjectiveConfig):
	corruption_rate: float = 0.15
	mean_span_width: int = 3
	mask_token_ids: typing.List[int] = dataclasses.field(default_factory=list)
	minimum_sequence_length: int = -1
	maximum_sequence_length: int = -1
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	shuffle: bool = False
	special_token_ids: typing.Dict[str, int] = dataclasses.field(default_factory=dict)
	rng_seed: int = dataclasses.field(default_factory=int)

	sentinel_start_id: int = 31900
	sentinel_end_id: int = 32000

	def __post_init__(self):
		if len(self.objective) == 0:
			self.objective = 'Autoencoding'
		self.np_rng = np.random.default_rng(self.rng_seed)

		## Patched to use the API from the original paper code.
		## Will refactor soon.
		self.corruption_config = _span.SpanCorruptionConfig(
			mean_span_width=self.mean_span_width,
			mean_corruption_percentage=self.corruption_rate,
			use_autoencoding_objective=True,
		)

	
	def run_checks(self, input_ids):
		'''
		'''
		if (self.minimum_sequence_length != -1) and len(input_ids) < self.minimum_sequence_length:
			return dict(
				status="error",
				message=f"The input sequence is not long enough. (len(input_ids): {len(input_ids)}, minimum_sequence_length: {self.minimum_sequence_length})",
			)
		
		if (self.maximum_sequence_length != -1) and (self.maximum_sequence_length < len(input_ids)):
			return dict(
				status="error",
				message=f"The input sequence is too long. (len(input_ids): {len(input_ids)}, maximum_sequence_length: {self.maximum_sequence_length})",
			)
		
		return dict(
			status="ok",
		)
	
	def __call__(self, input_ids, **kwargs,):
		'''
		'''

		remaining_space = int(kwargs.get("remaining_space", 0))
		unused_input_ids = None
		if len(input_ids) < remaining_space:
			unused_input_ids = input_ids[remaining_space-128:]
			input_ids = input_ids[:remaining_space-128]

		if self.maximum_sequence_length < len(input_ids):
			if unused_input_ids is not None:
				unused_input_ids = np.concatenate([unused_input_ids, input_ids[self.maximum_sequence_length:],])
			else:
				unused_input_ids = input_ids[self.maximum_sequence_length:]
			input_ids = input_ids[:self.maximum_sequence_length]



		check_status = self.run_checks(input_ids)
		if check_status["status"] != "ok":
			return check_status
		
		corruption_instance = _span.SpanCorruptionInstance(
			inputs=input_ids,
			max_allowed_length=int(kwargs.get("remaining_space", 0)),
			rng_seed=self.np_rng.integers(0, 2**32, size=1)[0],
			corruption_config=self.corruption_config,
		)
		
		result = _span.find_valid_result(corruption_instance = corruption_instance)

		input_ids = result['inputs']
		label_ids = result['labels']

		if self.shuffle:
			locs0 = np.where(input_ids >= self.sentinel_start_id)[0]
			locs1 = np.where(input_ids < self.sentinel_end_id)[0]
			# locs = union
			locs = np.intersect1d(locs0, locs1)

			# make numpy copy of inputs and labels
			input_ids = np.int32(input_ids.tolist())

			input_arrays = []
			rolling_inputs = input_ids
			for loc in locs:
				input_arrays.append(np.int32(rolling_inputs[:loc]))
				rolling_inputs = rolling_inputs[loc:]

			self.np_rng.shuffle(input_arrays)
			input_ids = np.concatenate(input_arrays)
			

		input_ids = np.concatenate([input_ids, [self.special_token_ids.get('start_generating_id', 1),],],)

		return dict(
			status="ok",
			input_ids=input_ids,
			label_ids=label_ids,
			objective=self.objective,
			unused_input_ids=unused_input_ids,
		)









if __name__ == "__main__":

	names = [
		"Testing_Name_000",
		"Another_Alias_111",
		"",
	]

	seen_objectives_counter = {}



	for name in names:

		x = NextTokenPredictionConfig(
			objective=name,
		)
		seen_objectives_counter[x.objective] = seen_objectives_counter.get(x.objective, 0) + 1



	for seen_objectives_counter_idx, (key, value) in enumerate(seen_objectives_counter.items()):
		print(f"  seen_objectives_counter[{key}]: {value}")



	for seen_objectives_counter_idx, (key, value) in enumerate(seen_objectives_counter.items()):
		assert(value == 1), f"  Error: {key} was seen {value} times. Expected 1."
		
		


