import numpy as np
from functools import partial
import copy
from objective_configs import (
	NextTokenPredictionConfig,
	PrefixLanguageModelingConfig,
	InfillingConfig,
	AutoencodingConfig,
	CopyingConfig,
	DeshufflingConfig,
)

from birdie_objectives import (
	_birdie_autoencoding,
	_birdie_infilling,
	_birdie_copying,
	_birdie_deshuffling,
	_birdie_selective_copying,
	_birdie_next_token_prediction,
	_birdie_prefix_language_modeling,
	_birdie_utils,
)



def get_objectives(config=None):
	## Stores all configured objectives
	pretraining_objectives = {}

	autoencoding_objectives = _birdie_autoencoding.get_autoencoding_objectives(config)
	copying_objectives = _birdie_copying.get_objectives(config)
	deshuffling_objectives = _birdie_deshuffling.get_objectives(config)
	infilling_objectives = _birdie_infilling.get_infilling_objectives(config)
	next_token_prediction_objectives = _birdie_next_token_prediction.get_objectives(config)
	prefix_language_modeling_objectives = _birdie_prefix_language_modeling.get_objectives(config)
	selective_copying_objectives = _birdie_selective_copying.get_objectives(config)

	pretraining_objectives.update(autoencoding_objectives)
	pretraining_objectives.update(copying_objectives)
	pretraining_objectives.update(deshuffling_objectives)
	pretraining_objectives.update(infilling_objectives)
	pretraining_objectives.update(next_token_prediction_objectives)
	pretraining_objectives.update(prefix_language_modeling_objectives)
	pretraining_objectives.update(selective_copying_objectives)

	pretraining_objectives = dict(sorted(pretraining_objectives.items()))

	for pretraining_objectives_idx, (key, value) in enumerate(pretraining_objectives.items()):
		pretraining_objectives[key]['config'].set_tokenizer(config['tokenizer'])
		

	assert(len(pretraining_objectives) > 0)
	return pretraining_objectives



def apply(config, inplace=False,):

	assert('tokenizer' in config), f"Please ensure a callable 'tokenizer' is in the config before loading the birdie objectives"

	if not inplace:
		config = copy.deepcopy(config)

	config['autoencoding_mask_token_ids'] = np.arange(config['autoencoding_sentinel_token_id_start'], config['autoencoding_sentinel_token_id_end'],)
	config['infilling_mask_token_ids'] = np.arange(config['infilling_sentinel_token_id_start'], config['infilling_sentinel_token_id_end'],)

	config['pretraining_objectives'] = get_objectives(config=config)
	config['reward_scaling_vector'] = _birdie_utils.calculate_reward_scaling_vector(config=config)


	indexed_pretraining_objectives = []
	for pretraining_objectives_idx, (key, value) in enumerate(config['pretraining_objectives'].items()):
		value['specific_name'] = key
		value['coarse_name'] = value['config'].objective
		value['fn'] = value['config']
		del value['config']
		indexed_pretraining_objectives.append(value)
	config['indexed_pretraining_objectives'] = indexed_pretraining_objectives
	config['indexed_unique_objective_ids'] = _birdie_utils.get_unique_objective_indices(config)


	return config








if __name__ == "__main__":

	story = '''Once upon a time, in a serene corner of a bustling town, there lived a small bird named Beep. Beep was a delicate creature with feathers that shimmered in shades of blue and green, catching the light of the sun in a mesmerizing display. Despite his vibrant appearance, Beep felt a profound sense of loneliness. He resided in a large, dry cage that stood alone in a quiet backyard, surrounded by tall, whispering trees and blooming flowers. The cage, though spacious, felt more like a vast prison to Beep, its metal bars cold and uninviting against the backdrop of nature's beauty.

Every morning, as the first rays of sunlight filtered through the leaves, Beep would perch on his favorite spot near the top of the cage. From there, he could see the expansive sky stretching endlessly above him and the lush greenery that surrounded his enclosure. Beep longed to spread his wings and soar freely among the clouds, to feel the wind beneath his feathers, and to sing joyously with other birds. However, the cage's sturdy construction made escape impossible, and Beep's dreams of freedom remained unfulfilled.

Determined to change his fate, Beep spent his days tweeting and singing heartfelt melodies. His songs were filled with longing and hope, each note a silent plea for liberation. He believed that his persistent singing might attract the attention of other birds or, perhaps, a kind-hearted human who could help him find his way out. Days turned into weeks, and weeks into months, but despite his unwavering efforts, Beep remained trapped within the confines of his dry, solitary cage.

As the seasons changed, so did Beep's surroundings. The once vibrant flowers began to wilt, and the trees shed their leaves, leaving the backyard feeling colder and more desolate. Beep's cage, too, seemed to lose its luster, becoming an ever-present reminder of his captivity. His songs, though still beautiful, carried a hint of despair, echoing through the empty space around him.

One chilly autumn afternoon, as Beep was singing a particularly melancholic tune, a gentle breeze carried his song beyond the confines of the backyard. Unbeknownst to Beep, a kind person named Emma was taking her usual evening stroll through the neighborhood. Emma had always loved birds and found solace in their songs, appreciating the simple beauty they brought to her daily walks. As she passed by the backyard, Beep's heartfelt melodies reached her ears, stirring something deep within her.

Moved by the sorrow she sensed in Beep's song, Emma decided to investigate. She approached the cage, her footsteps soft on the gravel path. Observing Beep's graceful movements and hearing his plaintive cries, Emma felt a surge of compassion. She knew she had to help this lonely bird find the freedom he so desperately desired.

Emma reached into her bag and retrieved a small key—a sentimental keepsake from her childhood pet bird, which had taught her the importance of kindness and empathy towards all living creatures. With a gentle touch, she inserted the key into the lock of Beep's cage and turned it slowly. The lock clicked open, and the cage door swung ajar, revealing the bright blue sky and the vast expanse beyond.

Beep hesitated for a moment, his heart pounding with a mix of fear and excitement. He had dreamed of this moment for so long, yet the reality of it felt surreal. Taking a deep breath, he spread his wings wide, feeling the cool air beneath them for the first time. With a burst of energy, Beep took flight, soaring upwards into the open sky. The sensation of freedom was exhilarating, and Beep's joyful tweets filled the air as he circled above Emma, expressing his gratitude through song.

Seeing that no other birds were trapped within the cage, Emma decided to take an extra precaution. She carefully placed a sturdy lock on the cage, ensuring that no other bird would fall victim to the same fate. This act of kindness transformed the cage from a symbol of confinement to one of protection, safeguarding other birds from potential captivity.

As Beep flew higher, he felt an overwhelming sense of joy and liberation. The world below was a tapestry of vibrant colors and diverse landscapes, each new sight more breathtaking than the last. He joined a flock of other birds, each with their own stories of freedom and adventure. Together, they danced through the sky, their songs blending into a harmonious symphony that celebrated their newfound liberty.

Emma watched Beep's departure with a heart full of happiness. She knew that her simple act of opening the cage had changed Beep's life forever. Inspired by this experience, Emma made it her mission to help other birds in need. She began visiting various neighborhoods, searching for cages and aviaries where birds were confined. With each lock she opened, she brought more joy and freedom to countless birds, transforming her community into a haven where avian life could flourish unhindered.

Beep thrived in his freedom, exploring every nook and cranny of the expansive world around him. He discovered hidden gardens, sparkling streams, and towering trees that seemed to touch the heavens. Along his journey, Beep met other birds who shared their own tales of escape and resilience. These friendships enriched his life, filling the void that loneliness had once carved within him.

Occasionally, Beep would return to Emma's backyard, not to the cage that once held him captive, but to visit the place where his freedom began. He would perch on the same spot, now a symbol of hope and liberation, and sing a special song of gratitude. Emma would greet him with a warm smile, her eyes reflecting the bond they shared through their mutual love and respect for all creatures.

Years passed, and Beep's legacy of freedom inspired others to act with kindness and empathy. The once-dry cage became a cherished landmark, representing the triumph of compassion over confinement. Birds from all walks of life flocked to the backyard, finding solace and safety in the space that had been transformed by Emma's unwavering dedication.

Beep's story spread far and wide, touching the hearts of people everywhere. It served as a poignant reminder that even the smallest acts of kindness can have profound and lasting impacts. Through his journey from a lonely bird in a dry cage to a free spirit soaring in the bright blue sky, Beep embodied the essence of hope, resilience, and the enduring power of compassion.

In the end, Beep never forgot the kind person who had given him his freedom. Their bond, though unspoken, remained a testament to the connection between humans and nature, a connection built on empathy and the shared desire for all living beings to live freely and happily. Beep's life was a beautiful symphony of freedom and friendship, forever grateful to the kind soul who had unlocked the door to his dreams.'''


	story = '''Every morning, as the first rays of sunlight filtered through the leaves, Beep would perch on his favorite spot near the top of the cage. From there, he could see the expansive sky stretching endlessly above him and the lush greenery that surrounded his enclosure. Beep longed to spread his wings and soar freely among the clouds, to feel the wind beneath his feathers, and to sing joyously with other birds. However, the cage's sturdy construction made escape impossible, and Beep's dreams of freedom remained unfulfilled.

Determined to change his fate, Beep spent his days tweeting and singing heartfelt melodies. His songs were filled with longing and hope, each note a silent plea for liberation. He believed that his persistent singing might attract the attention of other birds or, perhaps, a kind-hearted human who could help him find his way out. Days turned into weeks, and weeks into months, but despite his unwavering efforts, Beep remained trapped within the confines of his dry, solitary cage.

As the seasons changed, so did Beep's surroundings. The once vibrant flowers began to wilt, and the trees shed their leaves, leaving the backyard feeling colder and more desolate. Beep's cage, too, seemed to lose its luster, becoming an ever-present reminder of his captivity. His songs, though still beautiful, carried a hint of despair, echoing through the empty space around him.

One chilly autumn afternoon, as Beep was singing a particularly melancholic tune, a gentle breeze carried his song beyond the confines of the backyard. Unbeknownst to Beep, a kind person named Emma was taking her usual evening stroll through the neighborhood. Emma had always loved birds and found solace in their songs, appreciating the simple beauty they brought to her daily walks. As she passed by the backyard, Beep's heartfelt melodies reached her ears, stirring something deep within her.

Moved by the sorrow she sensed in Beep's song, Emma decided to investigate. She approached the cage, her footsteps soft on the gravel path. Observing Beep's graceful movements and hearing his plaintive cries, Emma felt a surge of compassion. She knew she had to help this lonely bird find the freedom he so desperately desired.

Emma reached into her bag and retrieved a small key—a sentimental keepsake from her childhood pet bird, which had taught her the importance of kindness and empathy towards all living creatures. With a gentle touch, she inserted the key into the lock of Beep's cage and turned it slowly. The lock clicked open, and the cage door swung ajar, revealing the bright blue sky and the vast expanse beyond.
Beep hesitated for a moment, his heart pounding with a mix of fear and excitement. He had dreamed of this moment for so long, yet the reality of it felt surreal. Taking a deep breath, he spread his wings wide, feeling the cool air beneath them for the first time. With a burst of energy, Beep took flight, soaring upwards into the open sky. The sensation of freedom was exhilarating, and Beep's joyful tweets filled the air as he circled above her, expressing his gratitude through song.

As Beep flew higher, he felt an overwhelming sense of joy and liberation. The world below was a tapestry of vibrant colors and diverse landscapes, each new sight more breathtaking than the last. He joined a flock of other birds, each with their own stories of freedom and adventure. Together, they danced through the sky, their songs blending into a harmonious symphony that celebrated their newfound liberty.

Beep thrived in his freedom, exploring every nook and cranny of the expansive world around him. He discovered hidden gardens, sparkling streams, and towering trees that seemed to touch the heavens. Along his journey, Beep met other birds who shared their own tales of escape and resilience. These friendships enriched his life, filling the void that loneliness had once carved within him.

Years passed, and Beep's legacy of freedom inspired others to act with kindness and empathy. The once-dry cage became a cherished landmark, representing the triumph of compassion over confinement. Beep's story spread far and wide, touching the hearts of people everywhere. It served as a poignant reminder that even the smallest acts of kindness can have profound and lasting impacts. Through his journey from a lonely bird in a dry cage to a free spirit soaring in the bright blue sky, Beep embodied the essence of hope, resilience, and the enduring power of compassion.

In the end, Beep never forgot the kind person who had given him his freedom. Their bond, though unspoken, remained a testament to the connection between humans and nature, a connection built on empathy and the shared desire for all living beings to live freely and happily. Beep's life was a beautiful symphony of freedom and friendship, forever grateful to the kind souls who had unlocked the door to his dreams.'''





	import os
	import sys
	sys.path.append(os.path.abspath('../'))
	sys.path.append(os.path.abspath('.'))

	import tokenizer as tokenizer_py
	tokenizer = tokenizer_py.Tokenizer()

	print(f"tokenizer: {tokenizer}")

	# text = "Hello. My name is Sam. What color is the sky?"
	# encoded = tokenizer.encode(text)

	dummy_cfg = {
		# "infilling_mask_token_ids": [11, 3333, 777,],
		# "autoencoding_mask_token_ids": [22, 444, 888,],

		"infilling_sentinel_token_id_start": 31900,
		"infilling_sentinel_token_id_end": 32000,

		"autoencoding_sentinel_token_id_start": 31900,
		"autoencoding_sentinel_token_id_end": 32000,

		"tokenizer": tokenizer,
	}

	dummy_cfg = apply(dummy_cfg)

	sampling_probabilities = [v['sampling_probability'] for v in dummy_cfg['pretraining_objectives'].values()]
	assert(np.isclose(np.sum(sampling_probabilities), 1.0)),  f"Exception (ValueError): 1 != np.sum(sampling_probabilities) == {np.sum(sampling_probabilities)}"

	reward_scaling_vector = dummy_cfg['reward_scaling_vector']
	sampling_probabilities_vector = _birdie_utils.get_sampling_probabilities_vector(dummy_cfg)

	print(f"\n  reward_scaling_vector {reward_scaling_vector.shape}: \n{reward_scaling_vector}\n")
	print(f"\n  sampling_probabilities_vector {sampling_probabilities_vector.shape}: \n{sampling_probabilities_vector}\n")


	encoded = tokenizer.encode(story)

	indexed_pretraining_objectives = dummy_cfg['indexed_pretraining_objectives']
	seen_coarse_names = dict()
	for indexed_pretraining_objectives_idx, (_indexed_pretraining_objectives) in enumerate(indexed_pretraining_objectives):

		callable_fn = _indexed_pretraining_objectives['fn']
		coarse_name = _indexed_pretraining_objectives['coarse_name']
		specific_name = _indexed_pretraining_objectives['specific_name']

		seen_coarse_names[coarse_name] = seen_coarse_names.get(coarse_name, 0) + 1


		result = callable_fn(encoded, remaining_space=2048,)
		if result['status'] != 'ok':
			print(f'  Skipping objective: {coarse_name} {specific_name}')
			for result_idx, (key, value) in enumerate(result.items()):
				print(f"  result[{key}]: {value}")
				
			continue

		input_ids = result['input_ids']
		label_ids = result['label_ids']

		if seen_coarse_names[coarse_name] > 1:
			continue

		print(f"")
		print(f"#" * 60,)
		print(f"  ## Objective name: {coarse_name}")
		print(f"  ### Objective settings: {specific_name}")
		print(f"  ### processed input_ids: {input_ids.shape}")
		print(f"  ### processed label_ids: {label_ids.shape}")
		print(f"\n")
		print(f"  tokenizer.decode(input_ids): {tokenizer.decode(input_ids)}")
		print(f"\n")
		print(f"  tokenizer.decode(label_ids): {tokenizer.decode(label_ids)}")
		print(f"\n")



		

'''



'''