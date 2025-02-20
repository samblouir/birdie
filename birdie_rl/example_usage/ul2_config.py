'''
	This contains:
	- A dummy config
	- The configuration for the UL2 objectives
	- The configuration for the UL2 objectives specialized for a decoder-only Transformer

Empirical observation notes:
	The SSMs seemed to strongly overfit to specific objectives.
	The UL2 decoder-only speccialization's infilling only includes:
	- 15% corruption + 3 token mean span width
	- 50% corruption + 32 token mean span width
	In Birdie, we found better performance add more objectives in-between the two (and adding more objectives in general.)

'''

dummy_config = [
	{
		"name": 'infilling',
		"prob": 1.0, # These sampling probabilities will be normalized automatically
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": 3,
	},
	{
		"name": 'next_token_prediction',
		"prob": 1.0, # These sampling probabilities will be normalized automatically
		"paradigm_token": "[C]",
	}
]


'''
	UL2 + Variation
'''
# Rough estimate from English data from ByT5, you may want to adjust this based on your domain and tokenizer.
# ByT5: https://arxiv.org/abs/2105.13626
multipler_for_operating_on_characters_instead_of_tokens = 4.0

# From UL2: https://arxiv.org/abs/2205.05131
ul2_config = [
	# S-Denoiser
	{
		"name": 'prefix_language_modeling',
		"prob": 1.0,
		"corruption_rate": 0.15,
		"prefix_fraction": 0.75,
		"paradigm_token": "[S]",
	},

	## R-denoisers
	{
		"name": 'infilling',
		"prob": 0.5,
		"corruption_rate": 0.15,
		"paradigm_token": "[R]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	},
	{
		"name": 'infilling',
		"prob": 0.5,
		"corruption_rate": 0.15,
		"paradigm_token": "[R]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 9,
	},

	## X-denoisers
	{
		"name": 'infilling',
		"prob": 0.25,
		"corruption_rate": 0.5,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	},
	{
		"name": 'infilling',
		"prob": 0.25,
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 8,
	},
	{
		"name": 'infilling',
		"prob": 0.25,
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	},
	{
		"name": 'infilling',
		"prob": 0.25,
		"corruption_rate": 0.50,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	},

]

# UL2 specialized for decoder-only models
# From "The unreasonable effectiveness of few-shot learning for machine translation": https://arxiv.org/pdf/2302.01398
# See section 2.2: "Data pre-processing and training objective"
'''
	> ...In this work, we use 2 (instead of 6) separate span
	> corruption instances with (noise density, mean noise span
	> length) given by (0.15, 3) and (0.5, 32) respectively. In addition to these two objectives and the prefix language modeling objective, we also include a standard causal language
	> modeling objective. We mix these objectives randomly,
	> sampling prefix language modeling 20% of the time, causal
	> language modeling 60% of the time, and the remaining span
	> corruption instances 20% of the time...
'''

ul2_decoder_config = [
	## C-Denoiser
	{
		"name": 'next_token_prediction',
		"prob": 0.6,
		"paradigm_token": "[C]",
	},
	## S-Denoiser
	{
		"name": 'prefix_language_modeling',
		"prob": 0.2,
		"prefix_fraction": 0.75,
		"paradigm_token": "[S]",
	},
	## R-denoisers
	{
		"name": 'infilling',
		"prob": 0.1,
		"corruption_rate": 0.15,
		"paradigm_token": "[R]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	},
	{
		"name": 'infilling',
		"prob": 0.1,
		"corruption_rate": 0.5,
		"paradigm_token": "[R]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 32,
	},
]