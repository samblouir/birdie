
'''
    Birdie
	(light with fewer variations)
'''

# This multiplier is a placeholder. Adjust if using character-level tokenization
# where objectives expect token-level counts for parameters like 'tokens_per_mask'.
# For the objectives below, it's assumed they operate primarily at the token level,
# so this multiplier is set to 1.0.
multipler_for_operating_on_characters_instead_of_tokens = 1.0

birdie_light_config = [


	# # S-Denoiser
	# {
	# 	"name": 'prefix_language_modeling',
	# 	# "prob": 1.0,
	# 	"corruption_rate": 0.15,
	# 	"prefix_fraction": 0.75,
	# 	"paradigm_token": "[S]",
	# },

	# ## R-denoisers
	# {
	# 	"name": 'infilling',
	# 	# "prob": 0.5,
	# 	"corruption_rate": 0.15,
	# 	"paradigm_token": "[R]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	# },
	# {
	# 	"name": 'infilling',
	# 	"prob": 0.5,
	# 	"corruption_rate": 0.15,
	# 	"paradigm_token": "[R]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 9,
	# },

	# ## X-denoisers
	# {
	# 	"name": 'infilling',
	# 	# "prob": 0.25,
	# 	"corruption_rate": 0.5,
	# 	"paradigm_token": "[X]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	# },
	# {
	# 	"name": 'infilling',
	# 	# "prob": 0.25,
	# 	"corruption_rate": 0.15,
	# 	"paradigm_token": "[X]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 8,
	# },
	# {
	# 	"name": 'infilling',
	# 	# "prob": 0.25,
	# 	"corruption_rate": 0.15,
	# 	"paradigm_token": "[X]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	# },
	# {
	# 	"name": 'infilling',
	# 	# "prob": 0.25,
	# 	"corruption_rate": 0.50,
	# 	"paradigm_token": "[X]",
	# 	"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	# },


	# ## C-Denoiser
	# {
	# 	"name": 'next_token_prediction',
	# 	# "prob": 0.6,
	# 	"paradigm_token": "[C]",
	# },


	# Selective Copying
	{
		"name": 'selective_copying',
		# "prob": 0.33,  # Example probability, adjust as needed
		# "corruption_rate": 0.5,
		"tokens_per_mask": int(16 * multipler_for_operating_on_characters_instead_of_tokens), # Average length of a span to be copied
		"shuffle": True,  # Shuffle the order of "find" instructions and "result" blocks
		"min_delimiter_prefix_length": 2,
		"max_delimiter_prefix_length": 16,
		"min_delimiter_suffix_length": 2,
		"max_delimiter_suffix_length": 16,
		"format_style": "context_query",  # Can be "query_context" or "context_query"
	},


	# Selective Copying
	{
		"name": 'selective_copying',
		# "prob": 0.33,  # Example probability, adjust as needed
		# "corruption_rate": 0.5,
		"tokens_per_mask": int(16 * multipler_for_operating_on_characters_instead_of_tokens), # Average length of a span to be copied
		"shuffle": True,  # Shuffle the order of "find" instructions and "result" blocks
		"min_delimiter_prefix_length": 2,
		"max_delimiter_prefix_length": 16,
		"min_delimiter_suffix_length": 2,
		"max_delimiter_suffix_length": 16,
		"format_style": "query_context",  # Can be "query_context" or "context_query"
	},


	# Selective Copying
	{
		"name": 'selective_copying',
		# "prob": 0.33,  # Example probability, adjust as needed
		# "corruption_rate": 0.5,
		"tokens_per_mask": int(8 * multipler_for_operating_on_characters_instead_of_tokens), # Average length of a span to be copied
		"shuffle": True,  # Shuffle the order of "find" instructions and "result" blocks
		"min_delimiter_prefix_length": 2,
		"max_delimiter_prefix_length": 16,
		"min_delimiter_suffix_length": 2,
		"max_delimiter_suffix_length": 16,
		"format_style": "context_query",  # Can be "query_context" or "context_query"
	},


	# Selective Copying
	{
		"name": 'selective_copying',
		# "prob": 0.33,  # Example probability, adjust as needed
		# "corruption_rate": 0.5,
		"tokens_per_mask": int(8 * multipler_for_operating_on_characters_instead_of_tokens), # Average length of a span to be copied
		"shuffle": True,  # Shuffle the order of "find" instructions and "result" blocks
		"min_delimiter_prefix_length": 2,
		"max_delimiter_prefix_length": 16,
		"min_delimiter_suffix_length": 2,
		"max_delimiter_suffix_length": 16,
		"format_style": "query_context",  # Can be "query_context" or "context_query"
	},

	# # Copying
	# {
	# 	"name": 'copying',
	# 	# "prob": 0.33,  # Example probability
	# 	"paradigm": "<|COPY|>", # Default from CopyingConfig
	# 	"paradigm_suffix": "",   # Default from CopyingConfig
	# },

	# # Deshuffling
	# {
	# 	"name": 'deshuffling',
	# 	# "prob": 0.34,  # Example probability
	# 	"paradigm": "<|DESHUFFLE|>", # Default from DeshufflingConfig
	# 	"paradigm_suffix": "",       # Default from DeshufflingConfig
	# 	"percentage_to_shuffle": 0.75, # Default from DeshufflingConfig
	# },

	# # Deshuffling
	# {
	# 	"name": 'deshuffling',
	# 	# "prob": 0.34,  # Example probability
	# 	"paradigm": "<|DESHUFFLE|>", # Default from DeshufflingConfig
	# 	"paradigm_suffix": "",       # Default from DeshufflingConfig
	# 	"percentage_to_shuffle": 0.25, # Default from DeshufflingConfig
	# },


]

