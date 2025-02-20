

multipler_for_operating_on_characters_instead_of_tokens = 4.0

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
		"prob": 0.5,
		"corruption_rate": 0.5,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 3,
	},
	{
		"name": 'infilling',
		"prob": 0.5,
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 8,
	},
	{
		"name": 'infilling',
		"prob": 0.5,
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	},
	{
		"name": 'infilling',
		"prob": 0.5,
		"corruption_rate": 0.15,
		"paradigm_token": "[X]",
		"mean_tokens_per_span": multipler_for_operating_on_characters_instead_of_tokens * 64,
	},

]