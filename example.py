import tokenizer as huggingface_tokenizer
from birdie.all_in_one import Birdie


model = "Your Model"
tokenizer = huggingface_tokenizer.get_tokenizer("Your Tokenizer")



from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", streaming=True,)


config = {
	"tokenizer": tokenizer,
	"pretraining_objectives": "birdie",

	"total_steps": 10_000,
	"steps_between_evaluations": 500,

	"ds_train": ds['train'],
	"ds_validation": ds['validation'],
}

birdie = Birdie(config)

for step_idx in range(10_000):

	if birdie.time_for_eval():
		model.eval()
		for x in birdie.get_validation_samples():
			preds = model(x['input_ids'])
			loss = cross_entropy(preds, x['labels'])
			birdie.update(loss)

		validation_losses = birdie.get_validation_losses()
	
	x = birdie.get_next_training_sample()

	preds = model(x['input_ids'])
	loss = cross_entropy(preds, x['labels'], x['loss_mask'])

	loss.backward()
	optimizer.step()


