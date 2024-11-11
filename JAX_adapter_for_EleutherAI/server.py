'''

This adapter is split off from Birdie's codebase.
It was created to use the EleutherAI LM Harness with JAX models.
Currently, it only supports max-likelihood/most multiple choice tasks, like ARC and MMLU.
It should be easy to extend this to support other tasks.

Here is a quick overview of the two parts:

1. A server that listens for requests, loads a model, makes predictions, and sends the results back.
	- Deserialize the request
	- Load the requested model
	- Run the model on the inputs
	- Serialize the output
	- Send the output back to the EleutherLM Harness

2. A custom model.py in EleutherLM Harness:
	- Takes data and other arguments from the EleutherLM Harness
	- Sends a request to the server
	- Receives the response
	- Returns the response to the EleutherLM Harness

Notes:
- JAX's official method to reset allocated GPU VRAM was causing segmenation faults. So, if the server already has a model loaded, and is asked to load a different model, the server process simply exits. For ease-of-use, I simply recursively call the server.py file with an outer for-loop, effectively automatically restarting it.

Usage:

Start the server. It should print messages.

You will have to git clone the Eleuther harness:
	Please see: https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install
Then, you must move birdie.py to lm-evaluation-harness/lm_eval/models/birdie.py.

This script uses Flask to launch a server that listens for a request from the EleutherLM Harness.

What this code does is:
	1. Launch the server
	2. Receive a request from the EleutherLM Harness
	3. This request contains the desired model tag and the inputs
	4. If we have not loaded a model:
		- Load the model
		- Store the model tag
	5. If we have already loaded a model, and it does not match the requested model tag:
		- Exit the program and let an outside script automatically call this again.
	6. Make a prediction using the model
	7. Send the prediction back to the EleutherLM Harness, which handles the rest from there.

'''

from flask import Flask, request, Response
import dill
import api
import os
import sys
import traceback
import jax


print_once = True
pretty_name = __file__.rsplit("/", 1)[-1]

model = None 

app = Flask(__name__)
@app.route('/data', methods=['POST'])
def data():
	global wrapper
	global print_once

	if print_once:
		print(f"  {pretty_name}:  Ready!")
		print_once = False


	try:
		# Get the raw data from get EleutherLM harness
		raw_data = request.data

		# Deserialize the data using dill.
		kwargs = dill.loads(raw_data)

		# Check for there being any inputs at all.
		if (len(kwargs.get("inputs", [])) == 0):
			print(f'  ERROR! Empty input! Sending back an empty list...')
			y = dill.dumps([])
			return Response(y, status=200, mimetype='application/octet-stream')
		
		if model is None:
			model = api.ModelWrapper(**kwargs)
			loaded_model_tag = kwargs['model_tag']
			
		if loaded_model_tag != kwargs['model_tag']:
			print(f'  INFO: The requested model tag is not what is loaded. Requested: {kwargs["model_tag"]}, Loaded: {loaded_model_tag}')

			# If this hangs, try using the os._exit() below to force-exit.
			os.exit(0)
			# os._exit(1) # Dirty exit

			## Segfault when using JAX's built-in memory reset, but this may work.
			# del model
			# model = server_dest.get_model(**kwargs)
			# loaded_model_tag = kwargs['model_tag']

		# This function accepts the 
		rv = model(**kwargs,)

		x = dill.dumps(rv)

		return Response(x, status=200, mimetype='application/octet-stream')
	except Exception as e:

		print(f'\n'*5)
		print(f"#"*60)
		print(f"#"*60)
		print(f"  server_host.py:  Exception! e: {e}")
		print(f"#"*60)
		traceback.print_exc()
		print(f"#"*60)
		os._exit(1)
		return Response(f"Exception! e: {e}", status=400)
	finally:
		print(f"  server_host.py:  finally block: Done!")
		
##################################################
	




	
if __name__ == '__main__':
	model = None
	loaded_model_tag = None
	# cache_responses = False

	## Read in cmd-line arguments
	arg_dict = {}
	for arg in sys.argv:
		if "=" in arg:
			key, val = arg.split("=")
			arg_dict[key] = int(val)
	# safe
	cuda_visible_devices = arg_dict.get('cuda_visible_devices', 0)
	port = int(arg_dict.get('port', '5000'))
	
	if (cuda_visible_devices != -1):
		os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
		os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
		os.environ['JAX_PLATFORM_NAME'] = 'gpu'

		# TODO: Will have to adjust for TPUs.
		assert("cuda" in str(jax.devices()).lower())
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = ""
		os.environ['JAX_PLATFORM_NAME'] = 'cpu'

	print(f"  Starting a server using these jax.devices(): {jax.devices()}")

	# Starts the server
	try:
		app.run(debug=False,  threaded=False, port=port,)
	except KeyboardInterrupt:
		print("Shutting down the server...")
















###################################################