import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))


class basic_ANN:
	# This sets up the network as a pair of weight arrays 

	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_weights = np.random.randn(hidden_nodes, input_nodes)
		self.output_weights = np.random.randn(output_nodes, hidden_nodes)


	# this takes an input array and outputs an output array

	def feed_forward(self, input):
		hidden_inputs = sigmoid(np.dot(self.input_weights, input.transpose()))
		outputs = sigmoid(np.dot(self.output_weights, hidden_inputs))
		return outputs
