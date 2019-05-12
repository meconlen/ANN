import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))


class basic_ANN:
	# This sets up the network as a pair of weight arrays 

	def __init__(self, input_n, hidden_n, output_n):
		self.input_nodes = input_n
		self.hidden_nodes = hidden_n
		self.output_nodes = output_n
		self.input_weights = np.random.randn(self.hidden_nodes, self.input_nodes)
		self.output_weights = np.random.randn(self.output_nodes, self.hidden_nodes)


	# this takes an input array and outputs an output array

	def feed_forward(self, input):
		hidden_inputs = np.array(map(sigmoid, np.dot(self.input_weights, input)))
		# np.array(map(sigmoid, hidden_inputs))
		return hidden_inputs