import numpy as np


class basic_ANN:
	def __init__(self, input_n, hidden_n, output_n):
		self.input_nodes = input_n
		self.hidden_nodes = hidden_n
		self.output_nodes = output_n
		self.input_weights = np.random.randn(self.hidden_nodes, self.input_nodes);
		
