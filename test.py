#!/usr/bin/python

import numpy as np
import unittest
import ANN

class Test_basic_ANN(unittest.TestCase):
	def test_init(self):
		network = ANN.basic_ANN(2, 3, 2)
		self.assertTrue(network.input_weights.shape == (3, 2))
		self.assertTrue(network.output_weights.shape == (2, 3))
		for x in network.input_weights:
			for y in x:
				self.assertTrue(y != 0)
		for x in network.output_weights:
			for y in x:
				self.assertTrue(y != 0)

	def test_basic_ANN_feed_forward(self):
		network = ANN.basic_ANN(2, 3, 2)
		test_output = network.feed_forward(np.array([1, 0]))
		print test_output


def main():
	print("Testing")
#	np.__config__.show()
	unittest.main()

if __name__=="__main__":
	main()