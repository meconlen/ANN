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

	def test_train(self):
		print "test_train"
		network = ANN.basic_ANN(2, 3, 1)
		network.input_weights = np.array([[-0.24434436,  0.85612428], [ 0.70503219,  0.35294645], [ 0.32192263,  0.82835977]])
		network.output_weights = np.array([[-0.10350651, -0.21068824,  0.60715585]])
		print network.input_weights
		print network.output_weights
		print ""
		print ANN.sigmoid(network.input_weights)

		error =  network.train(np.array([0, 0]), 0)
		print error
		self.assertTrue(error - 0.536555 < 1e-5)
		error =  network.train(np.array([0, 1]), 0)
		print error
		self.assertTrue(error - 0.556309 < 0.00001)
		error =  network.train(np.array([1, 0]), 0)
		print error
		self.assertTrue(error - 0.541293 < 0.00001)
		error =  network.train(np.array([1, 1]), 1)
		print error
		self.assertTrue(error - 0.440861< 0.00001)
def main():
	print("Testing")
#	np.__config__.show()
	unittest.main()

if __name__=="__main__":
	main()