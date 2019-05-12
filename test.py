#!/usr/bin/python

import numpy as np
import unittest
import ANN

class Test_basic_ANN(unittest.TestCase):
	def test_init(self):
		self.network = ANN.basic_ANN(2, 3, 2)
		self.assertTrue(self.network.input_nodes == 2)
		self.assertTrue(self.network.hidden_nodes == 3)
		self.assertTrue(self.network.output_nodes == 2)
		print "input weights"
		print self.network.input_weights
		print "output weights"
		print self.network.output_weights

		for x in self.network.input_weights:
			for y in x:
				self.assertTrue(y != 0)

		for x in self.network.output_weights:
			for y in x:
				self.assertTrue(y != 0)

def main():
	print("Testing")
#	np.__config__.show()
	unittest.main()

if __name__=="__main__":
	main()