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
		print self.network.input_weights

def main():
	print("Testing")
#	np.__config__.show()
	unittest.main()

if __name__=="__main__":
	main()