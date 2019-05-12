#!/usr/bin/python

import numpy as np
import unittest
import ANN

class Test_basic_ANN(unittest.TestCase):
	def test_init(self):
		self.network = ANN.basic_ANN(2, 2, 2)


def main():
	print("Testing")
	np.__config__.show()
	unittest.main()

if __name__=="__main__":
	main()