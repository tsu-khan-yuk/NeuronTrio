from random import randint
from numpy import exp


sigmoid = lambda x: (1 / (1 + exp(-x)))


class Neuron:
	weights = None
	bias = None
	
	def __init__(self):
		self.weights = float(randint(0, 9))
		self.bias = 0.0
	
	def feedforward(self, inputs: float) -> float:
		if not isinstance(inputs, (int, float)):
			raise TypeError
		result = self.weights * inputs + self.bias
		return sigmoid(result)
	
	def __str__(self):
		return 'Neuron W:{}, BIAS:{}'.format(self.weights, self.bias)