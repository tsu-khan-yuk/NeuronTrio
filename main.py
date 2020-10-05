from numpy import exp
INPUT = 10

sigmoid = lambda x: (1 / (1 + exp(-x)))


class Neuron:
	weights = None
	bias = None
	
	def __init__(self, weights: float, bias=0):
		if not isinstance(weights, (int, float)) or not isinstance(bias, (int, float)):
			raise TypeError
		self.weights = float(weights)
		self.bias = float(bias)
	
	def feedforward(self, inputs: float) -> float:
		if not isinstance(inputs, (int, float)):
			raise TypeError
		result = self.weights * inputs + self.bias
		return sigmoid(result)
	
	def __str__(self):
		return 'Neuron W:{}, BIAS:{}'.format(self.weights, self.bias)


class NeuralNetwork:
	neurons_list = None
	inputs = None
	
	def __init__(self, *args):
		if len(args) == 3:
			self.neurons_list = list()
			self.inputs = dict()
			for i in args:
				if isinstance(i, (int, float)):
					self.neurons_list.append(Neuron(i))
		else:
			raise Exception('You can use three neurons')
	
	def start_session(self, input_value: int) -> float:
		if not isinstance(input_value, (int, float)):
			raise TypeError
		first_n_output = self.neurons_list[0].feedforward(input_value)
		second_n_output = self.neurons_list[1].feedforward(first_n_output)
		third_n_output = self.neurons_list[2].feedforward(second_n_output)
		
		self.inputs = {
			'1 -> 2': first_n_output,
			'2 -> 3': second_n_output
		}
		
		return third_n_output
	
	def teach(self, y_model: float, y_real: float) -> float:
		"""
		:param y_model:  значение что выдала нейросеть
		:param y_real: значение что мы хотим получить
		:return:
		"""
		delta_for_n_3 = y_model * (1 - y_model)*(y_real - y_model)
		self.neurons_list[2].weights += delta_for_n_3
		return delta_for_n_3
	
	def __str__(self):
		string = "Neuron network: ["
		for i in self.neurons_list:
			string += '\n    {}'.format(i)
		string += '\n]'
		return string


if __name__ == '__main__':
	
	network = NeuralNetwork(0, 3, 1)
	print(network)
	
	y = network.start_session(INPUT)
	print('Result: X = {}; Y = {}\n'.format(INPUT, y))
	
	# y_model - значение что выдала нейросеть
	# y_real - значение что мы хотим получить
	print('Delta: {}'.format(network.teach(y_model=y, y_real=0.0)))

	print('After: {}'.format(network))
	
	for i in range(0, 10):
		y = network.start_session(INPUT)
		print('Delta: {}'.format(network.teach(y_model=y, y_real=0.0)))
		# print('After: {}'.format(network))
		print('Result: X = {}; Y = {}\n'.format(INPUT, y))
	
