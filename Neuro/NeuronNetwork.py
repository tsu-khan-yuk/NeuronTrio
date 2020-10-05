from .Neuron import Neuron


class NeuralNetwork:
	neurons_list = None
	inputs = None
	
	def __init__(self):
		self.inputs = list()
		self.neurons_list = list()
		for i in range(3):
			self.neurons_list.append(Neuron())
	
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
	
	def teach(self, y_model: float, y_real: float) -> None:
		"""
		:param y_model:  значение что выдала нейросеть
		:param y_real: значение что мы хотим получить
		:return: None
		"""
		# third neuron
		delta_for_n_3 = y_model * (1 - y_model) * (y_real - y_model)
		self.neurons_list[2].weights += delta_for_n_3

		# second neuron
		delta_for_n_2 = y_model * (1 - y_model) * (delta_for_n_3 * self.inputs['2 -> 3'])
		self.neurons_list[1].weights += delta_for_n_2

		# first neuron
		delta_for_n_1 = y_model * (1 - y_model) * (delta_for_n_2 * self.inputs['1 -> 2'])
		self.neurons_list[0].weights += delta_for_n_1
	
	def __str__(self):
		string = "Neuron network: ["
		for i in self.neurons_list:
			string += '\n    {}'.format(i)
		string += '\n]'
		return string
