from Neuro.NeuronNetwork import NeuralNetwork


def teaching(exmple: int, result: float, object: NeuralNetwork) -> bool:
	"""
	y_model - значение что выдала нейросеть
	y_real - значение что мы хотим получить
	"""
	for i in range(0, 10):
		y = object.start_session(INPUT)
		print('Delta: {}'.format(object.teach(y_model=y, y_real=result)))
		print('Result: X = {}; Y = {}\n'.format(INPUT, y))


if __name__ == '__main__':

	INPUT = 10
	
	network = NeuralNetwork()
	print(str(network) + '\n')
	
	y = network.start_session(INPUT)
	print('Result: X = {}; Y = {}'.format(INPUT, y))
	
	print('Delta: {}\n'.format(network.teach(y_model=y, y_real=1.0)))

	print('After: {}'.format(network))

	# teaching(INPUT, 0, network)
	
	
	
