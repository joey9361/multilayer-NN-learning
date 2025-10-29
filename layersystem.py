import random
import numpy as np

class Layer:
    def __init__(self, neuron_count, input_count):
        self.neuron_count = neuron_count
        weights_list = [[random.uniform(-2, 2) for _ in range(input_count)] for _ in range(neuron_count)]
        self.weights = np.array(weights_list)
        # KEEP bias 1D - this was working
        bias_list = [random.uniform(-1, 1) for _ in range(neuron_count)]
        self.bias = np.array(bias_list)

    def prediction(self, input):
        self.input = np.array(input)  # NO reshape - keep 1D
        sigmoid_input = np.dot(self.weights, self.input) + self.bias
        self.activation_function = 1 / (1 + np.exp(-sigmoid_input))
        return self.activation_function
    
    def learn(self, error, input, learning_rate=0.5):
        # Ensure error is proper array
        error = np.array(error).flatten()  # Keep everything 1D
        gradient = error * self._sigmoid_deriv(self.activation_function)
        
        # Reshape temporarily for weight gradient calculation only
        gradient_reshaped = gradient.reshape(-1, 1)
        input_reshaped = np.array(input).reshape(1, -1)
        
        weight_gradient = np.dot(gradient_reshaped, input_reshaped)
        
        # Update weights and bias with consistent 1D shapes
        self.weights += weight_gradient * learning_rate
        self.bias += gradient * learning_rate  # Use original 1D gradient

        return gradient  # Return 1D for backprop

    def _sigmoid_deriv(self, x):
        return x * (1 - x)
    
def training_loop(neural_layers, input_data: list[(list, int)], epoch_count: int):
    for epoch in range(epoch_count):
        total_error = 0
        for array_input, target in input_data:
            input_list=[np.array(array_input)]
            # gradient_list = []
            error_list = []
            # neural_layer_copy = neural_layers[:-1]

            for i in range(len(neural_layers)):
                next_input = neural_layers[i].prediction(input_list[i])
                input_list.append(next_input)
                if i == (len(neural_layers) - 1):
                    # output_error = target - next_input
                    # next_gradient = neural_layers[-1].learn(output_error, neural_layers[-1].input)
                    # gradient_list.append(next_gradient)
                    break
                

            for layer in neural_layers[::-1]:
                if layer == outputLayer:
                    output_error = target - input_list.pop()
                    next_gradient = layer.learn(output_error, layer.input)
                    next_error = np.dot(layer.weights.T, next_gradient)
                    error_list.append(next_error)
                    total_error += output_error ** 2
                    continue
                error = error_list.pop() * layer._sigmoid_deriv(input_list.pop())
                next_gradient = layer.learn(error, layer.input)
                next_error = np.dot(layer.weights.T, next_gradient)
                error_list.append(next_error)

                # gradient = gradient_list.pop()
                # next_layer = neural_layer_copy.pop()
                # next_error = np.dot(layer.weights.T, gradient) * next_layer._sigmoid_deriv(input_list.pop())
                # next_gradient = next_layer.learn(next_error, next_layer.input)
                # gradient_list.append(next_gradient)
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Error: {total_error}')
    

hiddenLayer1 = Layer(4, 2)
outputLayer = Layer(1, 4)
neural_layers = [hiddenLayer1, outputLayer]

training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

training_loop(neural_layers, training_data, 5000)


print('After learning')
for array, target in training_data:
        hidden_pred1 = hiddenLayer1.prediction(array)

        answer_input = hidden_pred1
        output_prediction = outputLayer.prediction(answer_input)

        print(f'Input: {array}, output: {output_prediction}, target: {target}')
