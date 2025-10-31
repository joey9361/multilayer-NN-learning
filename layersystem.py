import random
import numpy as np
from sklearn.datasets import load_digits

class Layer:
    def __init__(self, neuron_count, input_count):
        scale = np.sqrt(2 / (neuron_count + input_count))
        self.weights = np.random.randn(neuron_count, input_count) * scale
        self.bias = np.array([0.0 for _ in range(neuron_count)])
        # self.neuron_count = neuron_count
        # weights_list = [[random.uniform(-2, 2) for _ in range(input_count)] for _ in range(neuron_count)]
        # self.weights = np.array(weights_list)
        # # KEEP bias 1D - this was working
        # bias_list = [random.uniform(-1, 1) for _ in range(neuron_count)]
        # self.bias = np.array(bias_list)

    def prediction(self, input):
        self.input = np.array(input)  # NO reshape - keep 1D
        sigmoid_input = np.dot(self.weights, self.input) + self.bias
        self.activation_function = 1 / (1 + np.exp(-sigmoid_input))
        return self.activation_function
    
    def _sigmoid_deriv(self, x): 
        return x * (1 - x)
    
    def gradient(self, upstream_gradient):
        error = np.array(upstream_gradient).flatten()  # Keep everything 1D
        gradient = error * self._sigmoid_deriv(self.activation_function)

        return gradient
    
def forward_pass(neural_layers, first_input):
    input = first_input
    for i in range(len(neural_layers)):
        next_input = neural_layers[i].prediction(input)
        input = next_input
        if i == (len(neural_layers) - 1):
            break

def _backpropagate(upstream, layer, upstream_list, weight_dict, bias_dict):
    gradient = layer.gradient(upstream)
    next_upstream = np.dot(layer.weights.T, gradient)
    # calculate weight gradient with reshaped gradient and input matrices
    weight_gradient = np.dot(gradient.reshape(-1, 1), np.array(layer.input).reshape(1, -1))
    upstream_list.append(next_upstream)

    _update_W_and_B(weight_dict, bias_dict, layer, weight_gradient, gradient)
    
def _update_W_and_B(weight_dict, bias_dict, layer, weight_grad, bias_grad):
    if weight_dict[layer] is None:
        weight_dict[layer] = weight_grad
        bias_dict[layer] = bias_grad
    else:
        weight_dict[layer] += weight_grad
        bias_dict[layer] += bias_grad

def _adjust_W_and_B(neural_layers, input_data_length, learning_rate, weight_dict, bias_dict):
    for layer in neural_layers:
            weight_dict[layer] /= input_data_length
            bias_dict[layer] /= input_data_length
            layer.weights += weight_dict[layer] * learning_rate
            layer.bias += bias_dict[layer] * learning_rate

def training_loop_2(neural_layers, input_data, epoch_count, learning_rate=0.5):
    for epoch in range(epoch_count):
        total_error = 0
        total_weight_gradients = {layer: None for layer in neural_layers}
        total_bias_gradients = {layer: None for layer in neural_layers}
        # perform a single forward and backwards propagation once through each instance in the sample data
        for array_input, target in input_data:
            input = np.array(array_input)
            upstream_list = []

            # forward pass to output initializing every layers activation functions
            forward_pass(neural_layers, input)

            # back propagation through each layer, combining weight and bias changes from each sample data instance
            for layer in neural_layers[::-1]:
                # specifically backpropagate for output layer
                if layer == neural_layers[-1]:
                    output_error = target - layer.activation_function
                    _backpropagate(output_error, layer, upstream_list, total_weight_gradients, total_bias_gradients)

                    total_error += output_error ** 2
                    continue
                # get upstream gradient from previous layer then back propagate
                upstream_gradient = upstream_list.pop()
                _backpropagate(upstream_gradient, layer, upstream_list, total_weight_gradients, total_bias_gradients)
        
        # find average and adjust weight and bias accordingly
        _adjust_W_and_B(neural_layers, len(input_data), learning_rate, total_weight_gradients, total_bias_gradients)

        # print total error
        if epoch % 100 == 0:
            x = epoch
            print(f'Total error after {x} epoch: {total_error}')
        if total_error < 0.01:
            print(f'Early stoppage at epoch: {epoch}')
            break
        
def load_binary_digits():
    # load in training data of 1 or 0
    data = load_digits()
    mask = (data.target == 0) | (data.target == 1)
    X = data.images[mask] / 16
    y = data.target[mask]
    X_flatten = X.reshape(X.shape[0], -1)

    training_data = []
    for i in range(len(X_flatten)):
        training_data.append((X_flatten[i], y[i]))
    return training_data
                
    

    

hiddenLayer1 = Layer(16, 64)
hiddenLayer2 = Layer(8, 16)
outputLayer = Layer(1, 8)
neural_layers = [hiddenLayer1, hiddenLayer2, outputLayer]

training_data = load_binary_digits()

training_loop_2(neural_layers, training_data, 3000)


print('After learning')
for array, target in training_data:
        activations = [array]
        for layer in neural_layers:
            next_input = layer.prediction(activations.pop())
            activations.append(next_input)
        print(f'output: {activations.pop()} target: {target}')



