import random
import numpy as np
from sklearn.datasets import load_digits

class Layer:
    def __init__(self, neuron_count, input_count):
        self.neuron_count = neuron_count
        scale = np.sqrt(2 / (neuron_count + input_count))
        self.weights = np.random.randn(neuron_count, input_count) * scale
        self.bias = np.zeros(neuron_count)

    def prediction(self, input_batch):
        self.input = np.array(input_batch)  
        sigmoid_input = np.dot(input_batch, self.weights.T) + self.bias
        self.activation_function = 1 / (1 + np.exp(-sigmoid_input))
        return self.activation_function
    
    def _sigmoid_deriv(self, x): 
        return x * (1 - x)
    
    def gradient(self, upstream_gradient):
        error = np.array(upstream_gradient).reshape(-1, self.neuron_count)  
        gradient = error * self._sigmoid_deriv(self.activation_function)

        return gradient
    
def forward_pass(neural_layers, first_input):
    input = first_input
    for i in range(len(neural_layers)):
        next_input = neural_layers[i].prediction(input)
        input = next_input
        if i == (len(neural_layers) - 1):
            break

def _backpropagate(upstream, layer, upstream_list, weight_dict, bias_dict, batch_size):
    gradient = layer.gradient(upstream)
    next_upstream = np.dot(gradient, layer.weights)
    # calculate weight gradient with reshaped gradient and input matrices
    weight_gradient = np.dot(gradient.T, np.array(layer.input)) / batch_size
    average_bias_gradient = np.mean(gradient, axis = 0)
    upstream_list.append(next_upstream)
    # store weight and biases for update
    weight_dict[layer] = weight_gradient
    bias_dict[layer] = average_bias_gradient
    
def _adjust_W_and_B(neural_layers, input_data_length, learning_rate, weight_dict, bias_dict):
    for layer in neural_layers:
        layer.weights += weight_dict[layer] * learning_rate
        layer.bias += bias_dict[layer] * learning_rate

def training_loop_2(neural_layers, input_data, epoch_count, learning_rate=5.0):
    for epoch in range(epoch_count):
        total_error = 0
        total_weight_gradients = {layer: None for layer in neural_layers}
        total_bias_gradients = {layer: None for layer in neural_layers}

        # perform a single forward and backwards propagation once through each instance in the sample data
        input = np.array([i[0] for i in input_data])
        upstream_list = []

        # forward pass to output initializing every layers activation functions
        forward_pass(neural_layers, input)

        # back propagation through each layer, combining weight and bias changes from each sample data instance
        for layer in neural_layers[::-1]:
            # specifically backpropagate for output layer
            if layer == neural_layers[-1]:
                output_activations = neural_layers[-1].activation_function
                targets = np.array([y for x, y in input_data]).reshape(-1, 1)
                output_error = targets - output_activations
                _backpropagate(output_error, layer, upstream_list, total_weight_gradients, total_bias_gradients, len(input_data))

                total_error += np.mean(output_error) ** 2
                continue
            # get upstream gradient from previous layer then back propagate
            upstream_gradient = upstream_list.pop()
            _backpropagate(upstream_gradient, layer, upstream_list, total_weight_gradients, total_bias_gradients, len(input_data))
        
        # find average and adjust weight and bias accordingly
        _adjust_W_and_B(neural_layers, len(input_data), learning_rate, total_weight_gradients, total_bias_gradients)

        # print total error
        if (epoch != 0 and epoch % 1000 == 0) or epoch == (epoch_count - 1):
            x = epoch
            print(f'Total error after {x} epoch: {total_error/len(input_data)}')
            confidenceAnalytics(neural_layers, input_data)
        # if total_error < 0.01:
        #     print(f'Early stoppage at epoch: {epoch}')
        #     break
        
def load_binary_digits():
    # load in training data of 1 or 0
    data = load_digits()
    mask = (data.target == 0) | (data.target == 1)
    X = data.images[mask] / 16
    y = data.target[mask]
    X_flatten = X.reshape(X.shape[0], -1)

    training_data = []
    validation_data = []
    for i in range(len(X_flatten)):
        if i / len(X_flatten) >= 0.75:
            validation_data.append((X_flatten[i], y[i]))
        else:
            training_data.append((X_flatten[i], y[i]))
    return (training_data, validation_data)
                
def confidenceAnalytics(neural_layer, validation_data):
    confidence_correct = 0
    weak_prediction = 0
    wrong_prediction = 0
    total_samples = len(validation_data)

    validation_samples = np.array([x[0] for x in validation_data])
    targets = np.array([x[1] for x in validation_data])

    forward_pass(neural_layer, validation_samples)
    prediction = neural_layer[-1].activation_function

    for i in range(len(targets)):
        if targets[i] == 0 and prediction[i] > 0.1:
            if prediction[i] > 0.3:
                wrong_prediction += 1
                print(f'Wrong prediction: {prediction[i]} target: {targets[i]}')
                continue 
            weak_prediction += 1
            print(f'Weak prediction: {prediction[i]} target: {targets[i]}')
        elif targets[i] == 1 and prediction[i] < 0.9:
            if prediction[i] < 0.7:
                wrong_prediction += 1
                print(f'Wrong prediction: {prediction[i]} target: {targets[i]}')
                continue
            weak_prediction += 1
            print(f'Weak prediction: {prediction[i]} target: {targets[i]}')
        else:
            confidence_correct += 1

    print(f'Good predictions: {confidence_correct}, weak predictions: {weak_prediction}, confidence percentage: {confidence_correct/total_samples}')
    print(f'wrong predictions: {wrong_prediction}')

    
if __name__ == '__main__':
    sample_data = load_binary_digits()
    training_data = sample_data[0]
    validation_data = sample_data[1]

    hiddenLayer1 = Layer(32, 64)
    hiddenLayer2 = Layer(8, 32) 
    outputLayer = Layer(1, 8)
    neural_layers = [hiddenLayer1, hiddenLayer2, outputLayer] 

    training_loop_2(neural_layers, training_data, 3000)

    print('After learning')
    # validation_test(neural_layers, validation_data)
    confidenceAnalytics(neural_layers, validation_data)



