import random
import numpy as np
from sklearn.datasets import load_digits

class Layer:
    def __init__(self, neuron_count, input_count):
        self.neuron_count = neuron_count
        scale = np.sqrt(2 / (neuron_count + input_count))
        self.weights = np.random.randn(neuron_count, input_count) * scale
        self.bias = np.zeros(neuron_count)

    # input_batch is an array of inputs of each sample data iteration
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
    
    def softmax_activation(self, input_batch):
        """Activation function for output layer using softmax"""
        self.input = input_batch
        raw_input = np.dot(input_batch, self.weights.T) + self.bias
        logit_numeric_stability = raw_input - np.max(raw_input, axis=1, keepdims=True)
        self.activation_function = np.exp(logit_numeric_stability) / np.sum(np.exp(logit_numeric_stability), axis=1, keepdims=True)

        return self.activation_function
    
    def softmax_CE_gradient(self, true_class_array):
        """Gradient calculation for cross entropy softmax combo, dL/dz"""
        batch_size = len(true_class_array)
        num_classes = np.max(true_class_array) + 1

        Y = one_hot_encode(batch_size, num_classes, true_class_array)
        gradient = self.activation_function - Y # dZ = prediction - Y

        return gradient
    
    def cross_entropy_loss(self, true_class_array):
        predictions = self.activation_function
        true_probability = one_hot_encode(len(predictions), predictions.shape[1], true_class_array) 
        average_CE = -np.sum(true_probability * np.log(predictions)) / len(predictions)
        return average_CE

def one_hot_encode(batch_size, num_classes, true_class_array):
    Y = np.zeros((batch_size, num_classes)) # initialise array of 0's 
    for sample in range(batch_size): # append 1 to one hot encoded array
        Y[sample, true_class_array[sample]] = 1

    return Y

def forward_pass(neural_layers, first_input):
    input = first_input
    for i in range(len(neural_layers)):
        if i == (len(neural_layers) - 1):
            neural_layers[i].softmax_activation(input)
            break
        next_input = neural_layers[i].prediction(input)
        input = next_input
        

def _backpropagate(gradient, layer, upstream_list, weight_dict, bias_dict, batch_size):
    next_upstream = np.dot(gradient, layer.weights)
    upstream_list.append(next_upstream)
    # calculate weight gradient with reshaped gradient and input matrices
    weight_gradient = np.dot(gradient.T, np.array(layer.input)) / batch_size
    average_bias_gradient = np.mean(gradient, axis = 0)
    # store weight and biases for update
    weight_dict[layer] = weight_gradient
    bias_dict[layer] = average_bias_gradient

def _adjust_W_and_B(neural_layers, input_data_length, learning_rate, weight_dict, bias_dict):
    for layer in neural_layers:
        layer.weights -= weight_dict[layer] * learning_rate
        layer.bias -= bias_dict[layer] * learning_rate

def training_loop_2(neural_layers, input_data, epoch_count, learning_rate=2.0):
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
            # specifically backpropagate for output layer using softmax activation and cross entropy loss
            if layer == neural_layers[-1]:
                output_activations = neural_layers[-1].activation_function
                targets = np.array([y for x, y in input_data])
                gradient = layer.softmax_CE_gradient(targets) # call gradient explicitly
                CE_error = layer.cross_entropy_loss(targets)
                _backpropagate(gradient, layer, upstream_list, total_weight_gradients, total_bias_gradients, len(input_data))
                total_error += CE_error
                continue
            # get upstream gradient from previous layer then back propagate
            upstream_gradient = upstream_list.pop()
            gradient = layer.gradient(upstream_gradient) # call gradient explicitly
            _backpropagate(gradient, layer, upstream_list, total_weight_gradients, total_bias_gradients, len(input_data))
        
        # find average and adjust weight and bias accordingly
        _adjust_W_and_B(neural_layers, len(input_data), learning_rate, total_weight_gradients, total_bias_gradients)

        # print total error
        if (epoch != 0 and epoch % 1000 == 0) or epoch == (epoch_count - 1):
            x = epoch
            print(f'Total error after {x} epoch: {total_error}')
            # confidenceAnalytics(neural_layers, input_data)
        # if total_error < 0.01:
        #     print(f'Early stoppage at epoch: {epoch}')
        #     break
        
def load_binary_digits():
    # load in training data of 1 or 0
    data = load_digits()
    # mask = (data.target == 0) | (data.target == 1) | (data.target == 2) | (data.target == 3)
    X = data.images 
    y = data.target
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
    """Test validation samples for monitoring purposes, multiclass classification"""
    confidence_correct = 0
    weak_prediction = 0
    wrong_prediction = 0
    total_samples = len(validation_data)

    inputs = [x for x, y in validation_data]
    true_class = [y for x, y in validation_data]

    forward_pass(neural_layer, inputs)

    final_prediction = neural_layer[-1].activation_function.tolist()

    for row_index in range(len(final_prediction)):
        probability = max(final_prediction[row_index])
        predicted_neuron = final_prediction[row_index].index(probability)

        if probability < 0.9 and true_class[row_index] == predicted_neuron:
            weak_prediction += 1
            print(f'weak prediction of {probability * 100}% at index: {row_index}, predicted class: {predicted_neuron}, true class: {true_class[row_index]}')
        elif true_class[row_index] != predicted_neuron:
            wrong_prediction += 1
            print(f'predicted class: {predicted_neuron}, true class: {true_class[row_index]}, index: {row_index}')
        else:
            confidence_correct += 1
        
    print(f'good predictions: {confidence_correct}, weak predictions: {weak_prediction}, wrong predictions: {wrong_prediction}')
  
if __name__ == '__main__':
    sample_data = load_binary_digits()
    training_data = sample_data[0]
    validation_data = sample_data[1]

    hiddenLayer1 = Layer(128, 64)
    hiddenLayer2 = Layer(64, 128) 
    outputLayer = Layer(10, 64)
    neural_layers = [hiddenLayer1, hiddenLayer2, outputLayer] 

    training_loop_2(neural_layers, training_data, 2000)

    print('After learning')
    # validation_test(neural_layers, validation_data)
    confidenceAnalytics(neural_layers, validation_data)



