import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative (x):
    return x*(1-X)

X = np.array([[0,0],
              [1,0],
              [0,1,]
              [1,1]])
y=np.array([[0],
            [1],
            [1],
            [0]])

input_neurons = 2
hidden_neuron = 2
output_neuron = 1

hidden_weights = np.randon.uniform(size=(input_neurons,hidden_neuron))
hidden_bias = np.random.uniform(size=(1,hidden_neuron))

output_weight = np.random.uniform(size=(hidden_neuron,output_neuron))
output_bias = np.random.uniform(size=(1,output_neuron))

epoches = 10
learning_rate = 0.1

for epoch in epoches:
    hidden_layer_input = np.dot(X,hidden_weights)+hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output,output_weight) + output_bias
    output = sigmoid(output_layer_input)

    error = y-output
    output_delta = error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
