from keras.models import Sequential
from keras.layers import Dense
import numpy as np
model = Sequential()
model.add(Dense(2,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
input =np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([0,1,1,0])

model.fit(input,label,epochs=5000)

acc= model.evaluate(input,label)[1]
print(acc*100)

model_json = model.to_json()
with open("model.json","w") as file:
    file.write(model_json)
model.save_weights("model.h5")



'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Define network architecture
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Initialize weights and biases
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_biases = np.random.uniform(size=(1, hidden_neurons))

output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))
output_biases = np.random.uniform(size=(1, output_neurons))

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    output_delta = error * sigmoid_derivative(output)

    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    output_biases += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    hidden_weights += X.T.dot(hidden_delta) * learning_rate
    hidden_biases += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Test the trained network
hidden_layer_input = np.dot(X, hidden_weights) + hidden_biases
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
output = sigmoid(output_layer_input)

print("Predictions:")
print(output)
'''