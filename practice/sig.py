# %%
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold(x):
    return 1 if x > 0.5 else 0

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.01
epochs = 100000
error_history = []


weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_output = np.zeros((1, output_size))

for epoch in range(epochs):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    error = Y - predicted_output
    error_history.append(error)
    output_delta = error * (predicted_output * (1 - predicted_output))
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * (hidden_output * (1 - hidden_output))

    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for input_data in test_data:
    hidden_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    binary_output = threshold(predicted_output)

    print(f"Input: {input_data}, Predicted Output: {binary_output}, Binary Output: {binary_output}")

print(Y)



