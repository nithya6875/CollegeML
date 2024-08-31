import numpy as np


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# AND gate input and output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [0], [0], [1]])

# Initialize weights randomly with mean 0
v = np.random.uniform(-1, 1, (2, 2))  # Weights between input and hidden layer
w = np.random.uniform(-1, 1, (2, 1))  # Weights between hidden layer and output layer

# Learning rate and number of iterations
alpha = 0.05
iterations = 1000
convergence_error = 0.002

for i in range(iterations):
    # Forward pass
    hidden_input = np.dot(inputs, v)  # Hidden layer input
    hidden_output = sigmoid(hidden_input)  # Hidden layer output

    final_input = np.dot(hidden_output, w)  # Output layer input
    predicted_output = sigmoid(final_input)  # Output layer output

    # Calculate the error
    error = outputs - predicted_output

    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged after {i + 1} iterations.")
        break

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(w.T)
    d_hidden_output = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update weights
    w += hidden_output.T.dot(d_predicted_output) * alpha
    v += inputs.T.dot(d_hidden_output) * alpha

# Final results
print("Weights between input and hidden layer:\n", v)
print("Weights between hidden and output layer:\n", w)
print("Predicted Output:\n", predicted_output)
