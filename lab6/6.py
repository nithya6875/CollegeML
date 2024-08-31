'''
Use customer data provided below. Build a perceptron and learn to
classify the transactions as high or low value as provided in the
below table. Use sigmoid as the activation function. Initialise the
weights and learning rate with your choice.
'''

import numpy as np

# Sample Data
data = np.array([
    [20, 6, 2, 386, 1],
    [16, 3, 6, 289, 1],
    [27, 6, 2, 393, 1],
    [19, 1, 2, 110, 0],
    [24, 4, 2, 280, 1],
    [22, 1, 5, 167, 0],
    [15, 4, 2, 271, 1],
    [18, 4, 2, 274, 1],
    [21, 1, 4, 148, 0],
    [16, 2, 4, 198, 0]
])

# Separate features and labels
X = data[:, :-1]
y = data[:, -1]

# Normalize the input features
X = X / X.max(axis=0)

# Initialize weights and bias
weights = np.random.randn(X.shape[1])
bias = np.random.randn()
learning_rate = 0.1
epochs = 10000

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        # Calculate weighted sum
        z = np.dot(X[i], weights) + bias
        # Apply sigmoid activation
        output = sigmoid(z)
        # Calculate error
        error = y[i] - output
        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Final weights and bias
print("Weights:", weights)
print("Bias:", bias)

# Prediction function
def predict(features):
    z = np.dot(features, weights) + bias
    return sigmoid(z)

# Test predictions
for i in range(len(X)):
    prediction = predict(X[i])
    print(f"Transaction {i+1}: Predicted = {prediction:.2f}, Actual = {y[i]}")
