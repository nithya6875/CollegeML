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

# Adding a column of ones to include bias in the matrix operation
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculate pseudoinverse of X_b
X_b_pinv = np.linalg.pinv(X_b)

# Calculate weights using pseudoinverse
weights_pinv = np.dot(X_b_pinv, y)

# Extract bias and weights separately
bias_pinv = weights_pinv[0]
weights_pinv = weights_pinv[1:]

print("Pseudoinverse Weights:", weights_pinv)
print("Pseudoinverse Bias:", bias_pinv)

# Prediction function using pseudoinverse weights
def predict_pinv(features):
    z = np.dot(features, weights_pinv) + bias_pinv
    return sigmoid(z)

# Test predictions using pseudoinverse
for i in range(len(X)):
    prediction = predict_pinv(X[i])
    print(f"Transaction {i+1}: Predicted (Pseudoinverse) = {prediction:.2f}, Actual = {y[i]}")
