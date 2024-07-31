import numpy as np
import pandas as pd

# Load data from csv into pandas dataframe

df = pd.read_csv('Lab Session Data(Purchase data).csv')

# display the DataFrame
print(df.head())

# Segregate the columns into matrices A and C
A_columns = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
C_column = 'Payment (Rs)'

# Create matrix A and C
A = df[A_columns].values
C = df[C_column].values

# Display the matrices to verify
print("Matrix A:")
print(A)
print("Matrix C:")
print(C)

# Compute the pseudo-inverse of matrix A
A_pseudo_inv = np.linalg.pinv(A)

# Solve for X using the pseudo-inverse
X = np.dot(A_pseudo_inv, C)

# Display the result
print("Cost of each product (Candies, Mangoes, Milk Packets):")
print(X)
