'''
Write a program that accepts a matrix as input
and returns the transpose of the matrix.
'''

# Accept the matrix as input
matrix = []
rows = int(input("Enter the number of rows: "))
columns = int(input("Enter the number of columns: "))

for i in range(rows):
    row = []
    for j in range(columns):
        element = int(input(f"Enter element at position ({i+1}, {j+1}): "))
        row.append(element)
    matrix.append(row)

# Calculate the transpose of the matrix
transpose = []
for j in range(columns):
    column = []
    for i in range(rows):
        column.append(matrix[i][j])
    transpose.append(column)

# Print the transpose of the matrix
print("Transpose of the matrix:")
for row in transpose:
    print(row)