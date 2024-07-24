'''
Write a program that accepts two matrices A and B as
 input and returns their product AB. 
Check if A and B are multipliable; if not, return error message.

'''
def matrix_multiplication(A, B):
    if len(A[0]) != len(B):
        return "Error: Matrices are not multipliable"
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result

def main():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    result = matrix_multiplication(A, B)
    print(result)

if __name__ == "__main__":
    main()