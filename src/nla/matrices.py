import numpy as np
from typing import Literal

def broadcast_compatibility(shape_matrixA: tuple, shape_matrixB: tuple) -> bool: 
    Dimension_compatible = False
    
    reversed_shape_matrix_A = reversed(shape_matrixA)
    reversed_shape_matrix_B = reversed(shape_matrixB)
    
    for shape1, shape2 in zip(reversed_shape_matrix_A, reversed_shape_matrix_B):
        if shape1 != shape2 and shape1 != 1 and shape2 != 1:
            Dimension_compatible = False
        else: 
            Dimension_compatible = True
    
    return Dimension_compatible
        
def matmul_compatibility(shape_matrixA: tuple, shape_matrixB: tuple) -> bool:
    Compatible = True
    
    if len(shape_matrixB) == 1 or len(shape_matrixA) == 1:
        Compatible = broadcast_compatibility(shape_matrixA, shape_matrixB)
        return Compatible
    
    if shape_matrixA[-1] != shape_matrixB[-2]:
        Compatible = False
        
    if len(shape_matrixA) > 2 and len(shape_matrixB) > 2:
        batch_shape_A = shape_matrixA[:-2]
        batch_shape_B = shape_matrixB[:-2]

        Compatible = broadcast_compatibility(batch_shape_A, batch_shape_B)
        
    return Compatible
        
def matrix_operation(matrixA: np.ndarray, matrixB: np.ndarray, operation: Literal['add', 'sub']) -> np.ndarray:
    
    if len(matrixA) == 0 or len(matrixB) == 0:
        raise ValueError("Can not perform operations over empty matrix/matrices") 

    if operation == 'add': 
        operation = lambda x,y: x+y
    elif operation == 'sub':
        operation = lambda x,y: x-y
    else:
        raise ValueError("Operation is invalid (only addition and subtraction are accepted)")
    
    compatible = broadcast_compatibility(matrixA.shape, matrixB.shape)
    
    if not compatible: 
        raise ValueError("Dimensions are not compatible for given matrices")
    
    if len(matrixA.shape) == len(matrixB.shape):
        result = np.empty_like(matrixA)
        for idx, element in enumerate(zip(matrixA, matrixB)):
            result[idx] = operation(element[0], element[1])
            
        return result
    
    result = np.empty(shape=(matrixB.shape[0], matrixA.shape[0]))
    
    for i in range(matrixB.shape[0]):
        row = matrixB[i, :]
        for idx, element in enumerate(zip(row, matrixA)):
                result[i, idx] = operation(element[0], element[1])
                
    return result

def scalar_multiplication(matrix: np.ndarray, scalar_quantity: int | float) -> np.ndarray:
    
    if not isinstance(scalar_quantity, (int, float)):
        raise ValueError("scalar quantity is invalid")
    
    result = np.empty_like(matrix, dtype=np.float32)
    
    if len(matrix.shape) == 1:
        for idx, element in enumerate(matrix):
            print(element*scalar_quantity)
            result[idx] = element * scalar_quantity
        return result

    for row_number in range(len(matrix)):
        row = matrix[row_number, :]
        
        for idx, element in enumerate(row):
            result[row_number, idx] = element * scalar_quantity
            
    return result

def matrix_multiplication(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray:
    
    if not isinstance(matrixA, np.ndarray) or not isinstance(matrixB, np.ndarray):
        raise TypeError("Incorrect matrices provided, required np.ndarray")
    
    if len(matrixA) == 0 or len(matrixB) == 0:
        raise ValueError("Matrix can not be an empty")
    
    compatible = matmul_compatibility(matrixA.shape, matrixB.shape)
    
    if compatible: 
    
        result = np.empty(shape=(matrixA.shape[0], matrixB.shape[1]))
    
        for i in range(matrixA.shape[0]):
            for j in range(matrixB.shape[1]):
                total = 0
            
                for k in range(matrixA.shape[1]):
                    total += matrixA[i, k] * matrixB[k, j]
            
                result[i, j] = total
                
        return result
    
    else : raise ValueError("Dimensions are not compatible")

def transpose(matrix: np.ndarray) -> np.ndarray:
    
    if len(matrix) == 0:
        raise ValueError("Empty matrix can not be transposed")
    
    if len(matrix.shape) == 1:
        raise ValueError("(n, ) matrix can not be transposed")
    
    if matrix.shape[0] == matrix.shape[1]:
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j >= i:
                    temp = matrix[i][j]
                    matrix[i][j] = matrix[j][i]
                    matrix[j][i] = temp
                    
        return matrix
    
    if matrix.shape[0] != matrix.shape[1]:
        
        transposed_matrix = np.empty(shape=(matrix.shape[1], matrix.shape[0]))
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                transposed_matrix[j][i] = matrix[i][j]
                
        return transposed_matrix
    
def identity_matrix(shape: int) -> np.ndarray:
    
    if shape <= 0:
        raise ValueError("Identiy matrix can not be created for negative shapes")
    
    identity = np.empty(shape=(shape, shape), dtype=np.int32)
    
    for i in range(shape):
        for j in range(shape):
            if i == j: 
                identity[i,j] = 1
            else: identity[i,j] = 0
            
    return identity


def rank(matrix: np.ndarray) -> int:
    
    if len(matrix) == 0:
        raise ValueError("Matrix can not be empty")

    matrix = matrix.astype(np.float32).copy()

    rows, cols = matrix.shape

    pivot_row = 0
    pivot_col = 0
    rank = 0
    det_sign = 1

    while pivot_row < rows and pivot_col < cols:

        # 1. Find a non-zero pivot in the current column
        pivot_index = None

        for row in range(pivot_row, rows):
            if matrix[row, pivot_col] != 0:
                pivot_index = row
                break

        # 2. No pivot in this column → move to next column
        if pivot_index is None:
            pivot_col += 1
            continue

        # 3. Move the pivot row into position
        if pivot_index != pivot_row:
            matrix[[pivot_row, pivot_index]] = matrix[[pivot_index, pivot_row]]
            det_sign = det_sign * -1

        # 4. Eliminate everything below the pivot
        for row in range(pivot_row + 1, rows):

            multiplier = (
                matrix[row, pivot_col]
                / matrix[pivot_row, pivot_col]
            )

            matrix[row, :] = (
                matrix[row, :]
                - multiplier * matrix[pivot_row, :]
            )

        # 5. We found one pivot
        rank += 1

        # 6. Move to the next pivot position
        pivot_row += 1
        pivot_col += 1

    return rank, matrix, det_sign

def determinant(matrix: np.ndarray) -> float:
    
    if len(matrix.shape) == 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Rank can not be found for non square matrix")
    
    matrix, det_sign = rank(matrix)[-2:]
    rows = matrix.shape[0]
    
    determinant = 1
    for i in range(rows):
        determinant *= matrix[i, i]
        
    return determinant * det_sign