import numpy as np
import pytest

from nla import matrices

# ============================================================
# Matrix operation
# ============================================================

def test_addition_same_shape():
    a = np.array([[1,2],
                  [3,4]])
    b = np.array([[4,5],
                  [6,7]])
    expected = np.array([[5, 7],
                         [9, 11]])
    addition = matrices.matrix_operation(a, b, 'add')
    np.testing.assert_equal(addition, expected)
    
def test_addition_diff_shape():
    a = np.array([[1,2],
                  [3,4]])
    b = np.array([[4,5],
                  [6,7],
                  [7,8]])
    
    with pytest.raises(ValueError):
        matrices.matrix_operation(a, b, 'add')
        
def test_addition_diff_ndim():
    a = np.array([[[1, 2],
                   [3, 4]]])
    b = np.array([[4, 5],
                  [6, 7]])
    
    with pytest.raises(ValueError):
        matrices.matrix_operation(a, b, 'add')
        
def test_addition_empty():
    a = np.array([])
    b = np.array([])

    with pytest.raises(ValueError):
        matrices.matrix_operation(a, b, 'add')
        
def test_addition_zeros():
    a = np.array([[0, 0],
                  [0, 0]])
    b = np.array([[1, 2],
                  [3, 4]])
    expected = np.array([[1, 2],
                         [3, 4]])

    result = matrices.matrix_operation(a, b, 'add')

    np.testing.assert_array_equal(result, expected)
    

def test_addition_floats():
    a = np.array([[1.5, 2.2]])
    b = np.array([[3.5, 4.8]])
    expected = np.array([[5.0, 7.0]])

    result = matrices.matrix_operation(a, b, 'add')

    np.testing.assert_allclose(result, expected)
    

def test_addition_diff_dtype():
    a = np.array([[1, 2]], dtype=np.int32)
    b = np.array([[3, 4]], dtype=np.float64)
    expected = np.array([[4, 6]])

    result = matrices.matrix_operation(a, b, 'add')

    np.testing.assert_allclose(result, expected)
    

# ============================================================
# Scalar multiplication
# ============================================================

def test_scalar_multiplication():
    a = np.array([[1, 2, 3],
                 [5, 6, 7]])
    scalar = 2
    expected = np.array([[2, 4, 6],
                        [10, 12, 14]])

    result = matrices.scalar_multiplication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_neg_scalar_multiplication():
    a = np.array([[1, 2, 3],
                 [5, 6, 7]])
    scalar = -2
    expected = np.array([[-2, -4, -6],
                        [-10, -12, -14]])

    result = matrices.scalar_multiplication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_zero_scalar():
    a = np.array([1, 2, 3])
    scalar = 0
    expected = np.array([0, 0, 0])

    result = matrices.scalar_multiplication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_float_scalar_multiplication():
    a = np.array([[1, 2, 3],
                 [5, 6, 7]])
    scalar = 2.1
    expected = np.array([[2.1, 4.2, 6.3],
                         [10.5, 12.6, 14.7]])

    result = matrices.scalar_multiplication(a, scalar)

    np.testing.assert_allclose(result, expected)


def test_vector_passed_as_scalar():
    a = np.array([[1, 2, 3],
                 [6, 7, 8]])
    scalar = np.array([[1, 2],
                       [3, 4]])

    with pytest.raises(ValueError):
        matrices.scalar_multiplication(a, scalar)


def test_string_passed_as_scalar():
    a = np.array([[1, 2, 3],
                  [5, 6, 7]])
    scalar = "1.1"

    with pytest.raises(ValueError):
        matrices.scalar_multiplication(a, scalar)


def test_boolean_scalar():
    a = np.array([[1, 2, 3],
                 [5, 6, 7]])
    scalar = True
    expected = np.array([[1, 2, 3],
                         [5, 6, 7]])

    result = matrices.scalar_multiplication(a, scalar)

    np.testing.assert_allclose(result, expected)
    
# ============================================================
# Matrix Multiplication
# ============================================================

import numpy as np
import pytest
from nla import matrices


def test_matrix_multiplication_basic():
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    expected = np.array([[19, 22],
                         [43, 50]])

    np.testing.assert_array_equal(
        matrices.matrix_multiplication(A, B),
        expected
    )


def test_matrix_multiplication_non_square():
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    expected = np.array([[58, 64],
                         [139, 154]])

    np.testing.assert_array_equal(
        matrices.matrix_multiplication(A, B),
        expected
    )


def test_matrix_multiplication_identity():
    A = np.array([[1, 2],
                  [3, 4]])

    I = np.eye(2)

    np.testing.assert_array_equal(
        matrices.matrix_multiplication(A, I),
        A
    )


def test_matrix_multiplication_zero():
    A = np.array([[1, 2],
                  [3, 4]])

    Z = np.zeros((2, 2))

    expected = np.zeros((2, 2))

    np.testing.assert_array_equal(
        matrices.matrix_multiplication(A, Z),
        expected
    )


def test_matrix_multiplication_shape_mismatch():
    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    with pytest.raises(ValueError):
        matrices.matrix_multiplication(A, B)


def test_matrix_multiplication_invalid_dimension():
    A = np.array([1, 2, 3])

    B = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    with pytest.raises(ValueError):
        matrices.matrix_multiplication(A, B)


def test_matrix_multiplication_row_column():
    A = np.array([[1, 2, 3]])

    B = np.array([[4],
                  [5],
                  [6]])

    expected = np.array([[32]])

    np.testing.assert_array_equal(
        matrices.matrix_multiplication(A, B),
        expected
    )


def test_matrix_multiplication_does_not_modify_inputs():
    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[5, 6],
                  [7, 8]])

    A_original = A.copy()
    B_original = B.copy()

    matrices.matrix_multiplication(A, B)

    np.testing.assert_array_equal(A, A_original)
    np.testing.assert_array_equal(B, B_original)
    
    
# ============================================================
# Matrix Transpose
# ============================================================
    
import numpy as np
import pytest
from nla import matrices


def test_transpose_square_matrix():
    matrix = np.array([
        [1, 2],
        [3, 4]
    ])

    expected = np.array([
        [1, 3],
        [2, 4]
    ])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_rectangular_matrix():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    expected = np.array([
        [1, 4],
        [2, 5],
        [3, 6]
    ])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_single_row_matrix():
    matrix = np.array([[1, 2, 3]])

    expected = np.array([
        [1],
        [2],
        [3]
    ])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_single_column_matrix():
    matrix = np.array([
        [1],
        [2],
        [3]
    ])

    expected = np.array([[1, 2, 3]])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_1x1_matrix():
    matrix = np.array([[5]])

    expected = np.array([[5]])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_negative_and_float_values():
    matrix = np.array([
        [1.5, -2.5],
        [-3.5, 4.5]
    ])

    expected = np.array([
        [1.5, -3.5],
        [-2.5, 4.5]
    ])

    np.testing.assert_array_equal(
        matrices.transpose(matrix),
        expected
    )


def test_transpose_invalid_1d_input():
    matrix = np.array([1, 2, 3])
    
    print(matrix.T.shape)

    with pytest.raises(ValueError):
        matrices.transpose(matrix)


def test_transpose_empty_matrix():
    matrix = np.empty((0, 3))

    with pytest.raises(ValueError):
        matrices.transpose(matrix)
        

# ============================================================
# Identity Matrix
# ============================================================


import numpy as np
import pytest
from nla import matrices


def test_identity_matrix_1x1():
    result = matrices.identity_matrix(1)

    expected = np.array([[1]])

    np.testing.assert_array_equal(result, expected)


def test_identity_matrix_2x2():
    result = matrices.identity_matrix(2)

    expected = np.array([
        [1, 0],
        [0, 1]
    ])

    np.testing.assert_array_equal(result, expected)


def test_identity_matrix_3x3():
    result = matrices.identity_matrix(3)

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    np.testing.assert_array_equal(result, expected)


def test_identity_matrix_larger():
    result = matrices.identity_matrix(5)

    expected = np.eye(5)

    np.testing.assert_array_equal(result, expected)


def test_identity_matrix_is_square():
    result = matrices.identity_matrix(4)

    assert result.shape == (4, 4)


def test_identity_matrix_diagonal():
    result = matrices.identity_matrix(4)

    np.testing.assert_array_equal(
        np.diag(result),
        np.ones(4)
    )


def test_identity_matrix_off_diagonal():
    result = matrices.identity_matrix(4)

    off_diagonal = result.copy()
    np.fill_diagonal(off_diagonal, 0)

    np.testing.assert_array_equal(
        off_diagonal,
        np.zeros((4, 4))
    )


def test_identity_matrix_zero_dimension():
    with pytest.raises(ValueError):
        matrices.identity_matrix(0)


def test_identity_matrix_negative_dimension():
    with pytest.raises(ValueError):
        matrices.identity_matrix(-3)


def test_identity_matrix_non_integer():
    with pytest.raises((TypeError, ValueError)):
        matrices.identity_matrix(3.5)


# ============================================================
# Matrix Rank
# ============================================================

import numpy as np
import pytest
from nla import matrices


def test_rank_zero_matrix():
    matrix = np.zeros((3, 3))

    assert matrices.rank(matrix)[0] == 0


def test_rank_identity_matrix():
    matrix = np.eye(3)

    assert matrices.rank(matrix)[0] == 3


def test_rank_full_rank_square_matrix():
    matrix = np.array([
        [1, 2],
        [3, 4]
    ])

    assert matrices.rank(matrix)[0] == 2


def test_rank_singular_matrix():
    matrix = np.array([
        [1, 2],
        [2, 4]
    ])

    assert matrices.rank(matrix)[0] == 1


def test_rank_dependent_rows():
    matrix = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ])

    assert matrices.rank(matrix)[0] == 1


def test_rank_independent_rows():
    matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    assert matrices.rank(matrix)[0] == 3


def test_rank_rectangular_tall_matrix():
    matrix = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    assert matrices.rank(matrix)[0] == 2


def test_rank_rectangular_wide_matrix():
    matrix = np.array([
        [1, 0, 1],
        [0, 1, 1]
    ])

    assert matrices.rank(matrix)[0] == 2


def test_rank_requires_row_swap():
    matrix = np.array([
        [0, 1],
        [1, 0]
    ])

    assert matrices.rank(matrix)[0] == 2


def test_rank_requires_skipping_zero_column():
    matrix = np.array([
        [0, 1, 2],
        [0, 2, 4],
        [0, 3, 6]
    ])

    assert matrices.rank(matrix)[0] == 1


def test_rank_does_not_modify_input():
    matrix = np.array([
        [1, 2],
        [2, 4]
    ])

    original = matrix.copy()

    result = matrices.rank(matrix)[0]

    assert result == 1
    np.testing.assert_array_equal(matrix, original)


def test_rank_empty_matrix():
    matrix = np.empty((0, 3))

    with pytest.raises(ValueError):
        matrices.rank(matrix)


def test_rank_invalid_1d_input():
    matrix = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        matrices.rank(matrix)
        
# ============================================================
# Matrix Determinant
# ============================================================

import numpy as np
import pytest
from nla import matrices


def test_determinant_1x1():
    matrix = np.array([[5]])

    assert matrices.determinant(matrix) == 5


def test_determinant_2x2():
    matrix = np.array([
        [1, 2],
        [3, 4]
    ])

    assert matrices.determinant(matrix) == -2


def test_determinant_3x3():
    matrix = np.array([
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ])

    assert matrices.determinant(matrix) == 1


def test_determinant_identity():
    matrix = np.eye(4)

    assert matrices.determinant(matrix) == 1


def test_determinant_zero_matrix():
    matrix = np.zeros((3, 3))

    assert matrices.determinant(matrix) == 0


def test_determinant_singular_matrix():
    matrix = np.array([
        [1, 2],
        [2, 4]
    ])

    assert matrices.determinant(matrix) == 0


def test_determinant_negative():
    matrix = np.array([
        [0, 1],
        [1, 0]
    ])

    assert matrices.determinant(matrix) == -1


def test_determinant_requires_row_swap():
    matrix = np.array([
        [0, 2],
        [3, 4]
    ])

    assert matrices.determinant(matrix) == -6


def test_determinant_4x4():
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 6, 4, 8],
        [3, 1, 1, 2]
    ])

    assert matrices.determinant(matrix) == 72


def test_determinant_does_not_modify_input():
    matrix = np.array([
        [1, 2],
        [3, 4]
    ])

    original = matrix.copy()

    result = matrices.determinant(matrix)

    assert result == -2
    np.testing.assert_array_equal(matrix, original)


def test_determinant_non_square_matrix():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    with pytest.raises(ValueError):
        matrices.determinant(matrix)


def test_determinant_1d_input():
    matrix = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        matrices.determinant(matrix)


def test_determinant_empty_matrix():
    matrix = np.empty((0, 0))

    with pytest.raises(ValueError):
        matrices.determinant(matrix)