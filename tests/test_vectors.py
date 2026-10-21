import numpy as np
import pytest

from nla import vectors


# ============================================================
# Vector operations: addition / subtraction
# ============================================================

def test_addition_same_shape():
    a = np.array([[1, 2],
                  [3, 4]])
    b = np.array([[4, 5],
                  [6, 7]])
    expected = np.array([[5, 7],
                         [9, 11]])

    result = vectors.vector_operation(a, b, lambda x, y: x + y)

    np.testing.assert_array_equal(result, expected)


def test_addition_diff_shape():
    a = np.array([[1, 2],
                  [3, 4],
                  [4, 5]])
    b = np.array([[4, 5],
                  [6, 7]])

    with pytest.raises(ValueError):
        vectors.vector_operation(a, b, lambda x, y: x + y)


def test_addition_diff_ndim():
    a = np.array([[[1, 2],
                   [3, 4]]])
    b = np.array([[4, 5],
                  [6, 7]])

    with pytest.raises(ValueError):
        vectors.vector_operation(a, b, lambda x, y: x + y)


def test_addition_empty():
    a = np.array([])
    b = np.array([])

    with pytest.raises(ValueError):
        vectors.vector_operation(a, b, lambda x, y: x + y)


def test_addition_zeros():
    a = np.array([[0, 0],
                  [0, 0]])
    b = np.array([[1, 2],
                  [3, 4]])
    expected = np.array([[1, 2],
                         [3, 4]])

    result = vectors.vector_operation(a, b, lambda x, y: x + y)

    np.testing.assert_array_equal(result, expected)


def test_addition_floats():
    a = np.array([[1.5, 2.2]])
    b = np.array([[3.5, 4.8]])
    expected = np.array([[5.0, 7.0]])

    result = vectors.vector_operation(a, b, lambda x, y: x + y)

    np.testing.assert_allclose(result, expected)


def test_addition_diff_dtype():
    a = np.array([[1, 2]], dtype=np.int32)
    b = np.array([[3, 4]], dtype=np.float64)
    expected = np.array([[4, 6]])

    result = vectors.vector_operation(a, b, lambda x, y: x + y)

    np.testing.assert_allclose(result, expected)


# ============================================================
# Scalar multiplication
# ============================================================

def test_scalar_multiplication():
    a = np.array([1, 2, 3])
    scalar = 2
    expected = np.array([2, 4, 6])

    result = vectors.scalar_multipication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_neg_scalar_multiplication():
    a = np.array([1, 2, 3])
    scalar = -2
    expected = np.array([-2, -4, -6])

    result = vectors.scalar_multipication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_zero_scalar():
    a = np.array([1, 2, 3])
    scalar = 0
    expected = np.array([0, 0, 0])

    result = vectors.scalar_multipication(a, scalar)

    np.testing.assert_array_equal(result, expected)


def test_float_scalar_multiplication():
    a = np.array([1, 2, 3])
    scalar = 2.1
    expected = np.array([2.1, 4.2, 6.3])

    result = vectors.scalar_multipication(a, scalar)

    np.testing.assert_allclose(result, expected)


def test_vector_passed_as_scalar():
    a = np.array([1, 2, 3])
    scalar = np.array([[1, 2],
                       [3, 4]])

    with pytest.raises(TypeError):
        vectors.scalar_multipication(a, scalar)


def test_string_passed_as_scalar():
    a = np.array([1, 2, 3])
    scalar = "1.1"

    with pytest.raises(TypeError):
        vectors.scalar_multipication(a, scalar)


def test_boolean_scalar():
    a = np.array([1, 2, 3])
    scalar = True
    expected = np.array([1, 2, 3])

    result = vectors.scalar_multipication(a, scalar)

    np.testing.assert_allclose(result, expected)


# ============================================================
# Dot product
# ============================================================

def test_dot_product():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    expected = 14

    result = vectors.dot_product(a, b)

    np.testing.assert_array_equal(result, expected)


def test_dot_product_diff_ndim():
    a = np.array([1, 2, 3])
    b = np.array([[1, 2, 3]])

    with pytest.raises(ValueError):
        vectors.dot_product(a, b)


def test_dot_product_same_ndim_diff_shape():
    a = np.array([1, 2, 3])
    b = np.array([1, 2])

    with pytest.raises(ValueError):
        vectors.dot_product(a, b)


def test_dot_product_scalar():
    a = np.array([1, 2, 3])
    b = 1

    with pytest.raises(TypeError):
        vectors.dot_product(a, b)


def test_dot_product_empty_vector():
    a = np.array([1, 2, 3])
    b = np.array([])

    with pytest.raises(ValueError):
        vectors.dot_product(a, b)


def test_dot_product_orthogonal():
    a = np.array([1, 0])
    b = np.array([0, 1])
    expected = 0

    result = vectors.dot_product(a, b)

    np.testing.assert_array_equal(result, expected)


# ============================================================
# Norms
# ============================================================

def test_norm_empty_vector():
    a = np.array([])

    with pytest.raises(ValueError):
        vectors.norm(a, norm_kind="l2")


def test_norm_matrix():
    a = np.array([[1, 2],
                  [4, 5]])

    # Keep this only if your norm() is supposed to reject matrices.
    with pytest.raises(ValueError):
        vectors.norm(a, norm_kind="l2")


def test_norm_zero_vector():
    a = np.array([0, 0])
    expected = 0

    result = vectors.norm(a, norm_kind="l2")

    np.testing.assert_equal(result, expected)


def test_l2_norm():
    a = np.array([1, 2, 3])
    expected = np.sqrt(14)

    result = vectors.norm(a, norm_kind="l2")

    np.testing.assert_allclose(result, expected)


def test_l1_norm():
    a = np.array([1, -2, 3, -4])
    expected = 10

    result = vectors.norm(a, norm_kind="l1")

    np.testing.assert_equal(result, expected)


def test_linf_norm():
    a = np.array([1, 2, 3, -56, 34])
    expected = 56

    result = vectors.norm(a, norm_kind="l-inf")

    np.testing.assert_equal(result, expected)


def test_invalid_norm_kind():
    a = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        vectors.norm(a, norm_kind="invalid")


# ============================================================
# Distance
# ============================================================

def test_distance():
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 3, 4, 5])
    expected = 2.0

    result = vectors.distance(a, b)

    np.testing.assert_allclose(result, expected)


def test_distance_more_dimensional():
    a = np.array([[[1, 2],
                   [3, 4]]])
    b = np.array([[[2, 3],
                   [4, 5]]])

    with pytest.raises(ValueError):
        vectors.distance(a, b)


def test_distance_empty_vector():
    a = np.array([])
    b = np.array([2, 3, 4, 5])

    with pytest.raises(ValueError):
        vectors.distance(a, b)


def test_distance_matrix():
    a = np.array([[1, 2],
                  [4, 5]])
    b = np.array([[1, 2],
                  [4, 5]])

    with pytest.raises(ValueError):
        vectors.distance(a, b)


def test_distance_zero_vector():
    a = np.array([0, 0, 0, 0])
    b = np.array([-1, -1, -1, -1])
    expected = 2.0

    result = vectors.distance(a, b)

    np.testing.assert_allclose(result, expected)


def test_distance_scalar():
    a = 45
    b = np.array([-1, -1, -1, -1])

    with pytest.raises((TypeError, ValueError)):
        vectors.distance(a, b)


def test_distance_symmetry():
    a = np.array([2, 2, 2, 2])
    b = np.array([-1, -1, -1, -1])

    result1 = vectors.distance(a, b)
    result2 = vectors.distance(b, a)

    np.testing.assert_allclose(result1, result2)


def test_distance_different_shapes():
    a = np.array([1, 2, 3])
    b = np.array([1, 2])

    with pytest.raises(ValueError):
        vectors.distance(a, b)


# ============================================================
# Normalization
# ============================================================

def test_normalization():
    # Basic normalization
    vector = np.array([3, 4])
    result = vectors.normalization(vector)
    expected = np.array([0.6, 0.8])

    np.testing.assert_allclose(result, expected)

    # Negative / mixed values
    vector = np.array([-3, 4])
    result = vectors.normalization(vector)
    expected = np.array([-0.6, 0.8])

    np.testing.assert_allclose(result, expected)

    # Float input
    vector = np.array([1.5, 2.5])
    result = vectors.normalization(vector)

    expected_norm = np.sqrt(1.5**2 + 2.5**2)
    expected = vector / expected_norm

    np.testing.assert_allclose(result, expected)

    # Zero vector
    vector = np.array([0, 0, 0])

    with pytest.raises(ValueError):
        vectors.normalization(vector)

    # Empty vector
    vector = np.array([])

    with pytest.raises(ValueError):
        vectors.normalization(vector)

    # Output must have L2 norm = 1
    vector = np.array([3, 4])
    result = vectors.normalization(vector)

    assert np.isclose(
        vectors.norm(result, norm_kind="l2"),
        1.0
    )

    # Shape preservation
    vector = np.array([1, 2, 3])
    result = vectors.normalization(vector)

    assert result.shape == vector.shape

    # Direction preservation
    vector = np.array([3, 4])
    result = vectors.normalization(vector)

    np.testing.assert_allclose(result * 5, vector)

    # Single-component positive vector
    vector = np.array([5])
    result = vectors.normalization(vector)

    np.testing.assert_allclose(result, np.array([1.0]))

    # Single-component negative vector
    vector = np.array([-5])
    result = vectors.normalization(vector)

    np.testing.assert_allclose(result, np.array([-1.0]))


# ============================================================
# Cosine similarity
# ============================================================

def test_cosine_similarity():
    # Same direction → 1
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([2, 4, 6])

    result = vectors.cosine_similarity(vector1, vector2)

    assert np.isclose(result, 1.0)

    # Opposite direction → -1
    vector1 = np.array([1, 2])
    vector2 = np.array([-1, -2])

    result = vectors.cosine_similarity(vector1, vector2)

    assert np.isclose(result, -1.0)

    # Perpendicular vectors → 0
    vector1 = np.array([1, 0])
    vector2 = np.array([0, 1])

    result = vectors.cosine_similarity(vector1, vector2)

    assert np.isclose(result, 0.0)

    # General case
    vector1 = np.array([1, 0])
    vector2 = np.array([1, 1])

    result = vectors.cosine_similarity(vector1, vector2)

    assert np.isclose(result, 1 / np.sqrt(2))

    # Negative / mixed values
    vector1 = np.array([-1, 2])
    vector2 = np.array([3, -4])

    result = vectors.cosine_similarity(vector1, vector2)

    expected = (-11) / (np.sqrt(5) * 5)

    assert np.isclose(result, expected)

    # Different magnitudes, same direction → 1
    vector1 = np.array([3, 4])
    vector2 = np.array([30, 40])

    result = vectors.cosine_similarity(vector1, vector2)

    assert np.isclose(result, 1.0)

    # Zero vector → ValueError
    vector1 = np.array([0, 0])
    vector2 = np.array([1, 2])

    with pytest.raises(ValueError):
        vectors.cosine_similarity(vector1, vector2)

    # Different shapes → ValueError
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([1, 2])

    with pytest.raises(ValueError):
        vectors.cosine_similarity(vector1, vector2)

    # Empty vectors → ValueError
    vector1 = np.array([])
    vector2 = np.array([])

    with pytest.raises(ValueError):
        vectors.cosine_similarity(vector1, vector2)

    # Symmetry
    vector1 = np.array([2, 3])
    vector2 = np.array([4, -1])

    result1 = vectors.cosine_similarity(vector1, vector2)
    result2 = vectors.cosine_similarity(vector2, vector1)

    assert np.isclose(result1, result2)