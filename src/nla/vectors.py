import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)

def vector_operation(vector1: np.ndarray, vector2: np.ndarray, operation) -> np.ndarray:
    if vector1.shape == (0, ) or vector2.shape == (0, ):
        raise ValueError("Empty vectors provided")
    
    ndim_a = vector1.ndim
    ndim_b = vector2.ndim
    
    if not (ndim_a == ndim_b):
        raise ValueError("Invalid dimensions for vector 1 and vector 2")
    
    shape_a = vector1.shape
    shape_b = vector2.shape
    
    result_vector = np.empty_like(vector1, dtype=np.float32)
    
    if not (shape_a == shape_b):
        logger.error("Vector addition failed: shape mismatch %s != %s",
                                shape_a,
                                shape_b)
        raise ValueError("Invalid shapes for compatibility of vector 1 and vector 2")
    
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
        
    for i, (a, b) in enumerate(zip(vector1.flat, vector2.flat)):
        result_vector.flat[i] = operation(a, b)
            
    return result_vector


def scalar_multipication(vector: np.ndarray, scalar_quantity: int | float) -> np.ndarray: 

    
    if not np.isscalar(scalar_quantity):
        logger.error("Scalar multiplication failed: provided type {%s} for scalar",
                     type(scalar_quantity))
        raise TypeError("A scalar is required for scalar multiplication")
    
    if not isinstance(scalar_quantity, (int, float)):
        raise TypeError("Scalar should be int or float")
    
    _validate_vector(vector)
    
    vector = vector.astype(np.float32)
    shape = vector.shape
    
    for i in range(shape[0]):  
            vector[i] = vector[i] * scalar_quantity
            
    return vector

def _validate_vector(vector: np.ndarray) -> None:
    if vector.ndim > 2:
        raise ValueError("Vector can have at most 2 dimensions")

    if vector.ndim == 2 and vector.shape[0] != 1 and vector.shape[1] != 1:
        raise ValueError("Matrix provided; expected a vector")

def dot_product(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    
    if vector1.size == 0 or vector2.size == 0:
        raise ValueError("An empty vector is not allowed")
    
    if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
        raise TypeError("A vector is required for dot product")
    
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    ndim_a = vector1.ndim
    ndim_b = vector2.ndim
    
    if not (ndim_a == ndim_b):
        raise ValueError("Invalid dimensions for vector 1 and vector 2")

    _validate_vector(vector1)
    _validate_vector(vector2)
    
    if not (vector1.size == vector2.size):
        raise ValueError("Both vectors should contain same number of elements")
    
    vector1 = vector1.reshape(-1,1)
    vector2 = vector2.reshape(-1,1)
    
    result = 0
    
    for i in range(vector1.shape[0]):
            result += vector1[i, 0] * vector2[i, 0]
            
    return result

def norm(vector: np.ndarray, norm_kind:Literal['l1', 'l2', 'l-inf'] = 'l2') -> float:
    
    vector = vector.astype(np.float32)
    vector_dim = vector.ndim
    
    _validate_vector(vector)
    
    if vector_dim > 2:
        raise ValueError("Vector dimensions can only be one, provided: ", vector_dim)
    
    if vector.size == 0:
        raise ValueError("At least one component is required")
    
    match norm_kind:
        case "l2":
            sq_vector = np.square(vector)
            sq_vector_sum = np.sum(sq_vector)
            l2_norm = np.sqrt(sq_vector_sum)
            return l2_norm
            
        case "l1":
            abs_vector = np.abs(vector)
            l1_norm = np.sum(abs_vector)
            return l1_norm
            
        case "l-inf":
            abs_vector = np.abs(vector)
            l_inf = np.max(abs_vector)
            return l_inf

         
def distance(vector1: np.ndarray, vector2: np.ndarray, norm_kind:Literal['l1', 'l2', 'l-inf'] = 'l2') -> int | float:
    if np.isscalar(vector1) or np.isscalar(vector2):
        raise ValueError("Expected vector; got scalar")
   
    _validate_vector(vector1)
    _validate_vector(vector2)
    
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    vector1_dim = vector1.ndim
    vector2_dim = vector2.ndim
    
    if not (vector1_dim == vector2_dim):
        raise ValueError("Different dimensional vectors are not allowed")
    
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must have the same shape")
    
    dist_vector1_vector2 = vector_operation(vector1, vector2,
                                            operation=lambda x,y: x-y)
    
    return norm(dist_vector1_vector2, norm_kind)


def normalization(vector:np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    l2_norm = norm(vector=vector, norm_kind='l2')
    
    if l2_norm == 0:
        raise ValueError("Normalization can not be done, vector's magnitude is 0")
    
    normalized_vector = vector / l2_norm
    return normalized_vector

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    vector1_normalized = normalization(vector1)
    vector2_normalized = normalization(vector2)
    
    cos_theta = dot_product(np.transpose(vector1_normalized), vector2_normalized)
    return float(cos_theta)