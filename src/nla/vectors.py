import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)

def vector_operation(vector1: np.ndarray, vector2: np.ndarray, operation) -> np.ndarray:
    
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    ndim_a = vector1.ndim
    ndim_b = vector2.ndim
    
    if not (ndim_a == 2 and ndim_b == 2):
        raise ValueError("Invalid dimensions for vector 1 and vector 2 (Expected ndim = 2)")
    
    shape_a = vector1.shape
    shape_b = vector2.shape
    
    result_vector = np.empty(shape=shape_a)
    
    if not (shape_a == shape_b):
        logger.error("Vector addition failed: shape mismatch %s != %s",
                                shape_a,
                                shape_b)
        raise ValueError("Invalid shapes for vector 1 and vector 2")
            
    x, y = shape_a
    
        
    for i in range(x):
        for j in range(y):
                result_vector[i][j] = opearation(
                    vector1[i][j],
                    vector2[i][j]
                )
            
    return result_vector


def scalar_multipication(vector: np.ndarray, scalar_quantity: int | float) -> np.ndarray: 
    
    if not np.isscalar(scalar_quantity):
        logger.error("Scalar multiplication failed: provided type {%s} for scalar",
                     type(scalar_quantity))
        raise TypeError("A scalar is required for scalar multiplication")
    
    vector = vector.astype(np.float32)
    shape = vector.shape
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            vector[i,j] = vector[i,j] * scalar_quantity
            
    return vector



def dot_product(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    vector1_shape = vector1.shape
    vector2_shape = vector2.shape
    
    if not (vector1_shape[1] == vector2_shape[0]):
        logger.error("Dot product failed: shape mismatch - columns of vector 1 (%s) != rows of vector 2 (%s)",
                     vector1_shape[1],
                     vector2_shape[0])
        raise ValueError("Invalid shapes for vector 1 and vector 2")
    
    if (vector1_shape[0] != 1 and vector1_shape[1] != 1) or (vector2_shape[0] != 1 and vector2_shape[1] != 1):
        logger.error("Dot product failed: matrix provided")
        raise ValueError("Invalid shapes for vector 1 and 2")
    
    result_vector = np.empty(shape=(vector1_shape[0], vector2_shape[1]))
    
    for i in range(vector1_shape[0]):
        for j in range(vector2_shape[1]):
            z = 0
            for k in range(vector1_shape[1]):
                z += vector1[i,k] * vector2[k, j]
                
            result_vector[i,j] = z
            
    return result_vector

def norm(vector: np.ndarray, norm_kind:Literal['l1', 'l2', 'l-inf'] = 'l2') -> int | float:
    
    vector = vector.astype(np.float32)
    vector_dim = vector.ndim
    
    if vector_dim > 2:
        raise ValueError("Vector dimensions can only be one, provided: ", vector_dim)
    
    if vector.size == 0:
        raise ValueError("At least one component is required")
    
    match norm_kind:
        case "l2":
            sq_vector = np.square(vector)
            sq_vector_sum = np.sum(sq_vector)
            l1_norm = np.sqrt(sq_vector_sum)
            return l1_norm
            
        case "l1":
            abs_vector = np.abs(vector)
            l1_norm = np.sum(abs_vector)
            return l1_norm
            
        case "l-inf":
            abs_vector = np.abs(vector)
            l_inf = np.max(abs_vector)
            return l_inf

         
def distance(vector1: np.ndarray, vector2: np.ndarray, norm_kind:Literal['l1', 'l2', 'l-inf'] = 'l2') -> int | float:
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    vector1_dim = vector1.ndim
    vector2_dim = vector2.ndim
    
    if vector1_dim != 1 or vector2_dim != 1:
        raise ValueError("Vector dimensions can only be one")
    
    if not (vector1_dim == vector2_dim):
        raise ValueError("Both vector's dimensions should be equal dimensions")
    
    dist_vector1_vector2 = vector1 - vector2
    
    match norm_kind:
        case 'l1':
            l1_distance = norm(vector=dist_vector1_vector2, norm_kind='l1')
            return l1_distance
        
        case 'l2':
            l2_distance = norm(vector=dist_vector1_vector2, norm_kind='l2')
            return l2_distance
        
        case 'l-inf':
            l_inf_distance = norm(vector=dist_vector1_vector2, norm_kind='l-inf')
            return l_inf_distance
        

def normalization(vector:np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    l2_norm = norm(vector=vector, norm_kind='l2')
    
    if l2_norm == 0:
        raise ValueError("Normalization can not be done, vector's magnitude is 0")
    
    normalzed_vector = vector / l2_norm
    return normalzed_vector

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    vector1 = vector1.astype(np.float32)
    vector2 = vector2.astype(np.float32)
    
    vector1_normalized = normalization(vector1)
    vector2_normalized = normalization(vector2)
    
    cos_theta = dot_product(np.transpose(vector1_normalized), vector2_normalized)
    return cos_theta