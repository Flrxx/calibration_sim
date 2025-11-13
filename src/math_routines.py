import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union

def x_rot (angle: Union[int, float]) -> np.ndarray:
    mat = np.array([[1, 0, 0, 0],
                    [0, cos(angle), -sin(angle), 0],
                    [0, sin(angle), cos(angle), 0],
                    [0, 0, 0, 1]],dtype='float')

    return mat

def y_rot(angle: Union[int, float]) -> np.ndarray:
    mat = np.array([[cos(angle), 0, sin(angle), 0],
                    [0, 1, 0, 0],
                    [-sin(angle), 0, cos(angle), 0],
                    [0, 0, 0, 1]],dtype='float')
    return mat

def z_rot(angle: Union[int, float]) -> np.ndarray:
    mat = np.array([[cos(angle), -sin(angle), 0, 0],
                    [sin(angle), cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]],dtype='float')
    return mat

def arbitrary_axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    nu = 1 - cos(angle)
    x, y, z = axis
    mat = np.array([[cos(angle)+(nu*(x**2)),  (nu*x*y)-(sin(angle)*z), (nu*x*z)+(sin(angle)*y)],
                    [(nu*y*x)+(sin(angle)*z), cos(angle)+(nu*(y**2)),  (nu*y*z)-(sin(angle)*x)],
                    [(nu*z*x)-(sin(angle)*y), (nu*z*y)+(sin(angle)*x), cos(angle)+(nu*(z**2))]], dtype='float')
    return mat

def trans(vector: np.ndarray) -> np.ndarray:
    mat = np.array([[1, 0, 0, vector[0]],
                    [0, 1, 0, vector[1]],
                    [0, 0, 1, vector[2]],
                    [0, 0, 0, 1]],dtype='float')
    return mat