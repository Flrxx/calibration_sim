import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union

def dh_trans(params: Union[np.ndarray, list], angle: Union[int, float]) -> np.ndarray:
    a, alpha, d, theta_offtet, _ = params
    sa = sin(alpha)
    ca = cos(alpha)
    sq = sin(theta_offtet + angle)
    cq = cos(theta_offtet + angle)
    mat = np.array([[cq, -ca*sq,  sa*sq, a*cq],
                    [sq,  ca*cq, -sa*cq, a*sq],
                    [ 0,     sa,     ca,    d],
                    [ 0,      0,      0,    1]], dtype='float')
    return mat

def hayati_trans(params: Union[np.ndarray, list], angle: Union[int, float]) -> np.ndarray:
    a, alpha, beta, theta_offtet, _ = params
    sa = sin(alpha)
    ca = cos(alpha)
    sb = sin(beta)
    cb = cos(beta)
    sq = sin(theta_offtet + angle)
    cq = cos(theta_offtet + angle)
    mat = np.array([[-sa*sb*sq+cb*cq, -ca*sq,  sa*cb*sq+sb*cq, a*cq],
                    [ sa*sb*cq+cb*sq,  ca*cq, -sa*cb*cq+sb*sq, a*sq],
                    [         -ca*sb,     sa,           ca*cb,    0],
                    [              0,      0,               0,    1]], dtype='float')
    return mat