import matplotlib.pyplot as plt
import numpy as np
from math import acos

DEG = np.pi / 180

def wire_limits(model, angles):    
        # z limits
        matrix = model.get_transition_matrix(angles, "nominal")
        z_dir = matrix[0:3, 2]
        scalar_z = np.vdot(z_dir, -model.zero_wire_direction)
        
        #np.clip(scalar_z, -1.0, 1.0)
        #alpha_z = acos(scalar_z)
        #angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit

        wire = matrix[0:3, 3] - model.zero_offset_nominal
        wire_len = np.linalg.norm(wire) 
        scalar_wire = np.vdot(-(wire / wire_len), z_dir)
        np.clip(scalar_wire, -1, 1)
        alpha_z = acos(scalar_wire)
        angle_cons_z = np.array([alpha_z, model.angle_limit_z - alpha_z]) # so angle would fit
        
        # wire limits        
        len_cons = np.array([wire_len - model.wire_limits[0], model.wire_limits[1] - wire_len]) # so wire len would fit

        scalar_wire = np.vdot(wire / wire_len, model.zero_wire_direction)  # > 0 so wire stretch forward        
        alpha = acos(scalar_wire)
        angle_cons_wire = np.array([alpha, model.angle_limit_wire - alpha]) # so angle would fit

        return np.concatenate(([scalar_z], [scalar_wire], angle_cons_z,  [scalar_wire], angle_cons_wire, len_cons)) #angle_cons_wire,


a=np.array([14.654,23.941,141.203,-75.547,91.341,5.336]) * DEG
b=np.array([14.716,23.986,141.223,-75.396,89.248,-0.021]) * DEG
c=np.array([14.880,23.951,141.195,-75.698,89.514,-55.134]) * DEG
d=np.array([14.538,23.984,141.249,-75.926,90.060,25.401]) * DEG
e=np.array([14.847,23.899,141.374,-76.156,89.341,-70.747]) * DEG

print(wire_limits(a, 1))


