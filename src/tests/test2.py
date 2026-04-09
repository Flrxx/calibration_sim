import matplotlib.pyplot as plt
import numpy as np
from math import acos

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

print(objective(a, 1))
print(objective(b, 1))
print(objective(c, 1))


