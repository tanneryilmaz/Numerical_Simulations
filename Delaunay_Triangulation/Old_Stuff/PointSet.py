import numpy as np

def generate_rand_points(dimension_amplitudes, num_points):
    
    num_dimensions = len(dimension_amplitudes)
    
    points = np.random.rand(num_points, num_dimensions)
    
    #this loop scales each dimension 
    for i in range(num_dimensions):
        points[:, i] *= dimension_amplitudes[i]
    
    return points     
