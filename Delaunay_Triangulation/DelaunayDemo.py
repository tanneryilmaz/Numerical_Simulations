# Code from here: https://stackoverflow.com/questions/20025784/how-to-visualize-3d-delaunay-triangulation-in-python
# https://stackoverflow.com/questions/3810865/matplotlib-unknown-projection-3d-error

import numpy as np
import matplotlib.pyplot as plt

class PointCharge:
    '''This class represents a point charge'''
    def __init__(self, charge_loc, charge_magnitude):
        '''charge_loc parameter is a numpy array containing charge's coordinates. charge_magnitude is a positive, scalar value'''
        self.charge_loc = charge_loc
        self.charge_magnitude = charge_magnitude
        
        
def plot_delaunay_3D(ax, points, tri):
    edges = collect_edges(tri)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for (i,j) in edges:
        x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
        y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
        z = np.append(z, [points[i, 2], points[j, 2], np.nan])
    ax.plot3D(x, y, z, color='k', lw='0.5')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='k')

    
def collect_edges(tri):
    ''' This function is used by the plot_delaunay_3D function'''
    edges = set()

    def sorted_tuple(a,b):
        return (a,b) if a < b else (b,a)
    # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    for (i0, i1, i2, i3) in tri.simplices:
        edges.add(sorted_tuple(i0,i1))
        edges.add(sorted_tuple(i0,i2))
        edges.add(sorted_tuple(i0,i3))
        edges.add(sorted_tuple(i1,i2))
        edges.add(sorted_tuple(i1,i3))
        edges.add(sorted_tuple(i2,i3))
    return edges


def centroid(tri, index):
    '''returns the coordinate of the centroid of the triangle at the specified index in tri.simplices'''
    
    simplex_points = tri.points[tri.simplices[index]]
    
    return np.mean(simplex_points, axis = 0)


def generate_rand_points(dimension_amplitudes, num_points):
    '''Generates a set of random points in N dimensions'''
  
    num_dimensions = len(dimension_amplitudes)
    
    points = np.random.rand(num_points, num_dimensions)
    
    #this loop scales each dimension 
    for i in range(num_dimensions):
        points[:, i] *= dimension_amplitudes[i]
    
    return points


def random_linear_points(x_grid_amp, y_grid_amp, num_points, point_charge, offset):
    '''returns linearly spaced points emanating outward from point charge. offset parameter represents distance of the first data
    point from point charge. If offset is small, the first test data point will basically be very close to the point charge. num_points
    is the number of test points you would like to create.'''
    x = np.linspace(point_charge.charge_loc[0] + offset, 0.8 * x_grid_amp, num_points)
    y = np.linspace(point_charge.charge_loc[1] + offset, 0.8 * y_grid_amp, num_points)
    
    return np.array(list(zip(x, y)))

def random_circular_points(num_points, x_grid_amp, y_grid_amp):
    theta = np.linspace(0, 2*np.pi, num_points)
    r = (x_grid_amp / 2) * 0.8 * np.random.rand((num_points))
    points = np.array(list(zip(r * np.cos(theta) + x_grid_amp / 2, r * np.sin(theta) + y_grid_amp / 2)))
    
    return points


def plot_delaunay_2D(tri):
    '''A delaunay triangulation should be passed as an argument'''
    plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices.copy())
    plt.plot(tri.points[:,0], tri.points[:,1], 'o') # plotting corner points

    
def electric_field(point_charge, points):
    '''takes a PointCharge object and an array of points as arguments. Returns an array containing the electric field at each point 
    This function is intended to be used for computing the electric field at the points in a delaunay triangulation'''
    
    k = 8.99e9 # units of (N*m2)/C2
    q = point_charge.charge_magnitude
    
    distance = np.linalg.norm(points - point_charge.charge_loc, axis = 1) # distance between each point and the point charge
    
    E_field = (k * q) / (distance ** 2)
    
    return E_field
    
  

    
                                     