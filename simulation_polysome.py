import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive

def simulation_polysome(euler_angle, shift_coordinate, cycle_n, order = "trans_rot"):
    psi,phi,theta = euler_angle
    collect_coordinatesArray = np.zeros([cycle_n,3],dtype = np.float)   
    #zxz plane
    rotation_matrix = np.dot( np.dot(np.array([[np.cos(psi), -np.sin(psi), 0],
                                           [np.sin(psi), np.cos(psi), 0],
                                           [0, 0, 1]]),
                                  np.array([[1, 0, 0],
                                           [0, np.cos(phi),-np.sin(phi)],
                                           [0, np.sin(phi), np.cos(phi)]])),
                                  np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta), np.cos(theta), 0],
                                           [0, 0, 1]]))
    if order == "trans_rot": 
        coordinate_cycle = np.zeros(3,dtype = np.float)
        for i in range(cycle_n):   
            new_coordinate = np.dot(np.linalg.matrix_power(rotation_matrix,i), 
                                    shift_coordinate)
            coordinate_cycle = new_coordinate + coordinate_cycle        
            collect_coordinatesArray[i,:] = coordinate_cycle
 
    else:
        coordinate_cycle = np.zeros(3,dtype = np.float)
        for i in range(cycle_n):
            new_coordinate = np.dot(np.linalg.matrix_power(rotation_matrix,i+1), 
                                    shift_coordinate)
            coordinate_cycle = new_coordinate + coordinate_cycle
            collect_coordinatesArray[i,:] = coordinate_cycle
                
    return collect_coordinatesArray
        

if __name__ == '__main__':
    
    euler_angle = np.array([45,0,0])  #zxz planes
    shift_coordinate = np.array([2,2,0]) #x-y-z in any coordinate system
    cycle_n = 8
    collect_coordinatesArray = simulation_polysome(euler_angle, 
                                                   shift_coordinate,
                                                   cycle_n)
    #plot the results 
    #%matplotlib qt (put this sentence in the ipython to interactively see it)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(collect_coordinatesArray[:,0],
               collect_coordinatesArray[:,1],
               collect_coordinatesArray[:,2],'gray')
    ax1.scatter3D(collect_coordinatesArray[:,0],
               collect_coordinatesArray[:,1],
               collect_coordinatesArray[:,2], color = 'red')
    ax1.set_xlim([-2,2])
    ax1.set_ylim([-2,2])
    ax1.set_zlim([-2,2])
    plt.show()

        