import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def tom_fitPlane_from3Dpoint(xyz_array,   if_disp=0, ax=None):
    A = np.c_[xyz_array[:,0], xyz_array[:,1], np.ones(xyz_array.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, xyz_array[:,2]) 
    
    if if_disp:
        xmin,xmax  = np.min(xyz_array[:,0]), np.max(xyz_array[:,0])
        ymin,ymax = np.min(xyz_array[:,1]), np.max(xyz_array[:,1])
        x_p = np.linspace(xmin,xmax,80)
        y_p = np.linspace(ymin,ymax,80)
        x_p,y_p = np.meshgrid(x_p,y_p)
        z_p = C[0]*x_p + C[1]*y_p + C[2]
        
        if ax is None:
            fig = plt.figure(figsize = (10,10))
            ax = fig.add_subplot(111, projection ='3d')
            ax.scatter(xyz_array[:,0], xyz_array[:,1], xyz_array[:,2], s = 60, color=np.array([0.7,0.7,0.7]))
        ax.plot_surface(x_p,y_p,z_p, rstride=10,cstride=10,alpha = 0.5, color = np.array([0.52,0.52,0.52]))
        
        plt.show() 
        
    return np.array([C[0],C[1],C[2]])
    
    
