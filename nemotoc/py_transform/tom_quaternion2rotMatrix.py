import numpy as np

def tom_quaternion2rotMatrix(Qrotation):
    '''
    
    TOM_QUATERNION2ROTMATRIX calculates rotation matrix from Quat
    R = tom_quaternion2rotMatrix(Qrotation)

    PARAMETERS

    INPUT
        Qrotation         input Quaternion #(1D array)
  
    OUTPUT
        R                 rotation Matrix
                          
    EXAMPLE
  
    R = tom_quaternion2rotMatrix(np.array([0.9330,    0.2578 ,   0.0226 ,   0.2500]));
    
    '''
    w = Qrotation[0]
    x = Qrotation[1]
    y = Qrotation[2]
    z = Qrotation[3]
    
    Rxx = 1 - 2*(y**2 + z**2)
    Rxy = 2*(x*y - z*w)
    Rxz = 2*(x*z + y*w)
    Ryx = 2*(x*y + z*w)
    Ryy = 1 - 2*(x**2 + z**2)
    Ryz = 2*(y*z - x*w )
    Rzx = 2*(x*z - y*w )
    Rzy = 2*(y*z + x*w )
    Rzz = 1 - 2 *(x**2 + y**2)
    R = np.array([ 
        [Rxx,    Rxy,    Rxz],
        [Ryx,    Ryy,    Ryz],
        [Rzx,    Rzy,    Rzz]])
    
    return R
    
  

