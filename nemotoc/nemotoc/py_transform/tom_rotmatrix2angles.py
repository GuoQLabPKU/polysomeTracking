import numpy as np
def tom_rotmatrix2angles(rott):
    '''
    
    TOM_ROTMATRIX2ANGLES converts a rotation matrix to the corresponding ...

    [euler_out] = tom_rotmatrix2angles(rott)

    TOM_ROTMATRIX2ANGLES converts a rotation matrix to the corresponding
    euler angles. The rotation matrix has to be given in the zxz form

    PARAMETERS

    INPUT
        rott                 3x3 zxz rotation matrix
  
    OUTPUT
        euler_out            resulting Euler angles   

    ''' 
    euler_out = np.array([np.arctan2(rott[2,0], rott[2,1]),
                     np.arctan2(rott[0,2], -rott[1,2]),
                     np.arccos(rott[2,2])])
    euler_out = np.round(euler_out*180/np.pi,4)
        
    if abs(rott[2,2] - 1) < 10e-8:
        euler_out = np.array([round(np.arctan2(rott[1,0], rott[0,0])*180/np.pi,4),
                         0, 0])
    return euler_out