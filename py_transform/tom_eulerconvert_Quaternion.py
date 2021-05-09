import numpy as np
import os
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_transform.tom_sum_rotation import tom_sum_rotation
def tom_eulerconvert_Quaternion(angles, rotFlav = 'zxz'):  #the input should be 2D array, not 1D array
    '''
    
    TOM_EULERCONVERT_QUATERNION converts a euler angles to quaternions

    Q=tom_eulerconvert_Quaternion(angels,rotFlav)
   
    tom_eulerconvert_Quaternion converts a matrix of euler ang to a matrix
    of Quaternions

    PARAMETERS

    INPUT
       angels                 nx3 matrix of euler angles
       rotFlav                 ('zxz') or 'zyz'

    OUTPUT
       euler_out            resulting Euler angles

    EXAMPLE
       euler_out=tom_eulerconvert_Quaternion(np.array([[10 20 30], [0 0 20]]);
      
    '''
    Q = np.zeros([angles.shape[0], 4 ])
    
    i = 0
    for single_angle in angles: #single_angle is 1D array
        if rotFlav == 'zyz':
            _, angTmp = tom_eulerconvert_xmipp(single_angle[0], single_angle[1],
                                               single_angle[2])
        else:
            angTmp = single_angle
        
        _,_,R = tom_sum_rotation(angTmp, np.array([0,0,0]))
        r,c = R.shape
        if (r!=3) | (c!=3):
            print('Error: R must be a 3x3 matrix')
            os._exit(-1)
            
        Rxx = R[0,0]; Rxy = R[0,1]; Rxz = R[0,2]
        Ryx = R[1,0]; Ryy = R[1,1]; Ryz = R[1,2]
        Rzx = R[2,0]; Rzy = R[2,1]; Rzz = R[2,2]
        #calculate rotation angle
        w = np.sqrt(np.trace(R) + 1)/2
        #check if w is real number. If it is, then zero it
        if np.iscomplex(w) == True:
            w = 0
        
        x = np.sqrt( 1 + Rxx - Ryy - Rzz ) / 2
        y = np.sqrt( 1 + Ryy - Rxx - Rzz ) / 2
        z = np.sqrt( 1 + Rzz - Ryy - Rxx ) / 2

        idx  = np.argmax( [w,x,y,z] )

        if( idx == 0 ):
            x = ( Rzy - Ryz ) / (4*w)
            y = ( Rxz - Rzx ) / (4*w)
            z = ( Ryx - Rxy ) / (4*w)   

        if( idx == 1 ):
            w = ( Rzy - Ryz ) / (4*x)
            y = ( Rxy + Ryx ) / (4*x)
            z = ( Rzx + Rxz ) / (4*x)

        if( idx == 2 ):
            w = ( Rxz - Rzx ) / (4*y)
            x = ( Rxy + Ryx ) / (4*y)
            z = ( Ryz + Rzy ) / (4*y)

        if( idx == 3 ):
            w = ( Ryx - Rxy ) / (4*z)
            x = ( Rzx + Rxz ) / (4*z)
            y = ( Ryz + Rzy ) / (4*z)
            
        Q[i,:] = np.array([w,x,y,z])
        i += 1
        
    return Q

