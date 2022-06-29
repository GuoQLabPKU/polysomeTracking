import numpy as np
import os
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_transform.tom_sum_rotation import tom_sum_rotation

def tom_angular_distance(euler1, euler2, conv = 'zxz'):
    '''
    TOM_ANGULAR_DISTANCE calculates the angular distance between 2 rotations defined 
    in ZXZ plane
    angDist=tom_angular_distance(euler1,euler2,conv)

    PARAMETERS

    INPUT
        euler1           euler angle 1
        euler2           euler angle 2
        conv             ('zxz') convention 4 roation
                          or zyz

    EXAMPLE
   
        distInZXZ=tom_angular_distance([91 162 272],[85 153 251]);
  
    check 4 zyz
    [~,euler1ZYZ]=tom_eulerconvert_xmipp(91,162,272,'tom2xmipp');
    [~,euler2ZYZ]=tom_eulerconvert_xmipp(85,153,251,'tom2xmipp');
    distInZYZ=tom_angular_distance(euler1ZYZ,euler2ZYZ,'zyz');

    REFERENCES  
    '''
    if conv == 'zyz':
        _, euler1 = tom_eulerconvert_xmipp(euler1[0], euler1[1], euler1[2])
        _, euler2 = tom_eulerconvert_xmipp(euler2[0], euler2[1], euler2[2])
    #calculate the rotation matrix
    _, _, M1 = tom_sum_rotation(euler1, np.zeros(3))
    _, _, M2 = tom_sum_rotation(euler2, np.zeros(3))
    #calculate the relative rotation matrix
    Mrel = np.dot(np.linalg.inv(M1), M2)
    #calculate quaternions
    Qrel,_,_,_ = qGetQInt(Mrel)
    #calculate the angle 
    angDist = np.arccos(Qrel)/np.pi*180*2
    if angDist > 180:
        angDist = 360 - angDist
    if angDist < -180:
        angDist = 360 + angDist
        
    return angDist
    
def qGetQInt(R):
    '''
    qGetQ: convert 3x3 rotation matrix into quaternion of ZXZ plane
    
    '''
    r, c = R.shape
    if (r!=3) | (c!=3):
        print("Error: R must be 3x3 matrix/array!")
        os._exit(1)
    else:
        Rxx = R[0,0]; Rxy = R[0,1]; Rxz = R[0,2]
        Ryx = R[1,0]; Ryy = R[1,1]; Ryz = R[1,2]
        Rzx = R[2,0]; Rzy = R[2,1]; Rzz = R[2,2]
        #calculate rotation angle
        w = np.sqrt(abs(np.trace(R) + 1))/2
        #check if w is real number. If it is, then zero it
        if np.iscomplex(w) == True:
            w = 0
        
        
        x = np.sqrt( abs(1 + Rxx - Ryy - Rzz) ) / 2
        y = np.sqrt( abs(1 + Ryy - Rxx - Rzz )) / 2
        z = np.sqrt( abs(1 + Rzz - Ryy - Rxx) ) / 2

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
                    
    return w, x, y, z        
        
        
        
        