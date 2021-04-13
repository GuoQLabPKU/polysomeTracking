import numpy as np

from py_transform.tom_eulerconvert_Quaternion import tom_eulerconvert_Quaternion
from py_transform.tom_quaternion2rotMatrix import tom_quaternion2rotMatrix
from py_transform.tom_rotmatrix2angles import tom_rotmatrix2angles
def tom_average_rotations(rotations, rotFlav = 'zxz'):
    '''
    
    TOM_AVERAGE_ROTATIONS calculates the averages angle


    [avgRotEuler,avgRotMat]=tom_average_rotations(rotations,rotFlav)
   

    PARAMETERS

    INPUT
       rotations      nx3 matrix of rotations
       rotFlav          ('zxz') or zyz

   
    OUTPUT

       avgRotEuler      average euler angle (zxz)
       avgRotMat         average rotation matrix
                      

   EXAMPLE
   [avgRotEuler,avgRotMat] = tom_average_rotations([10 0 0; 20 0 0]);
   
    '''
    AngsQuat = tom_eulerconvert_Quaternion(rotations, rotFlav)
    A = np.zeros((4,4))
    M  = AngsQuat.shape[0]
    
    for i in range(M):
        q = AngsQuat[i,:]
        q = q.reshape(1,-1)
        A = np.dot(q.transpose(), q) + A
    
    #calculate the engivalue 
    w,v = np.linalg.eig(A)
    idx = w.argmax()
    eigV = v[:,idx]
    avgRotMat = tom_quaternion2rotMatrix(eigV)
    avgRotEuler = tom_rotmatrix2angles(avgRotMat)
    return avgRotEuler, avgRotMat
    

    
        
        