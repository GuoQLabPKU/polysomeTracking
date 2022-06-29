import numpy as np

from nemotoc.py_transform.tom_eulerconvert_Quaternion import tom_eulerconvert_Quaternion
from nemotoc.py_transform.tom_quaternion2rotMatrix import tom_quaternion2rotMatrix
from nemotoc.py_transform.tom_rotmatrix2angles import tom_rotmatrix2angles
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
    ifComplex_flag = 0
    
    for i in range(M):
        q = AngsQuat[i,:]
        q = q.reshape(1,-1)
        A = np.dot(q.transpose(), q) + A
    
    #calculate the engivalue 
    w,v = np.linalg.eig(A)
    idx = w.argmax()
    eigV = v[:,idx]
    #analysis the eigV
    a,b,c,d = eigV
    a = np.real(a)
    b = np.real(b)
    c = np.real(c)
    d = np.real(d)
    if np.sum([ np.iscomplex(a), np.iscomplex(b), np.iscomplex(c), np.iscomplex(d)   ]) > 0:
    
        print('warning: find non zero imag complex!')
        ifComplex_flag = 1
                             
    avgRotMat = tom_quaternion2rotMatrix([a,b,c,d])
    avgRotEuler = tom_rotmatrix2angles(avgRotMat)
    return avgRotEuler, avgRotMat, ifComplex_flag
    

    
        
        