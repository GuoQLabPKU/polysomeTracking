import numpy as np
def tom_pointrotate(r, phi, psi, the):
    '''
    TOM_POINTROTATE rotates point (= 3d vector)

    r = tom_pointrotate(r,phi,psi,the)

    A vector in 3D is rotated around/related with the origin = [0 0 0]. The puropose is
    for example to predict the location of a point in a volume after
    rotating it with tom_rotate3d. Take care that the coordinates are with
    respect to the origin!

    PARAMETERS

    INPUT
        r                   3D vector - e.g. np.array([1, 1, 1,])
        phi                 Euler angle - in deg.
        psi                 Euler angle - in deg.
        the                 Euler angle - in deg.
  
    OUTPUT
        r                   one dimension array

    EXAMPLE
        r = np.array([1, 1, 1,])
        r = tom_pointrotate(r,10,20,30)

    REFERENCES
    '''
    tom_phi = phi/180*np.pi
    tom_psi = psi*np.pi/180
    tom_theta = the*np.pi/180
    #calculate the roation matrix
    rotarray = np.linalg.multi_dot([
            np.array([[np.cos(tom_psi), -np.sin(tom_psi), 0],
                      [np.sin(tom_psi), np.cos(tom_psi), 0],
                      [0, 0, 1]]),
            np.array([[1, 0, 0],
                      [0, np.cos(tom_theta),-np.sin(tom_theta)],
                      [0, np.sin(tom_theta), np.cos(tom_theta)]]),
           np.array([[np.cos(tom_phi), -np.sin(tom_phi), 0],
                     [np.sin(tom_phi), np.cos(tom_phi), 0],
                     [0, 0, 1]])])
       
    r = np.dot(rotarray, r)
    return r