import numpy as np
def tom_eulerconvert_xmipp(rot, tilt, psi, flag = 'xmipp2tom'):
    '''    
    TOM_EULERCONVERT_XMIPP converts euler angles from xmipp(for example, eluer angles from relion) to the TOM
    convention and inverse

    [rotmatrix,euler_out] = tom_eulerconvert_xmipp(rot, tilt, psi)
   
    PARAMETERS

    INPUT
        rot                 input xmipp angle rot (or phi on tom2xmipp)
        tilt                input xmipp angle tilt (or psi on tom2xmipp)
        psi                 input xmipp angle psi (or theta on tom2xmipp)
        flag                (xmipp2tom) flag for direction use tom2xmipp for inverse transform 
   
    OUTPUT
        rotmatrix           resulting rotation matrix
        euler_out           resulting Euler angles(phi,psi,theta) --> one dimension
                            on tom2xmipp its euler(1)=rot; 
                                        euler(2)=tilt; 
                                        euler(3)=psi;

    EXAMPLE
        [rotmatrix,euler_out] = tom_eulerconvert_xmipp(10,20,30);



    %unit test: rot_mat should be the same
   
    [rot_mat_tom euler_tom]=tom_eulerconvert_xmipp(10,20,30); %xmipp2tom
    [rot_mat_xmipp euler_out]=tom_eulerconvert_xmipp(euler_tom(1),euler_tom(2),euler_tom(3),'tom2xmipp'); %tom2xmipp
    rot_mat_tom-rot_mat_xmipp

    (ps: rotmatrix indices are transposed !! have fun)
   
    '''
    if flag == 'xmipp2tom':
        
        rot2 = -psi*np.pi/180
        tilt = -tilt*np.pi/180
        psi = -rot*np.pi/180
        rot = rot2
        #ZYZ plane
        rotarray = np.dot( np.dot( np.array([[np.cos(rot), -np.sin(rot), 0],
                                             [np.sin(rot),  np.cos(rot), 0],
                                             [0, 0, 1]]),                     
                                   np.array([[np.cos(tilt), 0, np.sin(tilt)],
                                             [0, 1, 0],
                                             [-np.sin(tilt), 0, np.cos(tilt)]]) ),
                                   np.array([[np.cos(psi), -np.sin(psi), 0],
                                             [np.sin(psi), np.cos(psi), 0],
                                             [0, 0, 1]]) )
        #extract euler angles
        euler_out = np.array([np.arctan2(rotarray[2,0], rotarray[2,1]),
                     np.arctan2(rotarray[0,2], -rotarray[1,2]),
                     np.arccos(rotarray[2,2])])
        euler_out = np.array([round(i*180/np.pi,4) for i in euler_out])
        
        if abs(rotarray[2,2] - 1) < 10e-8:
            euler_out = [round(np.arctan2(rotarray[1,0], rotarray[0,0])*180/np.pi,4),
                         0, 0]
    else:
        tom_phi = rot*np.pi/180
        tom_psi = tilt*np.pi/180
        tom_theta = psi*np.pi/180
        #ZXZ plane
        rotarray = np.dot( np.dot(np.array([[np.cos(tom_psi), -np.sin(tom_psi), 0],
                                           [np.sin(tom_psi), np.cos(tom_psi), 0],
                                           [0, 0, 1]]),
                                  np.array([[1, 0, 0],
                                           [0, np.cos(tom_theta),-np.sin(tom_theta)],
                                           [0, np.sin(tom_theta), np.cos(tom_theta)]])),
                                  np.array([[np.cos(tom_phi), -np.sin(tom_phi), 0],
                                           [np.sin(tom_phi), np.cos(tom_phi), 0],
                                           [0, 0, 1]]))
        
        if abs(rotarray[2,2]-1) < 10e-8:
            euler_out = np.array([0, 0, round(np.arctan2(rotarray[0,1], rotarray[0,0])*180/np.pi,4)])
        else:
            euler_out = np.array([round(np.arctan2(rotarray[2,1],rotarray[2,0])*180/np.pi,4),
                         round(np.arccos(rotarray[2,2])*180/np.pi,4),
                         round(np.arctan2(rotarray[1,2], -rotarray[0,2])*180/np.pi,4)])
                
    return rotarray, euler_out       