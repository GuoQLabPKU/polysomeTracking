import numpy as np

def tom_sum_rotation(rots, shifts, order = 'trans_rot'):
    '''
    %TOM_SUM_ROTATION sums up N translations and N 3-tupel of Euler angles 

    [euler_out shift_out rott]=tom_sum_rotation(rots,shifts,order)

    TOM_SUM_ROTATION sums up N translations and N 3-tupel of Euler angles
    to only one rotation and translation

    PARAMETERS

    INPUT
        rots                N x 3 array with Euler angles, like this:
                            rots = [phi[1] psi[1] theta[1]; ... phi[N] psi[N]
                            theta[N]];
        shifts              N x 3 array with translations, like this:
                            shifts = [x[1] y[1] z[1]; ... x[N] y[N] z[N]];
        order               Either translate - rotate (trans_rot) or rotate -
                            translate (rot_trans)

   OUTPUT
       euler_out           resulting Euler angles, one dimension array
       shift_out           resulting translation vector, one dimension array
       rott                resulting rotation matrix

   EXAMPLE
   [euler_out shift_out rott]=tom_sum_rotation(np.array([10 20 30; -20 -10 -30]),np.array([5 5 5; 5 5 5]))
   [euler_out shift_out rott]=tom_sum_rotation([10 20 30; -20 -10 -30],
                                               [5 5 5; -2.9506 -3.7517  -7.2263])
   
   '''
   #check if the iput is one dimention 
    try:
        row_n = rots.shape[0]
        col_n = rots.shape[1]
    except IndexError:
        row_n = 1
        col_n = rots.shape[0]
        rots =  rots.reshape(1,-1)
        shifts = shifts.reshape(1,-1)
    rotM = np.zeros([row_n,3,3],dtype = np.float)
    if order == 'rot_trans':
        for i in range(row_n):
            tom_phi = rots[i,0]*np.pi/180
            tom_psi = rots[i,1]*np.pi/180
            tom_theta = rots[i,2]*np.pi/180
            #use zxz matrix
            rotM[i,:,:] = np.dot( np.dot(np.array([[np.cos(tom_psi), -np.sin(tom_psi), 0],
                                           [np.sin(tom_psi), np.cos(tom_psi), 0],
                                           [0, 0, 1]]),
                                  np.array([[1, 0, 0],
                                           [0, np.cos(tom_theta),-np.sin(tom_theta)],
                                           [0, np.sin(tom_theta), np.cos(tom_theta)]])),
                                  np.array([[np.cos(tom_phi), -np.sin(tom_phi), 0],
                                           [np.sin(tom_phi), np.cos(tom_phi), 0],
                                           [0, 0, 1]]))
        #sum up the shifts
        shift_out = np.zeros(col_n)
        for i in range(row_n):
            z = row_n
            rott = np.eye(col_n)
            for ii in np.arange(row_n,i+1,-1):
                rott = np.dot(rott, rotM[z-1,:,:])
                z -= 1
            shift_out = np.dot(rott, shifts[i,:]) + shift_out
        #sum up the rotations
        rott = np.eye(col_n)
        z  = row_n
        
        for i in range(row_n):
            rott = np.dot(rott,rotM[z-1,:,:])
            z -= 1
        #extract euler angles
        euler_out = np.array([np.arctan2(rott[2,0], rott[2,1]),
                     np.arctan2(rott[0,2],-1*rott[1,2]),
                     np.arccos(rott[2,2])])
        euler_out = np.array([np.round(i*180/np.pi,3) for i in euler_out])
        if abs(rott[2,2] - 1) < 10e-8:
            euler_out = np.array([np.around(np.arctan2(rott[1,0], rott[0,0])*180/np.pi,4),0,0])
        

    
    else:
        for i in range(row_n):
            tom_phi = rots[i,0]*np.pi/180
            tom_psi = rots[i,1]*np.pi/180
            tom_theta = rots[i,2]*np.pi/180
            #use zxz matrix
            rotM[i,:,:] = np.dot( np.dot(np.array([[np.cos(tom_psi), -np.sin(tom_psi), 0],
                                           [np.sin(tom_psi), np.cos(tom_psi), 0],
                                           [0, 0, 1]]),
                                  np.array([[1, 0, 0],
                                           [0, np.cos(tom_theta),-np.sin(tom_theta)],
                                           [0, np.sin(tom_theta), np.cos(tom_theta)]])),
                                  np.array([[np.cos(tom_phi), -np.sin(tom_phi), 0],
                                           [np.sin(tom_phi), np.cos(tom_phi), 0],
                                           [0, 0, 1]]))
        #sum up the shifts
        shift_out = np.zeros(col_n)
        for i in range(row_n):
            z = row_n
            rott = np.eye(col_n)
            for ii in range(row_n-i):
                rott = np.dot(rott, rotM[z-1,:,:])
                z -= 1
            shift_out = np.dot(rott, shifts[i,:]) + shift_out
        #sum up the rotations
        rott = np.eye(col_n)
        z  = row_n
        
        for i in range(row_n):
            rott = np.dot(rott,rotM[z-1,:,:])
            z -= 1
        #extract euler angles
        euler_out = np.array([np.arctan2(rott[2,0], rott[2,1]),
                     np.arctan2(rott[0,2],-1*rott[1,2]),
                     np.arccos(rott[2,2])])
        euler_out = np.array([np.round(i*180/np.pi,3) for i in euler_out])
        if abs(rott[2,2] - 1) < 10e-8:
            euler_out = np.array([np.around(np.arctan2(rott[1,0], rott[0,0])*180/np.pi,4),0,0])
        
        
    return euler_out, shift_out, rott        
        

            
        
    
           
           
           
