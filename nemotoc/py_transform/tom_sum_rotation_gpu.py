import cupy as np

def tom_sum_rotation(rots):
    '''
    %TOM_SUM_ROTATION sums up  N 3-tupel of Euler angles 

    [rott]=tom_sum_rotation(rots,shifts,order)

    TOM_SUM_ROTATION sums up  N 3-tupel of Euler angles
    to only one rotation 

    PARAMETERS

    INPUT
        rots                N x 3 array with Euler angles, like this:
                            rots = [phi[1] psi[1] theta[1]; ... phi[N] psi[N]
                            theta[N]];


   OUTPUT
   
       rott                resulting rotation matrix

   EXAMPLE
   rott=tom_sum_rotation(np.array([[10, 20, 30], [-20, -10, -30]]))
 
   '''
   #check if the iput is one dimention 
    try:
        row_n = rots.shape[0]
        col_n = rots.shape[1]
    except IndexError:
        row_n = 1
        col_n = rots.shape[0]
        rots =  rots.reshape(1,-1)

    rotM = np.zeros([row_n,3,3],dtype = np.float)

    for i in range(row_n):
        tom_phi = rots[i,0]*np.pi/180
        tom_psi = rots[i,1]*np.pi/180
        tom_theta = rots[i,2]*np.pi/180
        #use zxz matrix
        rotM[i,:,:] = np.dot(np.dot(
              np.array([[np.cos(tom_psi).item(), -np.sin(tom_psi).item(), 0], 
                        [np.sin(tom_psi).item(), np.cos(tom_psi).item(), 0], 
                        [0, 0, 1]]),
              np.array([[1, 0, 0],
                         [0, np.cos(tom_theta).item(),-np.sin(tom_theta).item()],
                         [0, np.sin(tom_theta).item(), np.cos(tom_theta).item()]])),
               np.array([[np.cos(tom_phi).item(), -np.sin(tom_phi).item(), 0],
                         [np.sin(tom_phi).item(), np.cos(tom_phi).item(), 0],[0, 0, 1]]))
        
    #sum up the rotations
    rott = np.eye(col_n)
    z  = row_n
    
    for i in range(row_n):
        rott = np.dot(rott,rotM[z-1,:,:])
        z -= 1
        
    return  rott        
        

            
        
    
           
           
           
