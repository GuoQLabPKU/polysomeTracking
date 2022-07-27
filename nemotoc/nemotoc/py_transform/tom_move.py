import numpy as np

def tom_move(image_input, move_array):
    '''
    
    TOM_MOVE(Image, [dx,dy,dz]). This function moves image given in X,Y,Z direction.
    the coordinates of the first pixel is now (dx+1,dy+1,dz+1). The rest of the image 
    is filled with zeros.
    
    
    PARAMETERS
    
    image_input   2D/3D array
    move_array    1D array. like np.array([dx,dy,dz]) OR np.aray([dx,dy])
    
    EXAMPLE
    
    a = np.arange(1,50).reshape(7,7)
    a = array([[ 1,  2,  3,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19, 20, 21],
       [22, 23, 24, 25, 26, 27, 28],
       [29, 30, 31, 32, 33, 34, 35],
       [36, 37, 38, 39, 40, 41, 42],
       [43, 44, 45, 46, 47, 48, 49]])
    c = tom_move(a,np.array([3,3]))
    c = array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  2.,  3.,  4.],
       [ 0.,  0.,  0.,  8.,  9., 10., 11.],
       [ 0.,  0.,  0., 15., 16., 17., 18.],
       [ 0.,  0.,  0., 22., 23., 24., 25.]])   
    
    '''
    dimensions = len(image_input.shape)
    if dimensions < 2:
        raise AttributeError('only 2D/3D images are supported!')
    #if 2D array
    elif dimensions == 2:
        s1,s2 = image_input.shape
        move_dimensions = len(move_array)
        if move_dimensions != 2:
            raise AttributeError('the dimenstion of image and move direction do not match!')
        else:
            if (np.abs(move_array[0]) >= s1) | (np.abs(move_array[1]) >= s2):
                raise RuntimeError('Out of limits move!')
            b = np.zeros((s1,s2), order = 'F')
            if (move_array[0] >= 0) & (move_array[1] >= 0):
                atmp = image_input[0:(s1-move_array[0]), 0:(s2-move_array[1])]
                b[move_array[0]:s1, move_array[1]:s2] = atmp
                c = b
            elif (move_array[0] < 0) & (move_array[1] >= 0):
                atmp = image_input[np.abs(move_array[0]):s1, 0:(s2-move_array[1])]
                satmp = atmp.shape
                b[0:satmp[0], move_array[1]:s2] = atmp
                c = b
            elif (move_array[0] >= 0) & (move_array[1] < 0):
                atmp = image_input[0:(s1-move_array[0]), np.abs(move_array[1]):s2]
                satmp = atmp.shape
                b[move_array[0]:s1, 0:satmp[1]] = atmp
                c = b
            else:
                atmp = image_input[np.abs(move_array[0]):s1, np.abs(move_array[1]):s2]
                satmp = atmp.shape
                b[0:satmp[0], 0:satmp[1]] = atmp
                c = b
        
    #if 3D -array 
    else:
        s1,s2,s3 = image_input.shape #s1=>z;s2=>x;s3=>y, three direction!
        move_dimensions = len(move_array) #elements in move_array:x,y,z
        if move_dimensions != 3:
            raise AttributeError('the dimenstion of image and move direction do not match!')
        else:
            if (np.abs(move_array[2]) >= s1) | (np.abs(move_array[0]) >= s2) | (np.abs(move_array[1]) >= s3):
                raise RuntimeError('Out of limits move')
            c = np.zeros((s1,s2,s3), order = 'F')
            b = np.zeros((s2,s3), order = 'F')
            if move_array[2] >= 0:
                for i in range(s1-move_array[2]):                 
                    if (move_array[0] >= 0) & (move_array[1] >= 0):
                        atmp = image_input[i,0:(s2-move_array[0]), 0:(s3-move_array[1])]
                        b[move_array[0]:s2, move_array[1]:s3] = atmp
                        c[i+move_array[2],:,:] = b
                    elif (move_array[0] < 0) & (move_array[1] >= 0):
                        atmp = image_input[i,np.abs(move_array[0]):s2, 0:(s3-move_array[1])]
                        satmp = atmp.shape
                        b[0:satmp[0], move_array[1]:s3] = atmp
                        c[i+move_array[2],:,:] = b
                    elif (move_array[0] >= 0) & (move_array[1] < 0):
                        atmp = image_input[i,0:(s2-move_array[0]), np.abs(move_array[1]):s3]
                        satmp = atmp.shape
                        b[move_array[0]:s2, 0:satmp[1]] = atmp
                        c[i+move_array[2],:,:] = b
                    else:
                        atmp = image_input[i,np.abs(move_array[0]):s2, np.abs(move_array[1]):s3]
                        satmp = atmp.shape
                        b[0:satmp[0], 0:satmp[1]] = atmp
                        c[i+move_array[2],:,:] = b            
        
            else:
                for i in range(np.abs(move_array[2]), s1):
                    if (move_array[0] >= 0) & (move_array[1] >= 0):
                        atmp = image_input[i,0:(s2-move_array[0]), 0:(s3-move_array[1])]
                        b[move_array[0]:s2, move_array[1]:s3] = atmp
                        c[i-np.abs(move_array[2]),:,:] = b
                    elif (move_array[0] < 0) & (move_array[1] >= 0):
                        atmp = image_input[i,np.abs(move_array[0]):s2, 0:(s3-move_array[1])]
                        satmp = atmp.shape                       
                        b[0:satmp[0], move_array[1]:s3] = atmp
                        c[i-np.abs(move_array[2]),:,:] = b
                    elif (move_array[0] >= 0) & (move_array[1] < 0):
                        atmp = image_input[i,0:(s2-move_array[0]), np.abs(move_array[1]):s3]
                        satmp = atmp.shape
                        b[move_array[0]:s2, 0:satmp[1]] = atmp
                        c[i-np.abs(move_array[2]),:,:] = b
                    else:
                        atmp = image_input[i,np.abs(move_array[0]):s2, np.abs(move_array[1]):s3]
                        satmp = atmp.shape
                        b[0:satmp[0], 0:satmp[1]] = atmp
                        c[i-np.abs(move_array[2]),:,:] = b   
                        
    return c