import numpy as np

from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_io.tom_starread import tom_starread

def tom_extendPoly(riboInfo, avgRot, avgShift, particleStar , pruneRad, 
                   NumAddRibo = 1, xyzborder = None):
    '''
    TOM_EXTENDPOLY add ribosomes to the end of each polysome
    
    EXAMPLE
    addRiboInfo = tom_extendPoly();
    
    PARAMETERS
    
    INPUT
    riboInfo    (nx7,np.array) the tail ribosomes of the whole polysomes
                     np.array([polyId, coordinateX, coordinateY,coordinateZ,phi,psi,theta ])
    avgRot          average rotation for extend long
    avgShift        average shift extend long
    particlStar    starfile wit the whole particles
    pruneRad         the diameter of one ribosome(pixel)
    NumAddRibo       (int) number of ribosomes you want put in the tail of the polysome
    xyzborder       (1x3, np.array) the xmax,ymax,zmax of the tomgram 
    
    '''
    riboN = riboInfo.shape[0]
    fillUpRibos  = np.array([]).reshape(-1, 8)
    fillUpMiddleRibos = np.array([]).reshape(-1, 8)
    for i in range(riboN):
        _, ang1 = tom_eulerconvert_xmipp(riboInfo[i,5], riboInfo[i,6], riboInfo[i,7])
        pos1 = riboInfo[i,2:5]
        compare_array = np.zeros([2,3])
        compare_array[0,:] = avgRot
        compare_array[1,:] = ang1
        #calculate euler angles of relative rotation
        ang2, _, _ = tom_sum_rotation(compare_array, np.zeros([2,3]))
        pos2 = tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) + pos1
        #out of border?
        if xyzborder is not None:
            if np.sum(pos2 > xyzborder) > 0:
                continue  
        #if exist?
        if checkRibo(particleStar,pos2,pruneRad) == 1:
            continue
        
        fillUpMiddleRibo = np.array([]).reshape(-1, 8)
        cycles = NumAddRibo - 1
        
        #add more than one ribosomes at end of each polysome
        while cycles > 0:
            
            ang1 = ang2
            pos1 = pos2
            fillUpMiddleRibo = np.concatenate((fillUpMiddleRibo, 
                                      np.array([[riboInfo[i,0], riboInfo[i,1], 
                                                 pos1[0], pos1[1], pos1[2],
                                                 ang1[0], ang1[1], ang1[2]]])), axis = 0)
            compare_array[1,:] = ang1
            ang2, _, _ =  tom_sum_rotation(compare_array, np.zeros([2,3]))
            pos2 = tom_pointrotate(avgShift, ang1[0], ang1[1], ang1[2]) + pos1
            #out of the border?
            if xyzborder is not None:
                if np.sum(pos2 > xyzborder) > 0:
                    continue 
            if checkRibo(particleStar,pos2,pruneRad) == 1:
                continue
            cycles = cycles - 1
            
        #generate new ribosomes into data 
        fillUpMiddleRibos = np.concatenate((fillUpMiddleRibos, fillUpMiddleRibo), axis = 0)
        fillUpRibos = np.concatenate((fillUpRibos, 
                                np.array([[riboInfo[i,0], riboInfo[i,1], 
                                           pos2[0], pos2[1],pos2[2],
                                           ang2[0], ang2[1], ang2[2]]])), axis = 0)
        
    return fillUpRibos, fillUpMiddleRibos #returned 2-2D arrays
                 

def checkRibo(particleStar, riboCoord, pruneRad, factor = 10):
    if isinstance(particleStar, str):
        particleStar = tom_starread(particleStar)
    else:
        particleStar = particleStar
    #I think linear numpy is faster than for loop   
    posAll = particleStar.loc[:,['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values 
    difflen = np.linalg.norm(posAll - riboCoord, axis = 1)
    #assert len(difflen) == posAll.shape[0]
    if np.sum(difflen <= pruneRad/factor) > 0:
        return 1
    else:
        return 0
 
 