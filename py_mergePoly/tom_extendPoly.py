import numpy as np

from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate
from py_io.tom_starread import tom_starread

def tom_extendPoly(tailRiboInfo, avgRot, avgShift, particleStar , pruneRad, 
                   NumAddRibo = 1,xyzborder = None):
    '''
    TOM_EXTENDPOLY add ribosomes to the end of each polysome
    
    EXAMPLE
    addRiboInfo = tom_extendPoly();
    
    PARAMETERS
    
    INPUT
    tailRiboInfo    (nx7,np.array) the tail ribosomes of the whole polysomes
                     np.array([polyId, coordinateX, coordinateY,coordinateZ,phi,psi,theta ])
    avgRot          average rotation for extend long
    avgShift        average shift extend long
    particlStar    starfile wit the whole particles
    pruneRad         the diameter of one ribosome(pixel)
    NumAddRibo       (int) number of ribosomes you want put in the tail of the polysome
    xyzborder       (1x3, np.array) the xmax,ymax,zmax of the tomgram 
    
    '''
    polyN = tailRiboInfo.shape[0]
    newExtendRibos  = np.array([]).reshape(-1, 7)
    gapRibos = np.array([]).reshape(-1, 7)
    for i in range(polyN):
        ang1 = tailRiboInfo[i,4:]
        pos1 = tailRiboInfo[i,1:4]
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
        
        gapRibo = np.array([]).reshape(-1, 7)
        cycles = NumAddRibo - 1
        while cycles > 0:
            
            ang1 = ang2
            pos1 = pos2
            gapRibo = np.concatenate((gapRibo, 
                                      np.array([[tailRiboInfo[i,0], pos1[0], pos1[1], pos1[2],
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
        #put new ribosomes into data        
        newExtendRibos = np.concatenate((newExtendRibos, 
                                np.array([[tailRiboInfo[i,0], pos2[0], pos2[1],pos2[2],
                                 ang2[0],ang2[1], ang2[2]]])
                                  ), axis = 0)
        gapRibos = np.concatenate((gapRibos, gapRibo), axis = 0)
    return newExtendRibos, gapRibos #returned 2-2D arrays
                 

def checkRibo(particleStar, riboCor, pruneRad):
    if isinstance(particleStar, str):
        particleStar = tom_starread(particleStar)
    else:
        particleStar = particleStar
    #I think linear numpy is faster than for loop   
    posAll = particleStar.loc[:,['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values 
    difflen = np.linalg.norm(posAll - riboCor)
    if np.sum(difflen <= pruneRad) > 0:
        return 1
    else:
        return 0
 
 