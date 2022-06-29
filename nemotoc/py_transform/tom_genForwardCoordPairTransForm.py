import pandas as pd
import numpy as np

from py_transform.tom_sum_rotation import tom_sum_rotation
from py_transform.tom_pointrotate import tom_pointrotate
from py_io.tom_starread import tom_starread 

def tom_genForwardCoordPairTransForm(centerCoord, transList, stateTransList = None):
    '''
    TOM_GENFORWARDCOORDPAIRTRANSFORM generates the euler angles 
    and coordinates of particles relative to another particle's coordinate 
    given the rotation and translocation
    '''
    if isinstance(transList, str):
        transList = tom_starread(transList)
        transList = transList['data_particles']
    if stateTransList is not None:       
        if isinstance(stateTransList, str):
            stateTransList = tom_starread(stateTransList)
            stateTransList = stateTransList['data_particles']
    #make a dict for color map
    classColor = { }
    #make a dataframe to store the data
    eachPointCoordList = np.zeros((transList.shape[0], 7))
    eachPointColorList = []
    if stateTransList is not None:
        avgPointCoordList = np.zeros((stateTransList.shape[0], 7))
        avgPointAngList = np.zeros((stateTransList.shape[0], 6))
        avgPointColorList = []
    else:
        avgPointCoordList = None
        avgPointAngList = None
        avgPointColorList = None
    
    pointNr = transList.shape[0]
    for singleP in range(pointNr):
        transVect, transRot = transList.loc[singleP, ['pairTransVectX', 'pairTransVectY',
                                                      'pairTransVectZ']].values.astype(np.float), \
                              transList.loc[singleP, ['pairTransAngleZXZPhi', 'pairTransAngleZXZPsi',
                                                      'pairTransAngleZXZTheta']].values.astype(np.float)
        singleCol = transList.loc[singleP, 'pairClassColour']
        
        #calculate the forward and invert coordinate by the rotation and translocation
        centerFr, centerIv,_,_ = genFowardInvertCoord(centerCoord, transVect, transRot)
        eachPointCoordList[singleP][0:3] = centerFr
        eachPointCoordList[singleP][3:6] = centerIv
        eachPointCoordList[singleP][6] = transList.loc[singleP, 'pairClass']
        eachPointColorList.append(singleCol)
        classColor[transList.loc[singleP, 'pairClass']] = singleCol
    if stateTransList is not None:
        classNr = stateTransList.shape[0]
        for singleCl in range(classNr):
            transVect, transRot = stateTransList.loc[singleCl, ['meanTransVectX', 'meanTransVectY',
                                                          'meanTransVectZ']].values.astype(np.float), \
                                  stateTransList.loc[singleCl, ['meanTransAngPhi', 'meanTransAngPsi',
                                                       'meanTransAngTheta']].values.astype(np.float)
            class_ = stateTransList.loc[singleCl, 'classNr']
            #calculate the forward and invert coordinate by the rotation and translocation
            centerFr, centerIv, centerFrRot, centerIvRot = genFowardInvertCoord(centerCoord, transVect, transRot)
            avgPointCoordList[singleCl][0:3] = centerFr
            avgPointCoordList[singleCl][3:6] = centerIv
            avgPointCoordList[singleCl][6] = class_
            avgPointAngList[singleCl][0:3] = centerFrRot
            avgPointAngList[singleCl][3:6] = centerIvRot
            avgPointColorList.append(classColor[class_])
    
    #make a dataframe and save it!
    eachPtr = pd.DataFrame(eachPointCoordList)
    eachPtr.columns = ['frx','fry','frz','inx','iny','inz', 'class']
    eachPtr['color'] = eachPointColorList
    if avgPointColorList is not None:
        avgPtr = pd.DataFrame(avgPointCoordList)
        avgPtr.columns = ['frx_avg','fry_avg','frz_avg','inx_avg','iny_avg','inz_avg','class']
        avgPtr['color'] = avgPointColorList        
        neigh3RiboPairTransList = makeTransList(avgPointCoordList, avgPointAngList, centerCoord, classColor)
        
    return eachPtr, avgPtr, neigh3RiboPairTransList

def makeTransList(avgPointCoordList, avgPointAngList, centerCoord, classColor):
    pairTransList = np.zeros((avgPointCoordList.shape[0]*2, 16))
    colorList = [ ]
    assert avgPointCoordList.shape[0] == avgPointAngList.shape[0]
    for single_row in range(avgPointCoordList.shape[0]):
        transRow1 = single_row*2
        transRow2 = single_row*2+1
        classNr = avgPointCoordList[single_row][6]
        pairTransList[transRow1][2:5] = avgPointCoordList[single_row][3:6]
        pairTransList[transRow1][5:8] = avgPointAngList[single_row][3:6]
        pairTransList[transRow1][8:11] = centerCoord
        pairTransList[transRow1][14] = classNr
        pairTransList[transRow1][15] = classNr
        
        pairTransList[transRow2][2:5] = centerCoord
        pairTransList[transRow2][8:11] = avgPointCoordList[single_row][0:3]
        pairTransList[transRow2][11:14] = avgPointAngList[single_row][0:3]
        pairTransList[transRow2][14] = classNr
        pairTransList[transRow2][15] = classNr   
        
        #addcolor
        colorList.append(classColor[classNr])  
        colorList.append(classColor[classNr])             
        
    #make dtaframe 
    neigh3RiboInfoList = pd.DataFrame(pairTransList)
    neigh3RiboInfoList.columns = ['pairIDX1','pairIDX2','pairCoordinateX1', 'pairCoordinateY1', 'pairCoordinateZ1',
                             'pairAnglePhi1', 'pairAnglePsi1', 'pairAngleTheta1',
                             'pairCoordinateX2', 'pairCoordinateY2', 'pairCoordinateZ2',
                             'pairAnglePhi2', 'pairAnglePsi2', 'pairAngleTheta2','pairClass','pairLabel']
    neigh3RiboInfoList['color'] = colorList
    return neigh3RiboInfoList
        
      
def genFowardInvertCoord(centerCoord, vect, rot, numOfRepeat=1, angTmp = None):
    if angTmp is None:
        angTmp = [0,0,0]
    trTmp = [0,0,0]
    for _ in range(numOfRepeat):
        angFr, trFr, _ = tom_sum_rotation(np.array([[angTmp[0], angTmp[1], angTmp[2]],
                                       [rot[0], rot[1], rot[2]]]),
                              np.array([trTmp,
                                        [vect[0], vect[1], vect[2]]]),'rot_trans')
        angTmp = list(angFr)
        trTmp  = list(trFr)
        
    #the invert
    angTmp = [0,0,0]
    trTmp = [0,0,0]    
    trI = tom_pointrotate(vect, -rot[1], -rot[0], -rot[2])*-1
    for _ in range(numOfRepeat):
        angIv, trIv, _ = tom_sum_rotation(np.array([angTmp,
                                           [-rot[1],-rot[0],-rot[2]]]),
                                  np.array([trTmp,
                                            [trI[0], trI[1], trI[2]]]), 'rot_trans')   
        angTmp = list(angIv)
        trTmp = list(trIv)
    return centerCoord+trFr, centerCoord + trIv, angFr, angIv
   