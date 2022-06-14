import numpy as np
from scipy.cluster.vq import kmeans, vq
from py_log.tom_logger import Log

def alignDir(pairList, iterN): #the input is subset of one dataframe pointer
    while iterN > 0:
        vects = np.array([pairList['pairTransVectX'].values,
                          pairList['pairTransVectY'].values,
                          pairList['pairTransVectZ'].values])
        vects = vects.transpose()
        
        vectsInv = np.array([pairList['pairInvTransVectX'].values,
                             pairList['pairInvTransVectY'].values,
                             pairList['pairInvTransVectZ'].values])
        vectsInv = vectsInv.transpose()
    
        if vects.shape[0] > 1:
            np.random.seed(1)
            centroids,_ = kmeans(vects,2) #the cis and trans order           
            cl, _ = vq(vects, centroids)            
        else:
            cl = 0        
    #    if np.sum(cl==0) > np.sum(cl==1):
    #        useCl = 0
    #    else:
    #        useCl = 1
    #    idx = np.where(cl == useCl)[0]
    #    meanV = np.mean(vects[idx,:], axis = 0)
        meanV = getCluster(vects, cl)
        diffV = vects - np.tile(meanV, (vects.shape[0],1))
        diffV = np.linalg.norm(diffV, axis = 1)
        diffVInv = vectsInv - np.tile(meanV, (vects.shape[0],1))
        diffVInv = np.linalg.norm(diffVInv, axis = 1)
    
        idxSwap = np.where(diffV > diffVInv)[0]

        rowNames = pairList._stat_axis.values.tolist()
        rowNamesSel = [rowNames[i] for i in idxSwap]
        #print(rowNamesSel[0])
        swap_name1 = ['pairIDX1', 'pairTransVectX','pairTransVectY', 
                      'pairTransVectZ', 'pairTransAngleZXZPhi',
                      'pairTransAngleZXZPsi', 'pairTransAngleZXZTheta',
                      'pairCoordinateX1', 'pairCoordinateY1',
                       'pairCoordinateZ1','pairAnglePhi1',
                       'pairAnglePsi1','pairAngleTheta1',
                      'pairClass1',  'pairPsf1']
        swap_name2 = ['pairIDX2', 'pairInvTransVectX','pairInvTransVectY',
                      'pairInvTransVectZ', 'pairInvTransAngleZXZPhi',
                      'pairInvTransAngleZXZPsi', 'pairInvTransAngleZXZTheta',
                      'pairCoordinateX2', 'pairCoordinateY2',
                       'pairCoordinateZ2','pairAnglePhi2',
                       'pairAnglePsi2','pairAngleTheta2',
                      'pairClass2',  'pairPsf2']
    
        backup = pairList.loc[rowNamesSel, swap_name1].values
        pairList.loc[rowNamesSel, swap_name1] =  pairList.loc[rowNamesSel, swap_name2].values
        pairList.loc[rowNamesSel, swap_name2] =  backup   
        
        iterN -= 1   
    return pairList.values

def getCluster(vects, cl):
    idx1 = np.where(cl == 1)[0]
    idx0 = np.where(cl == 0)[0]
    
    meanV1 = np.mean(vects[idx1,:], axis = 0)
    diffV1 = vects[idx1,:] - np.tile(meanV1, (len(idx1),1))
    diffV1 = np.linalg.norm(diffV1, axis = 1)
 
    meanV0 = np.mean(vects[idx0,:], axis = 0)
    diffV0 = vects[idx0,:] - np.tile(meanV0, (len(idx0),1))
    diffV0 = np.linalg.norm(diffV0, axis = 1)
    
    assert len(diffV1) == len(idx1)
    assert len(diffV0) == len(idx0)
    if np.mean(diffV0) > np.mean(diffV1):
        return meanV1
    else:
        return meanV0
    
    
def tom_align_transformDirection(transList, iterN = 1):
    log = Log('align transforms').getlog()   
    allClasses = transList['pairClass'].values
    allClassesU = np.unique(allClasses)
    for single_class in allClassesU:
        if single_class == 0: #if the class == 0: continue
            continue 
        if single_class == -1:
            log.warning('No cluster classes detected. Skip align the transforms.')
            break
        log.info('Align class%d for %d iterations'%(single_class, iterN))
        idx = np.where(allClasses == single_class)[0]
        transList.iloc[idx,:] = alignDir(transList.iloc[idx,:], iterN) 
    return transList
