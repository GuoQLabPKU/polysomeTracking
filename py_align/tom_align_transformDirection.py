import numpy as np
from scipy.cluster.vq import kmeans, vq

def alignDir(pairList): #the input is subset of one dataframe pointer
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
        #no need to call whiten functions
        centroids,_ = kmeans(vects,2) #the cis and trans order
        cl, _ = vq(vects, centroids)            
    else:
        cl = 0        
    if np.sum(cl==0) > np.sum(cl==1):
        useCl = 0
    else:
        useCl = 1
    idx = np.where(cl == useCl)[0]
    meanV = np.mean(vects[idx,:], axis = 0)
    diffV = vects - np.tile(meanV, (vects.shape[0],1))
    diffV = np.linalg.norm(diffV, axis = 1)
    diffVInv = vectsInv - np.tile(meanV, (vects.shape[0],1))
    diffVInv = np.linalg.norm(diffVInv, axis = 1)
    
    idxSwap = np.where(diffV < diffVInv)[0]
#    for single_idx in idxSwap:
#        swapPairOrderEntry(pairList.iloc[single_idx,:])  #the input is one pointer  
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
    
    return pairList.values

    
    
def tom_align_transformDirection(transList):
    allClasses = transList['pairClass'].values
    allClassesU = np.unique(allClasses)
    for single_class in allClassesU:
        if single_class == 0: #if the class == 0: continue
            continue 
        if single_class == -1:
            print('Warning: no clusters classes detected. Skipping align the transform.')
            break
        idx = np.where(allClasses == single_class)[0]
        transList.iloc[idx,:] = alignDir(transList.iloc[idx,:]) #class 0 will also be aligned and class -1
    return transList