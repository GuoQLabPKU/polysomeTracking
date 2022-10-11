from nemotoc.py_summary.tom_analysePolysomePopulation import calcVectStat, calcAngStat
from nemotoc.py_cluster.tom_A2Odist import tom_A2Odist
from nemotoc.py_align.tom_align_transformDirection import tom_align_transformDirection

import numpy as np

def tom_assignTransFromCluster(transList, clusterStat, cmb_metric, pruneRad, iterN = 5, threshold = 90,
                               worker_n = 1, gpu_list = None, freeMem = None):
    '''
    assign each transform into one cluster 
    transList: pairList retunred by NEMO-TOC 
    clusterStat: the detailed information of each cluster, dict [clusterNr:[maxDist,X,Y,Z,phi,psi,theta]]
    '''
    
    transAngVect = np.array([transList["pairTransAngleZXZPhi"].values, 
                             transList["pairTransAngleZXZPsi"].values, 
                             transList["pairTransAngleZXZTheta"].values]).transpose()
    transAngVectInv = np.array([transList["pairInvTransAngleZXZPhi"].values, 
                             transList["pairInvTransAngleZXZPsi"].values, 
                             transList["pairInvTransAngleZXZTheta"].values]).transpose()
    
    transVect = np.array([transList["pairTransVectX"].values, 
                             transList["pairTransVectY"].values, 
                             transList["pairTransVectZ"].values]).transpose()  
    transVectInv = np.array([transList["pairInvTransVectX"].values, 
                             transList["pairInvTransVectY"].values, 
                            transList["pairInvTransVectZ"].values]).transpose()
    #merge two arrays
    vectList = np.concatenate((transVect,transVectInv))
    angList = np.concatenate((transAngVect,transAngVectInv))
    assert transVect.shape[0] == transAngVectInv.shape[0]
    
    for _ in range(iterN):
        #record the information of each trans
        forwardBackwardDist = np.zeros((transList.shape[0],2))
        pairClassList = np.zeros(transList.shape[0], dtype = np.int)
        transDistMin = np.ones(transList.shape[0])*100000

        for j in clusterStat.keys():
            maxDist = clusterStat[j][0]
            clusterId = j 
            transVectCluster = clusterStat[j][1:4]
            angVectCluster = clusterStat[j][4:7]
            
            _, _, distCmb = tom_A2Odist(vectList, angList, transVectCluster, angVectCluster, worker_n, gpu_list, cmb_metric, pruneRad,0)
            assert len(distCmb) == transList.shape[0]*2
            forwardBackwardDist[:,0] = distCmb[0:transList.shape[0]]
            forwardBackwardDist[:,1] = distCmb[transList.shape[0]:transList.shape[0]*2]
            forwardBackwardDistSort = np.sort(forwardBackwardDist, axis = 1)
            transIdList = np.where(forwardBackwardDistSort[:,0] < maxDist)[0]
            
            for sgId in transIdList:
                if transDistMin[sgId] > forwardBackwardDistSort[sgId,0]:
                    transDistMin[sgId] = forwardBackwardDistSort[sgId,0]
                    pairClassList[sgId] = clusterId
            
        #update the centeroid of each cluster   
        #firstly, update the cluster number and align the direction
        transList['pairClass'] = pairClassList  
        transList = tom_align_transformDirection(transList, 1, 0)
        
        clusterStat = { }
        allClustersU = np.unique(pairClassList)
        for single_cluster in allClustersU:
            if single_cluster == 0:#no need analysis cluster0
                continue
            idx = np.where(pairClassList == single_cluster)[0]      
            vectStat, distsVect = calcVectStat(transList.iloc[idx,:])
            angStat, distsAng = calcAngStat(transList.iloc[idx,:])           
            if cmb_metric == 'scale2Ang':
                distsVect2 = distsVect/(2*pruneRad)*180
                distsCN = (distsAng+distsVect2)/2
            elif cmb_metric == 'scale2AngFudge':
                distsVect2 = distsVect/(2*pruneRad)*180
                distsCN = (distsAng+(distsVect2*2))/2
                
            clusterStat[single_cluster] = np.zeros(7)
            clusterStat[single_cluster][0] = np.percentile(distsCN, threshold)     
            clusterStat[single_cluster][1:4] = [vectStat['meanTransVectX'], vectStat['meanTransVectY'], vectStat['meanTransVectZ']]
            clusterStat[single_cluster][4:7] = [angStat['meanTransAngPhi'], angStat['meanTransAngPsi'], angStat['meanTransAngTheta']]
            
    return pairClassList, clusterStat
            
            
        
    
        
    

