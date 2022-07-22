from nemotoc.py_summary.tom_analysePolysomePopulation import calcVectStat, calcAngStat
from nemotoc.py_cluster.tom_A2Odist import tom_A2Odist
from nemotoc.py_align.tom_align_transformDirection import tom_align_transformDirection

import numpy as np

def tom_assignTransFromCluster(transList, clusterStat, cmb_metric, pruneRad, iterN = 5, worker_n = 1, gpu_list = None, freeMem = None):
    '''
    assign each transform into one cluster 
    transList: pairList retunred by NEMO-TOC 
    clusterStat: the detailed information of each cluster, dict [clusterNr:[maxDist,X,Y,Z,phi,psi,theta]]
    '''
    for _ in range(iterN):
        #transfer the clusterStat into array to calculate these distances 
        transVect = np.zeros((len(clusterStat.keys()), 3))
        angVect = np.zeros((len(clusterStat.keys()), 3))
        clusterId = { }
        for i,j in enumerate(clusterStat.keys()):
            clusterId[i] = j 
            transVect[i, :] = clusterStat[j][1:4]
            angVect[i, :] = clusterStat[j][4:7]
        #assign each transform into each cluster        
        pairClassList = np.zeros(transList.shape[0], dtype = np.int)
        for i in range(transList.shape[0]):
            trans = np.array([transList['pairTransVectX'].values[i], transList['pairTransVectY'].values[i], transList['pairTransVectZ'].values[i]])
            ang = np.array([transList['pairTransAngleZXZPhi'].values[i], transList['pairTransAngleZXZPsi'].values[i], transList['pairTransAngleZXZTheta'].values[i]])
            transInv = np.array([transList['pairInvTransVectX'].values[i], transList['pairInvTransVectY'].values[i], transList['pairInvTransVectZ'].values[i]])
            angInv = np.array([transList['pairInvTransAngleZXZPhi'].values[i], transList['pairInvTransAngleZXZPsi'].values[i], transList['pairInvTransAngleZXZTheta'].values[i]])
            _,_, distCmb = tom_A2Odist(transVect, angVect, trans, ang, worker_n, gpu_list, cmb_metric, pruneRad)
            _,_, distCmbInv = tom_A2Odist(transVect, angVect, transInv, angInv, worker_n, gpu_list, cmb_metric, pruneRad)
            minDist, argMin = np.min(distCmb), np.argmin(distCmb)
            minDistInv, argMinInv = np.min(distCmbInv), np.argmin(distCmbInv)
            if minDist <= minDistInv:
                clusterSg = clusterId[argMin]
                if minDist > clusterStat[clusterSg][0]:
                    clusterSg = 0
            else:
                clusterSg = clusterId[argMinInv]
                if minDistInv >  clusterStat[clusterSg][0]:
                     clusterSg = 0
                    
            pairClassList[i] = clusterSg
            
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
            clusterStat[single_cluster][0] = np.max(distsCN)
            clusterStat[single_cluster][1:4] = [vectStat['meanTransVectX'], vectStat['meanTransVectY'], vectStat['meanTransVectZ']]
            clusterStat[single_cluster][4:7] = [angStat['meanTransAngPhi'], angStat['meanTransAngPsi'], angStat['meanTransAngTheta']]
            
    return pairClassList, clusterStat
            
            
        
    
        
    

