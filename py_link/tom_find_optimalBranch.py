import numpy as np

from py_cluster.tom_pdist_All2One import tom_pdist_All2One

def tom_find_optimalBranch(pairList, shift, rot, clean_type = 'head', worker_n = 1, 
                           gpu_list = None, cmb_metric = 'scale2Ang', pruneRad = 100):

    '''
    TOM_FIND_OPTIMALBRANCH return the best transList when branches come out
    
    INPUT
        pairList     transList withe transform information stored
        shift        the average transShift (1D array, X,Y,Z)
        rot          the average transRot (1D array)
        clean_type   which kind of branches to clean. 'head'=>like [[1,2],[1,3]].
                     'tail' => like [[2,1],[3,1]] 
        worker_n     number of cpus
        gpu_list     number of gpus
        cmb_metric   metric to combine vect/angle distance 'scale2Angle'/'scale2AngFudge' 
        pruneRad     normalized factor(default,100)    
    
    OUTPUT
        cleanPairList  the pairList afte branches clean
    '''
    pairList = pairList.loc[:,['pairIDX1', 'pairIDX2', 'pairTransVectX', 'pairTransVectY', 'pairTransVectZ', 
                               'pairTransAngleZXZPhi', 'pairTransAngleZXZPsi', 'pairTransAngleZXZTheta']]
    #reset index to math tom_linkTransform
    pairList.reset_index(drop = True,inplace = True)
    if clean_type ==  'idx1':
        cleanIdx = oneDirectionBranchClean(pairList, 'pairIDX1', 
                                                shift, rot, worker_n, 
                                                gpu_list, cmb_metric, 
                                                pruneRad)       
    elif clean_type == 'idx2':
        cleanIdx = oneDirectionBranchClean(pairList, 'pairIDX2',
                                                shift, rot, worker_n, 
                                                gpu_list, cmb_metric, 
                                                pruneRad)      
    else:
        raise TypeError('Unrecoginazed paramter %s'%clean_type)
        
    return cleanIdx
        
        
        
        
def oneDirectionBranchClean(pairList, clean_direction, 
                            shift, rot, worker_n, gpu_list, 
                            cmb_metric, pruneRad):
        idx_counts = pairList[clean_direction].value_counts()[pairList[clean_direction].value_counts() > 1]
        if len(idx_counts) == 0:
            return None
        idx_counts = dict(idx_counts)
        #only keep pairList with branch
        idx = np.array(list(idx_counts.keys()))
        index = np.where(pairList[clean_direction].values == idx[:,None])[-1]
        pairList = pairList.iloc[index,:]
        _, _, combDist = tom_pdist_All2One(pairList.loc[:,['pairTransVectX', 'pairTransVectY', 'pairTransVectZ']].values,
                                           pairList.loc[:,['pairTransAngleZXZPhi', 'pairTransAngleZXZPsi', 
                                                         'pairTransAngleZXZTheta']].values,
                                           shift, rot, worker_n, gpu_list, cmb_metric, pruneRad)
        #sort the combDist 
        idxSort = np.argsort(combDist)
        pairListSort = pairList.iloc[idxSort, :]
        idxKeep = np.unique(pairListSort[clean_direction].values, return_index = True)[1]
        pairListSortKeep = pairListSort.iloc[idxKeep, :]
        assert pairListSortKeep.shape[0] == len(idx)  
        
        return pairListSortKeep.index.__array__()
    