import numpy as np
from py_cluster.tom_A2Odist import tom_A2Odist


def tom_find_poorBranch(pairList, shift, rot, worker_n = 1, gpu_list = None, cmb_metric = 'scale2Ang',
                        pruneRad = 100):
    '''
    TOM_CLEAN_POORBRANCH return the best transList when branches come out
    
    INPUT
        pairList     transList with transform information stored
        shift        the average transShift (1D array, X,Y,Z)
        rot          the average transRot (1D array)
        clean_type   which kind of branches to clean. 'idx1'=>like [[1,2],[1,3]].
                     'idx2' => like [[2,1],[3,1]] 
        worker_n     number of cpus
        gpu_list     number of gpus
        cmb_metric   metric to combine vect/angle distance 'scale2Angle'/'scale2AngFudge' 
        pruneRad     normalized factor(default,100)    
    
    OUTPUT
        cleanPairList  the pairList afte branches clean
    ''' 
    
    pairList_clean = pairList.loc[:,['pairIDX1', 'pairIDX2', 'pairTransVectX', 'pairTransVectY', 'pairTransVectZ', 
                               'pairTransAngleZXZPhi', 'pairTransAngleZXZPsi', 'pairTransAngleZXZTheta']]
    idx1_drop = tom_find_poorBranchOneDirection(pairList_clean, shift, rot, 'idx1', worker_n, gpu_list,
                                                   cmb_metric, pruneRad)
    idx2_drop = tom_find_poorBranchOneDirection(pairList_clean, shift, rot, 'idx2', worker_n, gpu_list,
                                                   cmb_metric, pruneRad)
    idx12_drop = np.union1d(idx1_drop, idx2_drop)
   
   
    return idx12_drop
    
       
def tom_find_poorBranchOneDirection(pairList, shift, rot, clean_type = 'idx1', worker_n = 1, 
                           gpu_list = None, cmb_metric = 'scale2Ang', pruneRad = 100):

    if clean_type ==  'idx1':
        idx_drop = oneDirectionBranchClean(pairList, 'pairIDX1', 
                                                shift, rot, worker_n, 
                                                gpu_list, cmb_metric, 
                                                pruneRad)       
    elif clean_type == 'idx2':
        idx_drop = oneDirectionBranchClean(pairList, 'pairIDX2',
                                                shift, rot, worker_n, 
                                                gpu_list, cmb_metric, 
                                                pruneRad)      
    else:
        raise TypeError('Unrecoginazed paramter %s'%clean_type)
        
    return idx_drop
        
        
        
        
def oneDirectionBranchClean(pairList, clean_direction, 
                            shift, rot, worker_n, gpu_list, 
                            cmb_metric, pruneRad):
    
    idx_counts = pairList[clean_direction].value_counts()[pairList[clean_direction].value_counts() > 1]
    if len(idx_counts) == 0:
        return np.array([],dtype = np.int)
    idx_counts = dict(idx_counts)
    #only keep pairList with branch
    idx = np.array(list(idx_counts.keys()))
    index = np.where(pairList[clean_direction].values == idx[:,None])[-1]
    pairList = pairList.iloc[index,:]
    _, _, combDist = tom_A2Odist(pairList.loc[:,['pairTransVectX', 'pairTransVectY', 'pairTransVectZ']].values,
                                       pairList.loc[:,['pairTransAngleZXZPhi', 'pairTransAngleZXZPsi', 
                                                     'pairTransAngleZXZTheta']].values,
                                       shift, rot, worker_n, gpu_list, cmb_metric, pruneRad)
    #sort the combDist 
    idxSort = np.argsort(combDist)
    pairListSort = pairList.iloc[idxSort, :]
    idx_dup = pairListSort.index.__array__()
    idxKeep = np.unique(pairListSort[clean_direction].values, return_index = True)[1]
    idx_drop = np.delete(idx_dup,idxKeep)
    assert (len(idx_drop) + len(idxKeep)) == len(idx_dup)  
   
    return idx_drop
    