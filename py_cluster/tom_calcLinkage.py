import numpy as np
import gc
from scipy.cluster.hierarchy import linkage
from py_memory.tom_memalloc import tom_memalloc


def tom_calcLinkage(transList, preCalcFold, maxDistInpix, cmb_metric='mean0+1std', worker_n = None, gpu_list = None,freeMem = None ):
    '''
    TOM_CALCLINKAGE: using hierachical clustering method to build linkage of different transforms 
        ll = tom_calcLinkage(transList, preCalcFold, maxDistInpix, cmb_metric)
  
    PARAMETERS
  
    INPUT
    
        transList                 dataframe returned from tom_calcTransforms
        preCalcFold               folder to store the linkage results, will be used in future clustering
        maxDistInpix              used as to scale the distance of shift path vectors
        cmb_metric                ('mean0+1std')different metrics to combine the distance of shift and rotation
                                  ('scale2Ang'/'scale2AngFudge' are also offered)
        worker_n                  (None)number of CPUs for parallel computation
        gpu_list                  (None)gpus list for gpu parallel computation
        freeM                     (None)free memory for computation

    OUTPUT
  
        ll                        linakge results (m*4 array float64) (should be a bit 
                                  different with the matlab version)
        
    EXAMPLE
     

    REFERENCES
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
    
    
    maxChunk = tom_memalloc(freeMem, worker_n, gpu_list)#maxChunk can be uint64(cpu) or dict(gpus)
    #using gpu or cpu
    if isinstance(worker_n, int):
        from py_cluster.tom_pdist_cpu import tom_pdist
    else:
        from py_cluster.tom_pdist_gpu import tom_pdist
       
    distsVect = tom_pdist(transVect,  maxChunk ,worker_n, gpu_list,'euc', transVectInv)
    del transVect, transVectInv
    gc.collect()
    
    distsAng =  tom_pdist(transAngVect,  maxChunk ,worker_n, gpu_list,'ang', transAngVectInv)
    del transAngVect, transAngVectInv, maxChunk
    gc.collect()       
    print("Using %s to combine angles and shifts"%cmb_metric)
    
    if cmb_metric == 'scale2Ang':
        distsVect = distsVect/(2*maxDistInpix)*180
        distsCN = (distsAng+distsVect)/2
    elif cmb_metric == 'scale2AngFudge':
        distsVect = distsVect/(2*maxDistInpix)*180
        distsCN = (distsAng+(distsVect*2))/2
    elif cmb_metric == 'mean+1std':
        if np.std(distsVect) > 0:
            distsVect_norm = (distsVect - np.mean(distsVect)) / np.std(distsVect) #mean+- 1std.
            distsVect_norm = (distsVect_norm - np.min(distsVect_norm)) / np.max(distsVect_norm - np.min(distsVect_norm)) #from 0-1
            distsVect_norm = distsVect_norm + 1.0 #from 1-2
        else:
            distsVect_norm = (distsVect - np.mean(distsVect)) + 1.0 #from 0-1(1,1,1)
        
        if np.std(distsAng) > 0:
            distsAng_norm = (distsAng - np.mean(distsAng)) / np.std(distsAng) #mean+- 1std.
            distsAng_norm = (distsAng_norm - np.min(distsAng_norm)) / np.max(distsAng_norm - np.min(distsAng_norm)) #from 0-1
            distsAng_norm = distsAng_norm + 1.0 #from 1-2
        else:
            distsAng_norm = (distsAng - np.mean(distsAng)) + 1.0 #from 0-1(1,1,1)   
        distsCN = distsVect_norm*distsAng_norm
        if np.std(distsCN) > 0:
            distsCN = (distsCN - np.mean(distsCN)) / np.std(distsCN) #mean+- 1std.
            distsCN = (distsCN - np.min(distsCN)) / np.max(distsCN - np.min(distsCN)) #from 0-1 
        else:
            distsCN = (distsCN - np.mean(distsCN)) + 1.0 #(1,1,1)
    print('Calculating linkage')
    ll = linkage(distsCN,'average') #looks like a slow step if too large of distsCN
    print("Calculating linkage done")
    #save the ll results
    np.save("%s/tree.npy"%preCalcFold,ll)
    
    return ll

#if __name__ == '__main__':
#    ll = tom_calcLinkage(startSt, 'py_io', 100, 'mean+1std')
    
    
    
    
