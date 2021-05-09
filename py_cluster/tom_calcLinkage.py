import numpy as np
from scipy.cluster.hierarchy import linkage
from py_cluster.tom_pdist import tom_pdist

def tom_calcLinkage(transList, preCalcFold, maxDistInpix, cmb_metric, maxChunk = 600000000):
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
    
    distsVect = tom_pdist(transVect, 'euc', transVectInv, maxChunk = maxChunk)
    distsAng =tom_pdist(transAngVect, 'ang', transAngVectInv,maxChunk = maxChunk)
    
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
            distsVect_norm = (distsVect_norm - np.min(distsVect_norm)) / max(distsVect_norm - min(distsVect_norm)) #from 0-1
            distsVect_norm = distsVect_norm + 1.0 #from 1-2
        else:
            distsVect_norm = (distsVect - np.mean(distsVect)) + 1.0 #from 0-1(1,1,1)
        
        if np.std(distsAng) > 0:
            distsAng_norm = (distsAng - np.mean(distsAng)) / np.std(distsAng) #mean+- 1std.
            distsAng_norm = (distsAng_norm - np.min(distsAng_norm)) / max(distsAng_norm - min(distsAng_norm)) #from 0-1
            distsAng_norm = distsAng_norm + 1.0 #from 1-2
        else:
            distsAng_norm = (distsAng - np.mean(distsAng)) + 1.0 #from 0-1(1,1,1)   
        distsCN = distsVect_norm*distsAng_norm
        if np.std(distsCN) > 0:
            distsCN = (distsCN - np.mean(distsCN)) / np.std(distsCN) #mean+- 1std.
            distsCN = (distsCN - np.min(distsCN)) / max(distsCN - min(distsCN)) #from 0-1 
        else:
            distsCN = (distsCN - np.mean(distsCN)) + 1.0 #(1,1,1)
    print('Calculating linkage')
    ll = linkage(distsCN,'average')
    print("Calculating linkage done")
    #save the ll results
    np.save("%s/tree.npy"%preCalcFold,ll)
    
    return ll

#if __name__ == '__main__':
#    ll = tom_calcLinkage(startSt, 'py_io', 100, 'mean+1std')
    
    
    
    