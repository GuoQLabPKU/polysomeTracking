import sys
sys.path.append('/lustre/Data/jiangwh/polysome/python_version/polysome/')
import numpy as np


from py_io.tom_starread import tom_starread
from py_cluster.tom_A2Odist import tom_A2Odist
from py_stats.tom_kdeEstimate import tom_kdeEstimate

trans = tom_starread('cluster-particles/run0/allTransforms.star')
transb4Relink = tom_starread('cluster-particles/run0/allTransformsb4Relink.star')
statSummary = tom_starread('cluster-particles/run0/stat/statPerClass.star')

trans = trans['data_particles']
transb4Relink = transb4Relink['data_particles']
statSummary = statSummary['data_particles']
#find the drop trans 
keep_idx = [ ]
for single_row in range(transb4Relink.shape[0]):
    idx1, idx2 = transb4Relink['pairIDX1'].values[single_row], \
                 transb4Relink['pairIDX2'].values[single_row]
    pair1 = trans[(trans['pairIDX1'] == idx1) & (trans['pairIDX2'] == idx2)].shape[0]
    pair2 = trans[(trans['pairIDX2'] == idx1) & (trans['pairIDX1'] == idx2)].shape[0]
    if (pair1+pair2) > 0:
        continue
    keep_idx.append(single_row)

transDelete = transb4Relink.iloc[keep_idx,:]
############################################################################
#cycle each class in trans and say how possible one trans from other classes 
#will be wrongly classify to another class
for single_row in range(statSummary.shape[0]):
    classN = statSummary['classNr'].values[single_row]
    avgShift = statSummary.loc[single_row,['meanTransVectX',
                                           'meanTransVectY',
                                           'meanTransVectZ']].values #1D array
    avgRot = statSummary.loc[single_row,['meanTransAngPhi',
                                         'meanTransAngPsi',
                                         'meanTransAngTheta']].values
    #for distance calculation from the same class
    transVectSame = trans[trans['pairClass'] == classN].loc[:,['pairTransVectX',
                         'pairTransVectY','pairTransVectZ']].values #2D array
    transRotSame = trans[trans['pairClass'] == classN].loc[:,['pairTransAngleZXZPhi',
                         'pairTransAngleZXZPsi','pairTransAngleZXZTheta']].values
    
    distVectSame, distAngSame, distCombineSame = tom_A2Odist(transVectSame, transRotSame, 
                                                                   avgShift, avgRot, 
                                                                   worker_n = 1, gpu_list = None, 
                                                                   cmb_metric = 'scale2Ang', pruneRad = 100)
    #for distance calculation from different class
    transVectDiff1 = trans[trans['pairClass'] != classN].loc[:,['pairTransVectX',
                         'pairTransVectY','pairTransVectZ']].values    
    transRotDiff1 = trans[trans['pairClass'] != classN].loc[:,['pairTransAngleZXZPhi',
                         'pairTransAngleZXZPsi','pairTransAngleZXZTheta']].values
    
    transVectDiffAll = np.concatenate((transVectDiff1,                                     
                                       transDelete.loc[:,['pairTransVectX',
                                                          'pairTransVectY',
                                                          'pairTransVectZ']].values),axis = 0)
    transRotDiffAll = np.concatenate((transRotDiff1,
                                      transDelete.loc[:,['pairTransAngleZXZPhi',
                                                         'pairTransAngleZXZPsi',
                                                         'pairTransAngleZXZTheta']]),axis = 0)
    distVectDiff, distAngDiff, distCombineDiff = tom_A2Odist(transVectDiffAll, transRotDiffAll, 
                                                                   avgShift, avgRot, 
                                                                   worker_n = 1, gpu_list = None, 
                                                                   cmb_metric = 'scale2Ang', pruneRad = 100)
    #call the fit of KDE function 
    _,_,borderVect, pVect = tom_kdeEstimate(distVectSame, 'Cluster %d'%classN, 'vect distance', '',
                              0, 0.05, distVectDiff, 'other clusters')
    _,_,borderAng, pAng = tom_kdeEstimate(distAngSame, 'Cluster %d'%classN, 'angle distance', '',
                              0, 0.05, distAngDiff, 'other clusters')
    _,_,borderComb,pComb = tom_kdeEstimate(distCombineSame, 'Cluster %d'%classN,'combined distance', '',
                              0, 0.05, distCombineDiff, 'other clusters') 
    print('Vect:',borderVect,pVect,'\n',
          'Ang:',borderAng, pAng,'\n',
          'Comb:',borderComb,pComb)
    if classN == 4:
        from py_stats.tom_noiseRandomRotate import tom_noiseRandomRotate
        from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
        coord = np.array([1727.880, 1618.620, 649.036])
        _, angle = tom_eulerconvert_xmipp(145.482,  114.036,  -12.858)
        coordTarget = np.array([1668.697,1662.067,637.071])
        distAng, distCN = tom_noiseRandomRotate(coord, angle, coordTarget, 
              avgShift, avgRot, 1, None, 'scale2Ang', 100, 1000)
        _,_,borderAng,pAng = tom_kdeEstimate(distAngSame, 'c%d'%classN, 'angle distance', '',
                              1, 0.05, distAng, 'c0')
        _,_,borderComb,pComb = tom_kdeEstimate(distCombineSame, 'c%d'%classN, 'combined distance', '',
                              1, 0.05, distCN, 'c0')
        print('Ang:',borderAng, pAng,'\n',
             'Comb:',borderComb,pComb)
        break