import numpy as np
import pandas as pd
import seaborn as sns

from nemotoc.py_io.tom_starread import tom_starread
def recall_Precision(idxPair_std, idxPair_cmp):
    idxPair_std_set = set(idxPair_std)
    idxPair_cmp_set = set(idxPair_cmp)
    TP = len(idxPair_std_set & idxPair_cmp_set)
    FP = len(idxPair_cmp_set - idxPair_std_set)
    FN = len(idxPair_std_set - idxPair_cmp_set)
    recall = 100*TP/(TP+FN)
    precision = 100*TP/(TP+FP)
    return recall, precision
       
def tom_compTransList(stdard, comp):
    assert isinstance(stdard, str)
    assert isinstance(comp, str)
    #load the standard transList
    transList_std = tom_starread(stdard)
    transList_std = transList_std['data_particles']
    #load the to be compared transList
    transList_cmp = tom_starread(comp)
    transList_cmp = transList_cmp['data_particles']
    #compare these two translists
    clusterU_std = np.unique(transList_std['pairClass'].values)
    clusterU_cmp = np.unique(transList_cmp['pairClass'].values)
    
    #swap the idx1 and idx2 for each translist
    swapIdx_std = np.where(transList_std['pairIDX1'] > transList_std['pairIDX2'])[0]
    swapIdx_cmp = np.where(transList_cmp['pairIDX1'] > transList_cmp['pairIDX2'])[0]
    backup_std_idx1 = transList_std.loc[swapIdx_std, 'pairIDX1'].values
    transList_std.loc[swapIdx_std, 'pairIDX1'] =  transList_std.loc[swapIdx_std, 'pairIDX2'].values
    transList_std.loc[swapIdx_std, 'pairIDX2'] = backup_std_idx1
    
    backup_cmp_idx1 = transList_cmp.loc[swapIdx_cmp, 'pairIDX1'].values
    transList_cmp.loc[swapIdx_cmp, 'pairIDX1'] =  transList_cmp.loc[swapIdx_cmp, 'pairIDX2'].values
    transList_cmp.loc[swapIdx_cmp, 'pairIDX2'] = backup_cmp_idx1  
    
    #add additional colums to store the particlePair 
    transList_std['idx1-idx2'] = ['%d-%d'%(int(i), int(j)) for i,j in zip(transList_std['pairIDX1'].values,
                                                                          transList_std['pairIDX2'].values)]
    transList_cmp['idx1-idx2'] = ['%d-%d'%(int(i), int(j)) for i,j in zip(transList_cmp['pairIDX1'].values,
                                                                          transList_cmp['pairIDX2'].values)]
    #calculate the recall score and accuracy score 
    recall_dict = { }
    precision_dict = { }
    for sgC_cmp in clusterU_cmp:
        if sgC_cmp == 0:
            continue
        trans_cmp_sgC_idxPair = transList_cmp[transList_cmp['pairClass'] == sgC_cmp]['idx1-idx2'].values
        recall_dict[sgC_cmp] = 0
        precision_dict[sgC_cmp] = 0
        for sgC_std in clusterU_std:
            trans_std_sgC_idxPair = transList_std[transList_std['pairClass'] == sgC_std]['idx1-idx2'].values
            recall, precision = recall_Precision(trans_std_sgC_idxPair, trans_cmp_sgC_idxPair)
            if recall > recall_dict[sgC_cmp]:
                recall_dict[sgC_cmp] = recall 
            if precision > precision_dict[sgC_cmp]:
                precision_dict[sgC_cmp] = precision
    
                
    #show the final figure as boxplot
    summary_data = pd.DataFrame({'score':list(recall_dict.values()) + list(precision_dict.values()),
                                 'type':['recall']*len(clusterU_cmp) + ['precision']*len(clusterU_cmp)})
    sns.boxplot(x = 'type', y = 'score', data = summary_data)
    sns.stripplot(x = 'type', y = 'score',size=10
          ,jitter=0.1, data = summary_data, color = 'black')
            
            
  
tom_compTransList("../../myNemoProj/projNeuron/run0/allTransformsFillUp.star",
                  "../../myNemoProj/projNeuron/run2/allTransformsFillUp.star",)
    
    