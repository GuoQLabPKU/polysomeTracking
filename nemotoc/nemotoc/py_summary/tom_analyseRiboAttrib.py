import numpy as np
import pandas as pd
import itertools


from nemotoc.py_io.tom_starread import tom_starread, generateStarInfos
from nemotoc.py_io.tom_starwrite import tom_starwrite
from nemotoc.py_log.tom_logger import Log

def tom_analyseRiboAttrib(transList, save_dir = '',
                          transListB4Relink = None, particleStar = None):
    '''
    TOM_ANALYSERIBOATTRIB summaries the overlapping of different transform class
    and ribosomes dropped during relink and those far away from others to 
    form transform.
    '''
    
    log = Log('ribosome overlapping analysis').getlog()
    
    if isinstance(transList, str):
        transList = tom_starread(transList)
        transList = transList['data_particles']
    if isinstance(particleStar, str):    
        particleStar = tom_starread(particleStar)
        particleStar = particleStar['data_particles']
    #initial variables
    pairName_list = [ ]
    riboC1Nr_list = [ ]
    riboC2Nr_list = [ ]
    overlapNr_list = [ ]
    overlapRatioOfC1_list = [ ]
    overlapRatioOfC2_list = [ ]
    aloneRiboNr = 0
    dropRelinkRiboNr = 0
    dropRelinkTransformNr = 0
    
    
    #analyse the overlapping ratio of ribosomes from different classes
    classU = np.unique(transList['pairClass'].values)
    ribo_Idx =  np.unique(np.concatenate((transList['pairIDX1'].values,
                                              transList['pairIDX2'].values)))
    if len(classU)==1:
        log.warning("Only class %d, skip overlapping analysis"%classU[0])
        return
        
    for classPair in itertools.combinations(classU, 2):
        c1, c2 = classPair
        transList_c1 = transList[transList['pairClass'] == c1]
        transList_c2 = transList[transList['pairClass'] == c2]
        ribo_c1Idx = np.unique(np.concatenate((transList_c1['pairIDX1'].values,
                                               transList_c1['pairIDX2'].values)))
        ribo_c2Idx = np.unique(np.concatenate((transList_c2['pairIDX1'].values,
                                               transList_c2['pairIDX2'].values)))
        #analyse the overlapping 
        overlap = np.intersect1d(ribo_c1Idx, ribo_c2Idx, assume_unique = True)
        pairName_list.append('c%d_c%d'%(c1,c2))
        riboC1Nr_list.append(len(ribo_c1Idx))
        riboC2Nr_list.append(len(ribo_c2Idx))
        overlapNr_list.append(len((overlap)))
        overlapRatioOfC1_list.append(np.round(len(overlap)/len(ribo_c1Idx),3))
        overlapRatioOfC2_list.append(np.round(len(overlap)/len(ribo_c2Idx),3))
    
    
    #analyse those dropped ribosome pairs 
    if transListB4Relink is None:
        if 0 in classU:
            c0 = transList[transList['pairClass'] == 0]
            c0_idx =  np.unique(np.concatenate((c0['pairIDX1'].values,
                                               c0['pairIDX2'].values)))
            #save 
            starInfo = generateStarInfos()
            starInfo['data_particles'] = particleStar.iloc[c0_idx,:]
            tom_starwrite('%s/stat_NonMeanningRibos.star'%save_dir,starInfo) 
        else:    
            log.warning("Skip summary dropped ribosome pairs")
        
    else:
        if isinstance(transListB4Relink, str):
            transListB4Relink = tom_starread(transListB4Relink)
            transListB4Relink = transListB4Relink['data_particles']
        #the number of transpairs dropped during relink
        dropRelinkTransformNr = transListB4Relink.shape[0] - transList.shape[0]
        #analyse the overlapping
        ribo_b4RelinkIdx = np.unique(np.concatenate((transListB4Relink['pairIDX1'].values,
                                                     transListB4Relink['pairIDX2'].values)))       
        dropRelinkRiboNr = len(np.setdiff1d(ribo_b4RelinkIdx, ribo_Idx, assume_unique = True))
        #save 
        dropRelinkIdx = np.setdiff1d(ribo_b4RelinkIdx, ribo_Idx, assume_unique = True)
        #save 
        starInfo = generateStarInfos()
        starInfo['data_particles'] = particleStar.iloc[dropRelinkIdx,:]
        tom_starwrite('%s/stat_NonMeanningRibos.star'%save_dir,starInfo)         
    
    #analyse alone ribsome Nr
    try:
        aloneRiboNr = particleStar.shape[0] - len(ribo_b4RelinkIdx)
        aloneRiboIdx =  np.setdiff1d(particleStar.index, ribo_b4RelinkIdx, assume_unique = True)
        starInfo = generateStarInfos()
        starInfo['data_particles'] = particleStar.iloc[aloneRiboIdx,:]
        tom_starwrite('%s/stat_AloneRibos.star'%save_dir,starInfo)  
    except UnboundLocalError:
        aloneRiboNr = particleStar.shape[0] - len(ribo_Idx)        
        aloneRiboIdx =  np.setdiff1d(particleStar.index, ribo_Idx, assume_unique = True)
        starInfo = generateStarInfos()
        starInfo['data_particles'] = particleStar.iloc[aloneRiboIdx,:]
        tom_starwrite('%s/stat_AloneRibos.star'%save_dir,starInfo)  
    #make dataframes to store these information
    overlapData = pd.DataFrame({'pairClass':pairName_list,
                                'riboNr_C1':riboC1Nr_list,
                                'riboNr_C2':riboC2Nr_list,
                                'overlapRiboNr':overlapNr_list,
                                'overlapRatioOfC1':overlapRatioOfC1_list,
                                'overlapRatioOfC2':overlapRatioOfC2_list})
    if transListB4Relink is not None:
        dropData = pd.DataFrame({'dropRelinkTransNr':[dropRelinkTransformNr],
                                 'keepTransNr':[transList.shape[0]],
                                 'dropRelinkRiboNr':[dropRelinkRiboNr],
                                 'aloneRiboNr':[aloneRiboNr],
                                 'totalRiboNr':[particleStar.shape[0]],
                                 'keepRiboNr':[len(ribo_Idx)]})
    else:
        dropData = ''
    #save the data 
    if len(save_dir)>0:
        starInfo = generateStarInfos()
        starInfo['data_particles'] = overlapData
        tom_starwrite('%s/statOverlapRibos.star'%save_dir,starInfo) 
        if not isinstance(dropData, str):
            starInfo['data_particles'] = dropData
            tom_starwrite('%s/statDropRibos.star'%save_dir,starInfo)
    else:
        print(overlap)
        print(dropData)
    
   