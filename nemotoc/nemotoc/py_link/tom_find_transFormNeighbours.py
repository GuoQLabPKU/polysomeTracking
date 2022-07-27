import numpy as np
from nemotoc.py_io.tom_extractData import tom_extractData
def findNeighOneTomo(pairList):

    
    st = tom_extractData(pairList)
    idxU = np.unique(np.concatenate((st['p1']['orgListIDX'],
                           st['p2']['orgListIDX'])))
    neigh_N_plus = np.zeros((np.max(idxU) + 1, 3), dtype = np.int)
    neigh_N_minus = np.zeros((np.max(idxU) + 1, 3), dtype = np.int)
    
    all_idx1 = st['p1']['orgListIDX']  #the row rank in the original row data star file(each row == one ribosome)
    all_pos1 = st['p1']['positions']
    all_idx2 = st['p2']['orgListIDX']
    all_pos2 = st['p2']['positions']
    all_class = st['label']['pairClass']  #1D array, which cluster belongs to
         
    for i in range(len(idxU)):         
         act_idx = idxU[i]        
         classes1, pT1 = find_neigh(all_idx1, all_pos1, all_class, act_idx) 
         #classes1 = -1 && p_act = []  OR classes1 = [1,2] && p_act = [x,y,z]
         classes2, pT2 = find_neigh(all_idx2, all_pos2, all_class, act_idx)
         
         if len(pT1) > 0:
             pT = pT1
         else:
             pT = pT2
        
         neigh_N_plus[idxU[i] , :] = classes1
         neigh_N_minus[idxU[i] ,:] = classes2
    
    idx1 = pairList['pairIDX1'].values
    idx2 = pairList['pairIDX2'].values
   
    for i in range(len(idxU)):
        act_idx = idxU[i]
        tmpIdx1 = np.where(idx1 == act_idx)[0]
        rowNames = pairList._stat_axis.values.tolist()
        rowNamesSel = [rowNames[i] for i in tmpIdx1]
        for ii in rowNamesSel:
            pairList.loc[ii, 'pairNeighPlus1'] = vect2ClassStr(neigh_N_plus[act_idx  ,:])
            pairList.loc[ii, 'pairNeighMinus1'] = vect2ClassStr(neigh_N_minus[act_idx ,:])
            
        tmpIdx2 = np.where(idx2 == act_idx)[0]
        rowNames = pairList._stat_axis.values.tolist()
        rowNamesSel = [rowNames[i] for i in tmpIdx2]
        for ii in rowNamesSel:
            pairList.loc[ii, 'pairNeighPlus2'] = vect2ClassStr(neigh_N_plus[act_idx  ,:])
            pairList.loc[ii, 'pairNeighMinus2'] = vect2ClassStr(neigh_N_minus[act_idx ,:])  
            
    return pairList.values, neigh_N_plus, neigh_N_minus
             
def find_neigh(all_idx, all_pos, all_class, act_Idx):
    p_act = [ ]
    tmpInd = np.where(all_idx == act_Idx)[0]
    
    if len(tmpInd) == 0:
        tmpInd = -1
        classes = -1
    
    elif len(tmpInd) > 0:
        p_act = all_pos[tmpInd,:] #2-D array
        tmp = all_class[tmpInd]
        
        if len(tmp) > 3:
            tmp = tmp[0:3]
        if len(tmp) < 3:
            tt = np.ones(3)*-1  #tt 2-D aray
            tt[0:len(tmp)] = tmp
            tmp = tt 
        ##only analysis for three classes???
        
        classes = tmp
        classes[np.where(classes == 0)] = -1
       
    ##not interested in class == 0,which failed to from cluster
    if len(p_act) > 0:
        p_act = p_act[0,:]
        
    return classes, p_act  #p_act should be 1D-array


def vect2ClassStr(vect):
    tmp = ';'.join([str(i) if i!=-1 else '' for i in vect ])
    tmp = tmp.replace(';;',';')
    if (len(tmp) == 0) | (tmp[0] == ';'):
        tmp = '-1;'
    
    if tmp[-1] != ';':
        tmp = "%s;"%tmp
        
    return tmp
        
    
    
    
            
            
            