import numpy as np

def tom_linkTransforms(pairList, maxBranchDepth, offset_PolyID):
    '''
    TOM_FIND_CONNECTEDTANSFORMS finds connedted transform 

    pairs=tom_find_connectedTransforms(pairList,outputName)

    PARAMETERS

    INPUT
       pairList               pair datframe file with the same class and same tomo            
       outputName             (opt.) name of the output pair star file
       branchDepth            (1) (opt.) Depth for branch tracing
       
    OUTPUT
       pairListAlg            aligned pair list 
       names                  the names of the pair list
       branchFound            if this class and tomo has polysomes with branch
       polyID                 number of polysomes in this tomo class

    EXAMPLE

    REFERENCES
    
    '''
    #the input is one dataframe with the same class as well as the same tomo
    indList = pairListToIndList(pairList)
    branchDepth, branchFound = findBranchDepth(indList, maxBranchDepth)
    IndListByPath = searchPathForEachEntry(indList, branchDepth)
    indList = addLabelToIndList(indList, IndListByPath, offset_PolyID)
    pairList = updatePairListByIndList(pairList, indList)
    offset_PolyID = np.max(np.unique(indList[:,2]))
    
    return pairList.values, list(pairList.columns), branchFound, offset_PolyID
    
    
    
def pairListToIndList(pairList):
    ind1 = pairList['pairIDX1'].values
    ind2 = pairList['pairIDX2'].values
    indList = np.array([ind1, ind2, np.zeros(len(ind1)), np.zeros(len(ind2)), np.ones(len(ind2))], dtype = np.int)  
    indList = indList.transpose()
    return indList
    

def findBranchDepth(indList, brDepth):
    branchNr1 = indList.shape[0] - len(np.unique(indList[:,0]))
    branchNr2 = indList.shape[0] - len(np.unique(indList[:,1]))
    branchNr = np.max([branchNr1, branchNr2])
    
    if branchNr > 0:
        branchFound = 1
        branchDepth = brDepth
    else:
        branchFound = 0
        branchDepth = 1
    
    return branchDepth, branchFound


def searchPathForEachEntry(indList, branchDepth):
    allPathN = [ ]
    for brNum in range(branchDepth):
        for searchStart in range(indList.shape[0]):
            tmpPath = searchPathForward(indList, searchStart, brNum)
            allPathN = uniquePathAdd(allPathN, tmpPath)
            
    return allPathN  #record information of polysome pathways 
            

def searchPathForward(cmbInd, zz, branchNr):  #branch begin with 0 end with the branch number - 1
    # zz:the row number in the pairindex datafram. Remember that cmbInd should has very long rows , zz ~= inf
    zzPath = 1 #not begin with 0
    tmpPath = np.array([[cmbInd[zz, 0], cmbInd[zz,1], zz, zzPath, 1]], dtype = np.uint64)
    cmbInd_circleFlag = np.zeros(cmbInd.shape[0], dtype = np.int)
    circRepeat = 0
    maxTrials = 1000
    
    for i in range(maxTrials):
        cmbInd_circleFlag[zz] = 1 #the polysomes track begins 
        idx2Search = cmbInd[zz, 1]
        idxT1 = np.where(cmbInd[:,0] == idx2Search)[0]
        
        if len(idxT1) > 1:
            idxT1 = idxT1[branchNr] #more than one branch
             
        zz = idxT1
        if len(idxT1) > 0:
            if cmbInd_circleFlag[idxT1] == 1:
                circRepeat = 1
        if (len(idxT1) == 0) | (circRepeat == 1):
            break
        
        zzPath += 1
        tmpPath = np.concatenate((tmpPath, np.array([[cmbInd[zz, 0], cmbInd[zz,1], zz, zzPath, 1]],dtype = np.uint64)
                                  ), axis = 0)
    if i == maxTrials:
        print('Warning: max tries reached in searchPathForward,\nwhich means you will get a long polysome.')
    
    return tmpPath  #the unit64 class 

def uniquePathAdd(allPath, newPath): #allPath == [], should be a ptr
    if len(newPath) == 0:
        return allPath
    
    if len(allPath) == 0:
        newPath_sort = newPath[newPath[:,2].argsort()]
        allPath.append(newPath_sort)
        return allPath
 
    memAN = np.zeros(len(allPath), dtype = np.int)  #1D array
    memNA = np.zeros(len(allPath), dtype = np.int)
    memBranch = np.zeros(len(allPath), dtype = np.int)
  
    newPathP = np.sort(newPath[:,2])
    newPath_sort = newPath[newPath[:,2].argsort()]
    for i in range(len(allPath)):  #never try to modify a list when cycle it!! You can cylce another thing to modity one list 
        actPath = allPath[i]
        actPath = actPath[:,2] #YOU SORTED THIS COL
        interSNPathactPATH = fastIntersect(newPathP, actPath)
        memAN[i] = (len(interSNPathactPATH) == len(newPathP))  #newPathP is subset of allPath[i]
        memNA[i] = (len(interSNPathactPATH) == len(actPath))  #allPath[i] is subset of newPath
        memBranch[i] = 0 #if has intersect 
        
        if (len(interSNPathactPATH) > 0)  &  (len(interSNPathactPATH) < len(actPath)) & (memAN[i] == 0):
            memBranch[i] = 1

    if np.sum(memBranch) > 0:
        ind = np.where(memBranch == 1)[0]
        #PathUnion = np.array([],dtype = np.uint64).reshape(-1, 5)
        for single_ind in ind:
            actPath = allPath[single_ind]
            actPathP = actPath[:,2]
            diff_path = np.setdiff1d( newPathP, actPathP, assume_unique = True)
            idxAadd = np.where(newPathP==diff_path[:,None])[-1]
            newPath_sort[idxAadd, 4] = 2 
            newPathUnion = np.concatenate((actPath, newPath_sort[idxAadd,:]  ), axis = 0)           
            #PathUnion = np.concatenate((PathUnion, newPathUnion), axis = 0)
            
        #PathUnion = PathUnion[PathUnion[:,2].argsort()]
        #PathUnion = PathUnion.astype(np.uint64)
        newPathUnion = newPathUnion[newPathUnion[:,2].argsort()]     
        for idx in ind:
            allPath[idx] = newPathUnion  #make the allPath has duplicate elements?
        #allPath[ind[0]] = newPathUnion
        #allPath_new = [allPath[i] for i in range(len(allPath)) if i not in ind[1:]] #Wenhong's idea       
        allPath_new = [allPath[i] for i, j in enumerate(memNA) if j==0] #discard the path which are the subset of pathnew
        return allPath_new
    
    if ((np.sum(memAN) == 0)  & (np.sum(memNA) == 0)):
        allPath.append(newPath_sort)
        return allPath
    
    if np.sum(memNA) > 0:
        allPath_new = [allPath[i] for i, j in enumerate(memNA) if j==0]
        allPath_new.append(newPath_sort)
        return allPath_new   
    if np.sum(memAN) > 0:
        return allPath
def fastIntersect(A,B):  #A-B are 1D arrays
    if (len(A) > 0) & (len(B) > 0):
        P = np.zeros(np.max( [np.max(A), np.max(B)] )+ np.uint64(1), dtype = np.uint64)
        P[A] = 1
        C = B[P[B].astype(np.bool)]    
    else:
        C = np.array([])
        
    return C  #C has the element from A & B
        
        
            
def addLabelToIndList(indList, allPathN, offset_PolyID): #the indList should be a ptr
    
    Path_N = len(allPathN)
    for i in range(Path_N):
        single_path = allPathN[i]
        indList[single_path[:,2], 2] = i+offset_PolyID+1  #which polysome it belongs, 0 means no polysome detected
        indList[single_path[:,2], 3] = single_path[:,3]
        indList[single_path[:,2], 4] = single_path[:,4]
        
    
    return indList
       
def updatePairListByIndList(pairList, indList):
    uLabels = np.unique(indList[:,2]) #this is the polysomes rank, begin with 1. 
    for i in range(len(uLabels)):
        idxTmp = np.where(indList[:,2] == uLabels[i])[0]
        idxPair, indListByLabel = getPathSortedindListByLabel(indList, idxTmp)
        rowNames = pairList._stat_axis.values.tolist()
        rowNamesSel = [rowNames[i] for i in idxPair]
        pairList.loc[rowNamesSel,:] = updatePairListOneLabel(pairList.loc[rowNamesSel,:],  indListByLabel, uLabels[i], rowNamesSel)
        
    return pairList
        
        
        
def getPathSortedindListByLabel(indList, idxTmp):
    indListByLabel = indList[idxTmp,:]
    idx = indListByLabel[:,3].argsort()
    indListByLabel = indListByLabel[idx,:]
    idxPair = idxTmp[idx]
    return idxPair, indListByLabel
    
def updatePairListOneLabel(pairList,indListByLabel, uLabel, rowNamesSel):
     '''
     the debug plot function is developing 
     '''
#    do_debugPlot = 0
     
     isLinear = 1
    
     pairList_len = pairList.shape[0]
     for i,j in enumerate(rowNamesSel):
         pairList.loc[j, 'pairLabel'] = uLabel + (indListByLabel[i, 4] - 1)/10
         pairList.loc[j, 'pairPosInPoly1'] = indListByLabel[i,3]
        
         if (i == (pairList_len-1)) & (indListByLabel[-1, 1] == indListByLabel[0,0] )  & (pairList_len > 1 ):
            #circular
             pairList.loc[j,'pairPosInPoly2'] = 1
             isLinear = 0
         if (i == (pairList_len-1)) & (indListByLabel[-1, 1] == indListByLabel[0,1] )  & (pairList_len > 1):
            #circular extended
             pairList.loc[j,'pairPosInPoly2'] = 2
             isLinear = 0
         if isLinear:
             pairList.loc[j, 'pairPosInPoly2'] = indListByLabel[i,3] + 1
         isLinear = 1
#        debugPlot(do_debugPlot, pairList,i)
        
     return pairList.values
            
#def  debugPlot(doPlot, pairList,i):
#    if doPlot:
        
        
        
    
    
    
    
    
    
        
        