import numpy as np
import pandas as pd
import os
import shutil

from py_io.tom_starread import tom_starread
from py_io.tom_starwrite import tom_starwrite
from py_mergePoly.tom_extendPoly import tom_extendPoly
from py_transform.tom_calcPairTransForm import tom_calcPairTransForm
from py_memory.tom_memalloc import tom_memalloc
from py_cluster.tom_pdist_gpu2 import fileSplit
from py_cluster.tom_pdist_gpu2 import genjobsList_oneGPU
from py_cluster.tom_calc_packages import tom_calc_packages
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp

def tom_addTailRibo(transList, pairClass, avgRot, avgShift,
                    cmbDiffMean, cmbDiffStd,
                    oriPartList, pruneRad, 
                    tranListOutput = '',  particleOutput = '',                 
                    NumAddRibo = 1, addRiboInfoPrint=1, worker_n = 1, gpu_list = None, 
                    xyzborder = None, cmb_metric = 'scale2Ang',
                    method = 'mean+2std'):
    '''
    TOM_ADDTAILRIBO put one/two/.. ribosomes at the end of each polysome to try 
    to link shorter polysome.
    
    EXAMPLE
    transList = tom_addTailRibo();
    
    PARAMETERS
    
    INPUT
        transList        
        pairClass        the polysome cluster class to process
        avgRot           the avg euler angles from ribo1 ==> ribo2 (ZXZ)
                         np.array([phi, psi, theta])
        avgShift         the avg shifts from ribo1 ==> ribo2 
                         np.array([x,y,z])
        cmbDiffMean      the mean of forward  distance 
                                       
        cmbDiffStd       the  std of forward  distance 
                       
        oriPartList      starfile of particles for update (add fillup ribos) 
        pruneRad         check if two ribsomes are close to each other(also for
                         angle distance normalization)
        
        transListOutput  ('', Opt)the pathway to store the transList
        particleOutput   ('', Opt)the pathway to store the particle.star
        worker_n/gpu_list    computation for distance(vect/angle) calculation
        NumAddRibo       (1,Opt)# of ribosomes to add at the end of each polysome
        xyzborder        the xmax/ymax/zmax of the tomo
                         np.array([xmax,ymax,zmax])
        method            (opt)the method to check if filled up ribos are in the same 
                          cluster class,now only 'mean+1std' & 'mean+2std' are offered
       
        
    
    OUTPUT
        transList        (dataframe) transList with fillup ribosomes transList
    
    '''
    if isinstance(transList, str):
        transList = tom_starread(transList)
    if isinstance(oriPartList, str):
        particleStar = tom_starread(oriPartList)
    transListU = transList[transList['pairClass'] == pairClass] 
    polyU = np.unique(transListU['pairLabel'].values) #unique polysome with branches
    if (polyU == -1).all():
        print('No polysomes detected! Check your transList!')
        return transList
    if len(polyU) == 1:
        print('Only one polysome, no need link short polys!')
        return transList
    # colllect the information of the tail ribo of each polysome
    tailRiboInfo  = np.array([]).reshape(-1, 7)
    headRiboInfo = np.array([]).reshape(-1, 7)
    tomoStarList = { } #store the tomoName for each polysome
    tomoIdList = { } #store the tomoId for each polysome
    headRiboIdxList = { } #store the index of head polysome for each polysome
    tailRiboIdxList = { } #store the index of tail polysome for each polysome
    

    for eachId in polyU:
        if eachId == -1:
            continue
        polySingle = transListU.loc[transListU['pairLabel'] == eachId] #pairPosInPoly1
        #in future, if eachId == **.1, then consider pairPosInPoly2 of eachId == **
        polySingle = polySingle.sort_values(['pairPosInPoly1'],ascending = False) #find the head/tail of this polysome
        if polySingle['pairIDX2'].values[0] not in tailRiboIdxList.values():#one ribosome can belong to different polysomes
            tailRiboInfo = np.concatenate((tailRiboInfo,
                                           np.array([[eachId,
                                                     polySingle['pairCoordinateX2'].values[0],
                                                     polySingle['pairCoordinateY2'].values[0],
                                                     polySingle['pairCoordinateZ2'].values[0],
                                                     polySingle['pairAnglePhi2'].values[0],
                                                     polySingle['pairAnglePsi2'].values[0],
                                                     polySingle['pairAngleTheta2'].values[0]]])), axis = 0)
    
    
   
        if polySingle['pairIDX1'].values[-1] not in headRiboIdxList.values():
            headRiboInfo = np.concatenate((headRiboInfo,
                                           np.array([[eachId,
                                                    polySingle['pairCoordinateX1'].values[-1],
                                                    polySingle['pairCoordinateY1'].values[-1],
                                                    polySingle['pairCoordinateZ1'].values[-1],
                                                    polySingle['pairAnglePhi1'].values[-1],
                                                    polySingle['pairAnglePsi1'].values[-1],
                                                    polySingle['pairAngleTheta1'].values[-1]]])),axis = 0)
    
        tomoStarList[eachId] = polySingle['pairTomoName'].values[0]
        tomoIdList[eachId] =   polySingle['pairTomoID'].values[0]
        headRiboIdxList[eachId] =  polySingle['pairIDX1'].values[-1]
        tailRiboIdxList[eachId] =  polySingle['pairIDX2'].values[0]
    #add ribosome(s) to the end of each polysome
    fillUpRiboInfos, fillUpMiddleRiboInfos = tom_extendPoly(tailRiboInfo, avgRot, avgShift, particleStar, pruneRad, 
                                   NumAddRibo, xyzborder)
    #fillUpRiboInfos store the information of filled up ribosomes which directly link another polysome
    #the structure of fillUpRiboInfos are the same as headRiboInfo
    #fillUpMiddleRiboInfos store the information of filled up ribsomes when we added more than one ribosome at tail of one polysome
    if fillUpRiboInfos.shape[0] == 0:
        print('Warning: can not extend polysomes! This may because hypo ribos are \
              already in the tomo but of different class OR out of the tomo border.')
        return transList
    #calculate angle /vector distance with avgshift/avgrot
    transListAct = genTransList(fillUpRiboInfos, headRiboInfo, tomoIdList, headRiboIdxList)
    if transListAct.shape[0] == 0:
        print('Can not link short polys! This may polys are in different tomos.')
        return transList
    
    transVect = transListAct[:,3:6]
    #append avgShift to vect trans array for tom_pdist
    transVect = np.append(transVect, avgShift.reshape(-1,3), axis  = 0)   
    transAngVect = transListAct[:,6:9] 
    transAngVect = np.append(transAngVect,avgRot.reshape(-1,3), axis = 0)
    #calculate distance between hypo trans and average trans
    distsCN = getCombinedDist(transListAct.shape[0], transVect, transAngVect, worker_n, gpu_list, cmb_metric, pruneRad)
    
    if method == 'mean+2std':
        index = np.argwhere(distsCN < (cmbDiffMean + 2*cmbDiffStd)).reshape(1,-1)[0]
    elif method  == 'mean+_1std':
        index = np.argwhere(distsCN < (cmbDiffMean + 1*cmbDiffStd)).reshape(1,-1)[0]
#    index1 = np.argwhere(distsCN <= (cmbDiffMean)).reshape(1,-1)[0]
#    index2 = np.argwhere(distsCN >= cmbDiffStd).reshape(1,-1)[0]
#    index  = np.intersect1d(index1, index2)
    transList_filter = transListAct[index]
    if transList_filter.shape[0] == 0:
        print('Warning: can not add fillup ribos at tail of polysomes')
        return transList
    #debug for ribosome info output
    if addRiboInfoPrint:
        debug_output(transList_filter, distsCN, index)        
    ##################################################
    ##################################################
    ##################################################
    #generate particle infos for fill up ribos
    #update the transList and starfile
    tomoFillUpRibo = []
    lastId = -1
    for eachId in transList_filter[:,0]:
        if eachId == lastId:
            continue
        lastId = eachId
        tomoFillUpRibo.append(tomoStarList[eachId]) #find tha tomoname of each fill up ribos according to polyID
            
    appendRiboStruct,idxFillUpRiboDict = genParticleFile(particleStar.columns, 
                                                     transList_filter, 
                                                     particleStar.iloc[0,:], tomoFillUpRibo, 
                                                     particleStar.shape[0])
    particleStar = pd.concat([particleStar, appendRiboStruct], axis = 0)
    particleStar.reset_index(drop = True, inplace = True)
    
    #generate transList and append into transList
    idxFillUpRibo = np.array([idxFillUpRiboDict[i] for i in transList_filter[:,0]])
    tomoFillUpRibo = [tomoStarList[i] for i in transList_filter[:,0]]
    transListFillUp = genStarFile(transList_filter, idxFillUpRibo, transList_filter[:, 1],  
                                  tomoFillUpRibo, particleStar, 
                                  pruneRad, oriPartList, pairClass, 
                                    '0.00-0.00-1.00') #transListU['pairClassColour'].values[0])
    transList = pd.concat([transList, transListFillUp], axis = 0)
    transList.reset_index(drop = True, inplace = True)
    
    #next update transList & particleStar for middle fill up ribos(if NumAddRibo > 1)
    if fillUpMiddleRiboInfos.shape[0] > 0:
    
        transListFillupMiddle, tomoNameFillUpMiddle, fillUpMiddleIdx = genFillupMiddleTrans(fillUpMiddleRiboInfos, 
                                                                        fillUpRiboInfos,
                                                                        transList_filter[:,0],                                      
                                                                        particleStar.shape[0],avgShift, avgRot ,
                                                                        tomoIdList, tomoStarList, idxFillUpRiboDict)  
        
        fillUpRiboStruct, _ = genParticleFile(particleStar.columns, transListFillupMiddle, 
                           particleStar.iloc[0,:], tomoNameFillUpMiddle, -1)
        #update particlesStar
        particleStar = pd.concat([particleStar, fillUpRiboStruct], axis = 0)
        particleStar.reset_index(drop = True, inplace = True)   
        #update transList
        transListFUData = genStarFile(transListFillupMiddle, transListFillupMiddle[:,0],
                                      transListFillupMiddle[:,1], tomoNameFillUpMiddle, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass,
                                      '0.00-0.00-1.00')#, transListU['pairClassColour'].values[0])
        
        transList = pd.concat([transList, transListFUData], axis = 0)
        transList.reset_index(drop = True, inplace = True)    
       
        #remember update the translist for each tail ribo of one poly to head fillup ribo of the same poly
        transListT2F, tomoNameT2F = genTransTailToExtend(transList_filter[:,0], 
                                                              tailRiboInfo,fillUpMiddleRiboInfos, 
                                                              tailRiboIdxList, fillUpMiddleIdx, 
                                                              tomoIdList, tomoStarList,
                                                              avgShift,avgRot)
        transListT2FillUpData =   genStarFile(transListT2F, transListT2F[:,0],
                                      transListT2F[:,1], tomoNameT2F, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass, 
                                      '0.00-0.00-1.00')#transListU['pairClassColour'].values[0])
        
        transList = pd.concat([transList, transListT2FillUpData], axis = 0)
        transList.reset_index(drop = True, inplace = True)
    else:
        #remember update the translist for each tail ribo of one poly to head fillup ribo of the same poly
        transListT2F, tomoNameT2F = genTransTailToExtend(transList_filter[:,0], 
                                                              tailRiboInfo,fillUpRiboInfos, 
                                                              tailRiboIdxList, idxFillUpRiboDict, 
                                                              tomoIdList, tomoStarList,
                                                              avgShift,avgRot)
        transListT2AddData = genStarFile(transListT2F, transListT2F[:,0],
                                      transListT2F[:,1], tomoNameT2F, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass,
                                      '0.00-0.00-1.00')#transListU['pairClassColour'].values[0])        
        
        transList = pd.concat([transList, transListT2AddData], axis = 0)
        transList.reset_index(drop = True, inplace = True)        
    #save the transList and particlStar file
    saveStruct(particleOutput,particleStar)
    return transList


def getCombinedDist(transSize, transVect, transAngVect, worker_n, gpu_list, cmb_metric, pruneRad):    
    maxChunk = tom_memalloc(None, worker_n, gpu_list)#maxChunk can be uint64:cpu or dict:gpus
    #using gpu or cpu & make jobList
    if isinstance(worker_n, int):
        from py_cluster.tom_pdist_cpu import tom_pdist
        tmpDir = 'tmpPdistcpu' 
        jobListdict = genJobListCPU(transSize, tmpDir, maxChunk)
    else:        
        if len(gpu_list) == 1:
            from py_cluster.tom_pdist_gpu2 import tom_pdist                
        else:
            from py_cluster.tom_pdist_gpu import tom_pdist
        tmpDir = 'tmpPdistgpu' 
        jobListdict = genJobListGpu(transSize, tmpDir, maxChunk)
    #the jobList is quite important for tom_pdist calculation. For example, we add one ribosome at end of one polysome(P1) 
    #then link another polysome(P2),then transvect should have two rows:one is for P1->P2, another is for avgShift.Then we can 
    #make jobList like [0,-1]. 0 reprensts the first row of transVect(p1->p2), -1 represents the last row of transVect(avgShift).
    #Then we can make tom_pdist calculate the distance between row 0 and row -1.Therefore, the strucure of jobList should look 
    #like [0 -1;1 -1;2 -1;3 -1;4 -1.....]. Each non -1 element in jobList will be compared with the -1 element in jobList. Therefore,
    #each hypothetical trans will be compared with avgShif/avgRot
    
    
    #calculate distance of vector & angle 
    distsVect = tom_pdist(transVect,  maxChunk, worker_n, gpu_list, 'euc', 
                          '', 0, tmpDir, jobListdict, transSize, 0)
    distsAng =  tom_pdist(transAngVect,  maxChunk ,worker_n, gpu_list,'ang',
                          '', 0, tmpDir, jobListdict, transSize,1)
    #check if distsvect & distsang compared with avgRot/avgShift are within mean+2std
    #combined the vect/trans distance 
    if cmb_metric == 'scale2Ang':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+distsVect)/2
    elif cmb_metric == 'scale2AngFudge':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+(distsVect*2))/2    
    return distsCN

def genTransList(fillUpRiboInfos, headRiboInfo, tomoIdList, headRiboIdxList):
    transListAct  =  np.array([]).reshape(-1, 29)
    #check if fillUpRibo can link the head of other polysomes, get the translist data 
    count = 0
    for i in range(fillUpRiboInfos.shape[0]):
        for j in range(headRiboInfo.shape[0]):
            if (abs(fillUpRiboInfos[i,0] - headRiboInfo[j,0]) < 1)  |  (tomoIdList[fillUpRiboInfos[i,0]] !=  tomoIdList[headRiboInfo[j,0]]):
                #the first condition is whether two ribosomes are from the same polysome. the second condition is whether 
                #two ribsome are from the same tomogram
                continue
            pos1 = fillUpRiboInfos[i,1:4]
            ang1 = fillUpRiboInfos[i,4:]
            pos2 = headRiboInfo[j,1:4]
            ang2 = headRiboInfo[j,4:]
            posTr1, angTr1, lenPosTr1, lenAngTr1 = tom_calcPairTransForm(pos1,ang1,pos2,ang2,'exact')
            posTr2, angTr2, _, _ = tom_calcPairTransForm(pos2,ang2,pos1,ang1,'exact')
            #fast check if posTr1, angTr1 is in the same class
            transListAct = np.concatenate((transListAct,             
                             np.array([[fillUpRiboInfos[i,0], headRiboIdxList[headRiboInfo[j,0]], 
                                       tomoIdList[headRiboInfo[j,0]],
                                       posTr1[0], posTr1[1], posTr1[2], angTr1[0], angTr1[1], angTr1[2],                                        
                                       posTr2[0], posTr2[1], posTr2[2], angTr2[0], angTr2[1], angTr2[2],
                                       lenPosTr1, lenAngTr1,
                                       pos1[0],pos1[1],pos1[2],ang1[0],ang1[1],ang1[2],
                                       pos2[0],pos2[1],pos2[2],ang2[0],ang2[1],ang2[2]]])),
                                        axis = 0)
            count += 1
    return transListAct    

def saveStruct(filename,starfile):
    
    if len(filename) == 0:
        return

    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"] = [ ]
    for i,j in enumerate(starfile.columns):
        header["fieldNames"].append('_%s #%d'%(j,i+1))
    tom_starwrite(filename, starfile, header)


def genTransTailToExtend(polyIdx, tailInfo,fillUpInfo, tailIdxList, fillUpIdx, 
                         tomoIdList, tomoStarList,avgShift,avgRot):
    
    uniq_poly = np.unique(polyIdx)
    transList = np.zeros([len(uniq_poly), 29])
    tomoNames = [ ]
    last_poly = -1
    i = 0
    for single_poly in polyIdx:      
        if single_poly == last_poly:
            continue
        last_poly = single_poly
        tailRibo = tailInfo[tailInfo[:,0] == single_poly][0] #one array data 
        fuRibo = fillUpInfo[fillUpInfo[:,0] == single_poly][0] #one array data 
        transList[i,0] = tailIdxList[single_poly]
        transList[i,1] = fillUpIdx[single_poly]
        transList[i,2] = tomoIdList[single_poly]
        transList[i, 3:6] = avgShift
        transList[i, 6:9] = avgRot
        transList[i, 9:12] = -1
        transList[i, 12:15] = -1
        transList[i, 15:17] = -1
        transList[i,17:20] = tailRibo[1:4]
        transList[i,20:23] = tailRibo[4:]
        transList[i,23:26] = fuRibo[1:4]
        transList[i,26:29] = fuRibo[4:]      
        i+=1
        tomoNames.append(tomoStarList[single_poly])
    return transList, tomoNames 


def genFillupMiddleTrans(fillUpRiboMiddleInfos, fillUpRiboInfos, polyIds, 
                         particleN ,avgShift, avgRot ,
                         tomoIdList, tomoStarList, idxFillUpRiboDict):
   
    keepIdx = []
    tomoNames = []
    fillUpMiddleIdx = { } #this dict store the idx of each fillupmiddle ribo,but only one ribo for each poly

    for i in range(fillUpRiboMiddleInfos.shape[0]):
        if fillUpRiboMiddleInfos[i,0] in polyIds:
            keepIdx.append(i)
    fillUpMiddleRibo_keep = fillUpRiboMiddleInfos[keepIdx,:] #only keep fillupmiddle ribos of successfully filup polys
    #generate tranList
    transListFillupMiddle = np.array([]).reshape(-1, 29)
    last_poly = -1
    for single_poly in polyIds:
        if single_poly == last_poly:
            continue
        last_poly = single_poly
        begin = 0
        fillUpMiddleIdx[single_poly] = particleN 
        
        fillUpMiddleRibosPerPoly = fillUpMiddleRibo_keep[fillUpMiddleRibo_keep[:,0] == single_poly]
        transList_singlePoly = np.zeros([fillUpMiddleRibosPerPoly.shape[0],29])     
        for i in range(fillUpMiddleRibosPerPoly.shape[0] - 1):
            transList_singlePoly[i,0] = particleN + begin #idx of thie ribo
            transList_singlePoly[i,1] = particleN + begin + 1 #idx of next ribo
            transList_singlePoly[i,2] = tomoIdList[single_poly]
            transList_singlePoly[i, 3:6] = avgShift
            transList_singlePoly[i, 6:9] = avgRot
            transList_singlePoly[i, 9:12] = -1
            transList_singlePoly[i, 12:15] = -1
            transList_singlePoly[i, 15:17] = -1
            transList_singlePoly[i,17:20] = fillUpMiddleRibosPerPoly[i,1:4] #pos
            transList_singlePoly[i,20:23] = fillUpMiddleRibosPerPoly[i,4:] #angle 
            transList_singlePoly[i,23:26] = fillUpMiddleRibosPerPoly[i+1,1:4] #pos 
            transList_singlePoly[i,26:29] = fillUpMiddleRibosPerPoly[i+1,4:]#angle 
            begin += 1
       #fillup the final row 
        fillUpRibo = fillUpRiboInfos[fillUpRiboInfos[:,0] == single_poly][0]
        
        transList_singlePoly[-1,:] =  np.array([particleN + begin, idxFillUpRiboDict[single_poly], tomoIdList[single_poly],
                avgShift[0],avgShift[1], avgShift[2], avgRot[0], avgRot[1], avgRot[2],
                -1,-1,-1, -1,-1,-1,-1,-1,
                fillUpMiddleRibosPerPoly[-1, 1], fillUpMiddleRibosPerPoly[-1, 2], 
                fillUpMiddleRibosPerPoly[-1, 3],fillUpMiddleRibosPerPoly[-1, 4],
                fillUpMiddleRibosPerPoly[-1, 5],
                fillUpMiddleRibosPerPoly[-1, 6], 
                fillUpRibo[1], fillUpRibo[2], fillUpRibo[3],
                fillUpRibo[4], fillUpRibo[5], fillUpRibo[6]])
        
        transListFillupMiddle = np.concatenate((transListFillupMiddle, transList_singlePoly), axis = 0)
        particleN += (begin+1)
        tomoName = [tomoStarList[single_poly] for i in range(fillUpMiddleRibosPerPoly.shape[0])]
        tomoNames.extend(tomoName)
        
    return transListFillupMiddle,tomoNames,fillUpMiddleIdx



def genStarFile(transList, idx1, idx2, tomoName12, particleStar, maxDist, oriPartList, 
                pairClass, pairColour):
    idx1 = idx1.astype(np.int)
    idx2 = idx2.astype(np.int)
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"] = ['pairIDX1','pairIDX2','pairTomoID',
                                  'pairTransVectX','pairTransVectY','pairTransVectZ',
                                  'pairTransAngleZXZPhi', 'pairTransAngleZXZPsi','pairTransAngleZXZTheta',
                                  'pairInvTransVectX','pairInvTransVectY','pairInvTransVectZ',
                                  'pairInvTransAngleZXZPhi', 'pairInvTransAngleZXZPsi','pairInvTransAngleZXZTheta',
                                  'pairLenTrans','pairAngDist',
                                  'pairCoordinateX1','pairCoordinateY1','pairCoordinateZ1',
                                  'pairAnglePhi1','pairAnglePsi1','pairAngleTheta1',
                                  'pairClass1','pairPsf1',
                                   'pairNeighPlus1','pairNeighMinus1', 'pairPosInPoly1',
                                  'pairCoordinateX2','pairCoordinateY2','pairCoordinateZ2',
                                  'pairAnglePhi2','pairAnglePsi2','pairAngleTheta2',
                                  'pairClass2','pairPsf2',
                                   'pairNeighPlus2','pairNeighMinus2', 'pairPosInPoly2',
                                  'pairTomoName','pairPixelSizeAng',
                                  'pairOriPartList',
                                  'pairMaxDist','pairClass','pairClassColour',
                                  'pairLabel','pairScore']
    idxTmp = np.array([idx1, idx2]).transpose()
    #pixel = particleStar['rlnDetectorPixelSize'].values[0]
    pixel = 3.42
    classesPart1 = particleStar['rlnClassNumber'].values[idx1]
    classesPart2 = particleStar['rlnClassNumber'].values[idx2]
    psfsPart1 = particleStar['rlnCtfImage'].values[idx1]
    psfsPart2 = particleStar['rlnCtfImage'].values[idx2]
    neighPMPart = np.tile(['-1:-1','-1:-1'],(transList.shape[0],1))
    posInPolyPart = np.repeat(-1,transList.shape[0])
    

    #make the final startSt dict, which is differen with st dict
    #transform the array into dataframe
    startSt_data = pd.DataFrame(idxTmp, columns = header["fieldNames"][0:2])
    startSt_data[header["fieldNames"][2]] = transList[:,2].astype(np.int)
    transform_data = pd.DataFrame(transList[:,3:23],columns = header["fieldNames"][3:23])
    startSt_data = pd.concat([startSt_data,transform_data],axis = 1)
    startSt_data[header["fieldNames"][23]] = classesPart1
    startSt_data[header["fieldNames"][24]] = psfsPart1
    startSt_data[header["fieldNames"][25:27]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][27]] = posInPolyPart
    startSt_data[header["fieldNames"][28:34]] = pd.DataFrame(transList[:,23:29])
    startSt_data[header["fieldNames"][34]] = classesPart2
    startSt_data[header["fieldNames"][35]] = psfsPart2
    startSt_data[header["fieldNames"][36:38]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][38]] = posInPolyPart
    startSt_data[header["fieldNames"][39]] = tomoName12
    startSt_data[header["fieldNames"][40]] = np.repeat([pixel], transList.shape[0])
    startSt_data[header["fieldNames"][41]] = np.repeat([oriPartList],transList.shape[0])
    startSt_data[header["fieldNames"][42]] = np.repeat([maxDist],transList.shape[0])
    startSt_data[header["fieldNames"][43]] = pairClass
    startSt_data[header["fieldNames"][44]]  = np.repeat([pairColour],transList.shape[0])
    startSt_data[header["fieldNames"][45:47]] = pd.DataFrame(np.tile([-1,-1],(transList.shape[0],1)))
    
    return startSt_data
    

def genParticleFile(colName, transList, example_info, tomoName, particleN): 
    particleStruct = { }
    idxFillUpRibo = { } #this dict store the idx of each fill up ribo
    for single_name in colName:
        particleStruct[single_name] = []
         
    catchName = ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ',
                  'rlnAngleRot','rlnAngleTilt','rlnAnglePsi',
                  'rlnMicrographName','rlnImageName','rlnCtfImage']
    remainName = [ ]
    last_poly = -1
    count = 0
    for i in range(transList.shape[0]):
        polyId = transList[i,0] #for fillup middle ribos, this is idx(not polyid!)
        if polyId == last_poly:
            continue
        last_poly = polyId
        particleStruct['rlnCoordinateX'].append(transList[i,17])
        particleStruct['rlnCoordinateY'].append(transList[i,18])
        particleStruct['rlnCoordinateZ'].append(transList[i,19])
        _,angles = tom_eulerconvert_xmipp(transList[i,20], transList[i,21], transList[i,22], 'tom2xmipp')
        particleStruct['rlnAngleRot'].append(angles[0])
        particleStruct['rlnAngleTilt'].append(angles[1])
        particleStruct['rlnAnglePsi'].append(angles[2])
        particleStruct['rlnImageName'].append('notImplemented.mrc')
        particleStruct['rlnCtfImage'].append('notImplemented.mrc')
        idxFillUpRibo[polyId] = count + particleN
        if 'rlnOriginX' in colName:
            particleStruct['rlnOriginX'].append(0)
            particleStruct['rlnOriginY'].append(0)
            particleStruct['rlnOriginZ'].append(0)
            if count == 0:
                catchName.append('rlnOriginX')
                catchName.append('rlnOriginY')
                catchName.append('rlnOriginZ')
    
        if 'rlnClassNumber' in colName:
            particleStruct['rlnClassNumber'].append(-1) #class == -1 ==>represents added image Ribos
            if count == 0:
                catchName.append('rlnClassNumber')
        if 'rlnGroupNumber' in colName:
            particleStruct['rlnGroupNumber'].append(-1)
            if count == 0:
                catchName.append('rlnGroupNumber')
        count += 1
        #pick remaing colName
        if len(remainName) == 0:
            remName = [i for i in colName if i not in catchName]
        for singleName in remName:
            particleStruct[singleName].append(example_info[singleName])
    #make a dataframe
    particleStruct['rlnMicrographName'] = tomoName
    appendRibo = pd.DataFrame(particleStruct)
    
    return appendRibo,idxFillUpRibo

def genJobListGpu(lenJobs, tmpDir, maxChunk): #maxChunk is one dict for gpu(gpuindex be the key)
    jobList = np.zeros([lenJobs,2], dtype = np.int) 
    jobList[:,0] = np.arange(lenJobs)
    jobList[:,1] = np.repeat(-1, lenJobs)
         
    #split the jobsList into different GPUs
    gpu_list, start_site, file_size  = fileSplit(maxChunk, lenJobs)  
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir) #remove the .npy anyway
    os.mkdir(tmpDir)
    jobsListSt_dict = { }
    for gpu_id, startsite, filesize in zip(gpu_list, start_site, file_size):
        packages = genjobsList_oneGPU(startsite, filesize, maxChunk[gpu_id])
        jobListSt = [ ] # is one list with dicts stored
        for i in range(packages.shape[0]):
            jobListChunk = jobList[packages[i,0]:packages[i,1], :]
            jobListSt.append({ })
            jobListSt[i]["file"] = "%s/jobListChunk_%d_gpu%d.npy"%(tmpDir, i, gpu_id)
            jobListSt[i]["start"] = packages[i,0]
            jobListSt[i]["stop"] = packages[i,1]
            np.save(jobListSt[i]["file"], jobListChunk)  #will waste a long time for writing and reading!  
        jobsListSt_dict[gpu_id] = jobListSt
    return jobsListSt_dict
      
def genJobListCPU(lenJobs, tmpDir, maxChunk):
    jobList = np.zeros([lenJobs,2], dtype = np.int) 
    jobList[:,0] = np.arange(lenJobs)
    jobList[:,1] = np.repeat(-1, lenJobs) 
    numOfPackages = np.int(np.ceil(lenJobs/maxChunk)) #if the size of joblist < maxChunck, only single cpu will used  
    packages = tom_calc_packages(numOfPackages, lenJobs) #split the jobList into different size, the packages is one array
    #make new directory to store the Tmp jobList
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir) #remove the .npy anyway
    os.mkdir(tmpDir)
    jobListSt = [ ] # is one list with dicts stored
    for i in range(packages.shape[0]):
        jobListChunk = jobList[packages[i,0]:packages[i,1], :]
        jobListSt.append({ })
        jobListSt[i]["file"] = "%s/jobListChunk_%d.npy"%(tmpDir, i)
        jobListSt[i]["start"] = packages[i,0]
        jobListSt[i]["stop"]  = packages[i,1]
        np.save(jobListSt[i]["file"], jobListChunk)  
    return jobListSt  

def debug_output(transList_filter, dists, idx):
    print('Sucessfully fill up these ribos')
    print('euler angle:Rot      Tilt       Psi, position:X          Y           Z ')
    for i in range(transList_filter.shape[0]):
        _,angles = tom_eulerconvert_xmipp(transList_filter[i,20], transList_filter[i,21], transList_filter[i,22], 'tom2xmipp')
        print('%.3f  %.3f  %.3f    %.3f    %.3f    %.3f'%(angles[0],angles[1],angles[2], transList_filter[i,17],transList_filter[i,18],transList_filter[i,19]))
    print('trans angle:Phi    Psi      Theta, trans shift:X         Y           Z')
    for i in range(transList_filter.shape[0]):
        print('%.3f  %.3f  %.3f    %.3f    %.3f    %.3f'%(transList_filter[i,6],transList_filter[i,7],transList_filter[i,8], 
                                                          transList_filter[i,3],transList_filter[i,4],transList_filter[i,5]))   
    print('the distscmb with Tavg is:')
    print(dists[idx])
       