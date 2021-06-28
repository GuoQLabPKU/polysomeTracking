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
from py_io.tom_extractData import tom_extractData
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp

def tom_addTailRibo(transList, pairClass, avgRot, avgShift,
                    cmbDiffMean, cmbDiffStd,
                    oriPartList, pruneRad, 
                    tranListOutput = '',  particleOutput = '',                 
                    NumAddRibo = 1, worker_n = 1, gpu_list = None, 
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
        pairClass        the cluster class to process
        avgRot           the euler angles from ribo1 ==> ribo2 (ZXZ)
                         np.array([phi, psi, theta])
        avgShift         the shifts from ribo1 ==> ribo2 
                         np.array([x,y,z])
        cmbDiffMean    the mean of forward  distance 
                                       
        cmbDiffStd  the  std of forward  difference
                       
        particleStar     starfile of particles for update and checking 
        xyzborder        the xmax/ymax/zmax of the tomo
                         np.array([xmax,ymax,zmax])
        pruneRad         check if two ribsomes are the same according  to coords
        maxDistInpix     the distance standard for normalization

        worker_n/gpu_list    computation for distance(vect/angle) calculation
        transListOutput  ('', Opt)the pathway to store the transList
        particleOutput   ('', Opt)the pathway to store the particle.star
        method            (opt)the method to check if transform pairs are the same
                          now only 'mean+1std' & 'mean+2std' are offered
        NumAddRibo       (1,Opt)# of ribosomes to add at the end of each polysome
    
    OUTPUT
        transList        (dataframe) transList with added ribosomes pairs trans
    
    '''
    if isinstance(transList, str):
        transList = tom_starread(transList)
    if isinstance(oriPartList, str):
        particleStar = tom_starread(oriPartList)
    transListU = transList[transList['pairClass'] == pairClass] 
    polyU = np.unique(transListU['pairLabel'].values) #unique polysome with branches
    if (polyU == -1).all():
        print('no polysomes detected!')
        return transList
    if len(polyU) == 1:
        print('only one polysome, no need link short polys!')
        return transList
    # colllect the information of each tail of each polysome
    tailRiboInfo  = np.zeros([len(polyU), 7])
    headRiboInfo = np.zeros([len(polyU), 7])
    tomoStarList = { }
    tomoIdList = { }
    headIdxList = { }
    tailIdxList = { }
    count = 0

    for eachId in polyU:
        if eachId == -1:
            continue
        polySingle = transListU.loc[transListU['pairLabel'] == eachId] #pairPosInPoly1
        #in future, if eachId == **.1, then consider pairPosInPoly2 of eachId == **
        polySingle = polySingle.sort_values(['pairPosInPoly1'],ascending = False)
        tailRiboInfo[count,:] = np.array([eachId,
                                       polySingle['pairCoordinateX2'].values[0],
                                       polySingle['pairCoordinateY2'].values[0],
                                       polySingle['pairCoordinateZ2'].values[0],
                                       polySingle['pairAnglePhi2'].values[0],
                                       polySingle['pairAnglePsi2'].values[0],
                                       polySingle['pairAngleTheta2'].values[0]])
        headRiboInfo[count,:] = np.array([eachId,
                                       polySingle['pairCoordinateX1'].values[-1],
                                       polySingle['pairCoordinateY1'].values[-1],
                                       polySingle['pairCoordinateZ1'].values[-1],
                                       polySingle['pairAnglePhi1'].values[-1],
                                       polySingle['pairAnglePsi1'].values[-1],
                                       polySingle['pairAngleTheta1'].values[-1]])
        tomoStarList[eachId] = polySingle['pairTomoName'].values[0]
        tomoIdList[eachId] =   polySingle['pairTomoID'].values[0]
        headIdxList[eachId] =  polySingle['pairIDX1'].values[-1]
        tailIdxList[eachId] =  polySingle['pairIDX2'].values[0]
        count += 1
    #add  ribosome(s) to the end of each polysome
    print('tail',tailRiboInfo)
    print('head',headRiboInfo)
    addedRiboInfos, fillUpRiboInfos = tom_extendPoly(tailRiboInfo, avgRot, avgShift, particleStar, pruneRad, 
                                   NumAddRibo, xyzborder)
    print('add info',addedRiboInfos)
    if addedRiboInfos.shape[0] == 0:
        print('warning: can not extend polysomes!')
        return transList
    
    transListAct  =  np.array([]).reshape(-1, 29)
    #check if addedRibos can link the head of other polysomes, get the translist data 
    count = 0
    for i in range(addedRiboInfos.shape[0]):
        for j in range(headRiboInfo.shape[0]):
            if (abs(addedRiboInfos[i,0] - headRiboInfo[j,0]) < 1)  |  (tomoIdList[addedRiboInfos[i,0]] !=  tomoIdList[headRiboInfo[j,0]]):
                continue
            pos1 = addedRiboInfos[i,1:4]
            ang1 = addedRiboInfos[i,4:]
            pos2 = headRiboInfo[j,1:4]
            ang2 = headRiboInfo[j,4:]
            posTr1, angTr1, lenPosTr1, lenAngTr1 = tom_calcPairTransForm(pos1,ang1,pos2,ang2,'exact')
            posTr2, angTr2, _, _ = tom_calcPairTransForm(pos2,ang2,pos1,ang1,'exact')
            #fast check if posTr1, angTr1 is in the same class
            transListAct = np.concatenate((transListAct,             
                             np.array([[addedRiboInfos[i,0], headIdxList[headRiboInfo[j,0]], 
                                       tomoIdList[headRiboInfo[j,0]],
                                       posTr1[0], posTr1[1], posTr1[2], angTr1[0], angTr1[1], angTr1[2],                                        
                                       posTr2[0], posTr2[1], posTr2[2], angTr2[0], angTr2[1], angTr2[2],
                                       lenPosTr1, lenAngTr1,
                                       pos1[0],pos1[1],pos1[2],ang1[0],ang1[1],ang1[2],
                                       pos2[0],pos2[1],pos2[2],ang2[0],ang2[1],ang2[2]]])),
                                        axis = 0)
            count += 1
    #calculate angle /vector distance with angshift/avgrot
    if transListAct.shape[0] == 0:
        print('no need link short polys!')
        return transList
    
    transVect = transListAct[:,3:6]
    print('tranvect', transVect)
    transVect = np.append(transVect, avgShift.reshape(-1,3), axis  = 0)
    
    transAngVect = transListAct[:,6:9] 
    print('transang',transAngVect)
    transAngVect = np.append(transAngVect,avgRot.reshape(-1,3), axis = 0)
    
    maxChunk = tom_memalloc(None, worker_n, gpu_list)#maxChunk can be uint64(cpu) or dict(gpus)
    #using gpu or cpu & make jobList
    if isinstance(worker_n, int):
        from py_cluster.tom_pdist_cpu import tom_pdist
        tmpDir = 'tmpPdistcpu' 
        jobListdict = genJobListCPU(transListAct.shape[0], tmpDir, maxChunk)
    else:        
        if len(gpu_list) == 1:
            from py_cluster.tom_pdist_gpu2 import tom_pdist
                  
        else:
            from py_cluster.tom_pdist_gpu import tom_pdist
        tmpDir = 'tmpPdistgpu' 
        jobListdict = genJobListGpu(transListAct.shape[0], tmpDir, maxChunk)
        
    #calculate distance of vector & angle 
   
    distsVect = tom_pdist(transVect,  maxChunk, worker_n, gpu_list, 'euc', 
                          '', 0, tmpDir,jobListdict, transListAct.shape[0], 0)
    distsAng =  tom_pdist(transAngVect,  maxChunk ,worker_n, gpu_list,'ang',
                          '', 0, tmpDir, jobListdict, transListAct.shape[0],1)
   
    #check if distsvect & distsang compared with avgRot/avgShift are within mean+2std
    #combined the vect/trans distance 
    if cmb_metric == 'scale2Ang':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+distsVect)/2
    elif cmb_metric == 'scale2AngFudge':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+(distsVect*2))/2
    #read the distance scores of this cluster
    print('vect',distsVect)
    print('ang', distsAng)
    print(cmbDiffMean, cmbDiffStd)
    if method == 'mean+2std':
        index = np.argwhere(distsCN < (cmbDiffMean + 20*cmbDiffStd)).reshape(1,-1)[0]
    elif method  == 'mean+_1std':
        index = np.argwhere(distsCN < (cmbDiffMean + 1*cmbDiffStd)).reshape(1,-1)[0]
    
    transList_filter = transListAct[index]  
    #if continue
    if transList_filter.shape[0] == 0:
        print('Warning: can not extend added polysomes for clasesses difference!')
        return transList
    #update the transList and starfile
    #generate starfile for extended tail ribo
    tomoExtendRibo = [tomoStarList[eachId] for eachId in transList_filter[:,0]]
    appendRiboStruct,idxExtendPoly = genParticleFile(particleStar.columns, 
                                                     transList_filter, 
                                                     particleStar.iloc[0,:], tomoExtendRibo, 
                                                     particleStar.shape[0])
    particleStar = pd.concat([particleStar, appendRiboStruct], axis = 0)
    particleStar.reset_index(drop = True, inplace = True)
    #generate transList and append into transList
    idxExtendRibo = np.array([idxExtendPoly[i] for i in transList_filter[:,0]])
 
    transListExtend = genStarFile(transList_filter, idxExtendRibo, transList_filter[:, 1],  
                                  tomoExtendRibo, particleStar, 
                                  pruneRad, oriPartList, pairClass, 
                                  transListU['pairClassColour'].values[0])
    transList = pd.concat([transList, transListExtend], axis = 0)
    transList.reset_index(drop = True, inplace = True)
    #firstly clean the fillup matrix and put them to starfile and transList  
    if fillUpRiboInfos.shape[0] > 0:
        particleN = particleStar.shape[0] 
        transListFillup, tomoNameFillUp, fillUpHeadIdx = genFillupTrans(fillUpRiboInfos, 
                                                                        addedRiboInfos,
                                                         transList_filter[:,0],                                      
                                                         particleN ,avgShift, avgRot ,
                                                         tomoIdList, tomoStarList, idxExtendPoly)  
        
        fillUpRiboStruct, _ = genParticleFile(particleStar.columns, transListFillup, 
                           particleStar.iloc[0,:], tomoNameFillUp, -1)
        #add particlesStar
        print(fillUpRiboStruct)
        particleStar = pd.concat([particleStar, fillUpRiboStruct], axis = 0)
        particleStar.reset_index(drop = True, inplace = True)   
        #add transList
        transListFUData = genStarFile(transListFillup, transListFillup[:,0],
                                      transListFillup[:,1], tomoNameFillUp, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass, transListU['pairClassColour'].values[0])
        
        transList = pd.concat([transList, transListFUData], axis = 0)
        transList.reset_index(drop = True, inplace = True)    
       
        #add tail to fillup trans
        transListT2FillUp, tomoNameT2F = genTransTailToExtend(transList_filter[:,0], 
                                                              tailRiboInfo,fillUpRiboInfos, 
                                                              tailIdxList, fillUpHeadIdx, 
                                                              tomoIdList, tomoStarList,
                                                              avgShift,avgRot)
        transListT2FillUpData =   genStarFile(transListT2FillUp, transListT2FillUp[:,0],
                                      transListT2FillUp[:,1], tomoNameT2F, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass, transListU['pairClassColour'].values[0])
        
        transList = pd.concat([transList, transListT2FillUpData], axis = 0)
        transList.reset_index(drop = True, inplace = True)
    else:
        #generate transList of single extend poly with tail poly
        transListT2Add, tomoNameT2A = genTransTailToExtend(transList_filter[:,0], 
                                                              tailRiboInfo,addedRiboInfos, 
                                                              tailIdxList, idxExtendPoly, 
                                                              tomoIdList, tomoStarList,
                                                              avgShift,avgRot)
        transListT2AddData =   genStarFile(transListT2Add, transListT2Add[:,0],
                                      transListT2Add[:,1], tomoNameT2A, 
                                      particleStar, pruneRad, oriPartList,
                                      pairClass, transListU['pairClassColour'].values[0])        
        
        transList = pd.concat([transList, transListT2AddData], axis = 0)
        transList.reset_index(drop = True, inplace = True)        
    #save the transList and particlStar file
    saveStruct(tranListOutput,transList)
    saveStruct(particleOutput,particleStar)
    return transList
        
     
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


def genTransTailToExtend(polyU, tailInfo,fillUpInfo, tailIdxList, fillUpIdx, 
                         tomoIdList, tomoStarList,avgShift,avgRot):
    
    transList = np.zeros([len(polyU), 29])
    tomoNames = [ ]
    for i, single_poly in enumerate(polyU):
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
        transList[i,17:20] = tailRibo[1:4 ]
        transList[i,20:23] = tailRibo[4:]
        transList[i,23:26] = fuRibo[1:4 ]
        transList[i,26:29] = fuRibo[4:]  
        tomoNames.append(tomoStarList[single_poly])
        
    return transList, tomoNames 


def genFillupTrans(fillUpRiboInfos, addedRiboInfos, polyU, particleN ,avgShift, avgRot ,
                   tomoIdList, tomoStarList, idxExtendPoly):
   
    keepIdx = []
    tomoNames = []
    fillUpHeadIdx = { }

    for i in range(fillUpRiboInfos.shape[0]):
        if fillUpRiboInfos[i,0] in polyU:
            keepIdx.append(i)
    fillUpRibo_keep = fillUpRiboInfos[keepIdx,:]
    #generate tranList
    transListFillup = np.array([]).reshape(-1, 29)
    
    for single_poly in polyU:
        begin = 0
        fillUpHeadIdx[single_poly] = particleN 
        
        fillUpRibosPerPoly = fillUpRibo_keep[fillUpRibo_keep[:,0] == single_poly]
        transList_singlePoly = np.zeros([fillUpRibosPerPoly.shape[0],29])     
        for i in range(fillUpRibosPerPoly.shape[0] - 1):
            transList_singlePoly[i,0] = particleN + begin
            transList_singlePoly[i,1] = particleN + begin + 1
            transList_singlePoly[i,2] = tomoIdList[single_poly]
            transList_singlePoly[i, 3:6] = avgShift
            transList_singlePoly[i, 6:9] = avgRot
            transList_singlePoly[i, 9:12] = -1
            transList_singlePoly[i, 12:15] = -1
            transList_singlePoly[i, 15:17] = -1
            transList_singlePoly[i,17:20] = fillUpRibosPerPoly[i,1:4 ]
            transList_singlePoly[i,20:23] = fillUpRibosPerPoly[i,4:-1 ]
            transList_singlePoly[i,23:26] = fillUpRibosPerPoly[i+1,1:4 ]
            transList_singlePoly[i,26:29] = fillUpRibosPerPoly[i+1,4:-1]  
            begin += 1
       #fillup the final row 
        extendInfo = addedRiboInfos[addedRiboInfos[:,0] == single_poly][0]
        
        transList_singlePoly[-1,:] =  np.array([particleN + begin, idxExtendPoly[single_poly], tomoIdList[single_poly],
                avgShift[0],avgShift[1], avgShift[2], avgRot[0], avgRot[1], avgRot[2],
                -1,-1,-1, -1,-1,-1,-1,-1,
                fillUpRibosPerPoly[-1, 1], fillUpRibosPerPoly[-1, 2], 
                fillUpRibosPerPoly[-1, 3],fillUpRibosPerPoly[-1, 4],
                fillUpRibosPerPoly[-1, 5],
                fillUpRibosPerPoly[-1, 6], extendInfo[1], extendInfo[2], extendInfo[3],
                extendInfo[4], extendInfo[5], extendInfo[6]])
        
        transListFillup = np.concatenate((transListFillup, transList_singlePoly), axis = 0)
        particleN += (begin+1)
        tomoName = [tomoStarList[single_poly] for i in range(fillUpRibosPerPoly.shape[0])]
        tomoNames.extend(tomoName)
        
    return transListFillup,tomoNames,fillUpHeadIdx



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
    pixel = particleStar['rlnDetectorPixelSize'].values[0]
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
    

def genParticleFile(colName, transList, example_info, tomoName, extandFlag): 
    particleStruct = { }
    idxExtendPoly = { }
    for single_name in colName:
        particleStruct[single_name] = []
         
    catchName = ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ',
                  'rlnAngleRot','rlnAngleTilt','rlnAnglePsi',
                  'rlnMicrographName','rlnImageName','rlnCtfImage']
    remName = [ ]
    count = 0
    for i in range(transList.shape[0]):
        polyU = transList[i,0]
        particleStruct['rlnCoordinateX'].append(transList[i,17])
        particleStruct['rlnCoordinateY'].append(transList[i,18])
        particleStruct['rlnCoordinateZ'].append(transList[i,19])
        _,angles = tom_eulerconvert_xmipp(transList[i,20], transList[i,21], transList[i,22], 'tom2xmipp')
        particleStruct['rlnAngleRot'].append(angles[0])
        particleStruct['rlnAngleTilt'].append(angles[1])
        particleStruct['rlnAnglePsi'].append(angles[2])
        particleStruct['rlnImageName'].append('notImplemented.mrc')
        particleStruct['rlnCtfImage'].append('notImplemented.mrc')
        idxExtendPoly[polyU] = count + extandFlag
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
            particleStruct['rlnGroupNumber'].append(0)
            if count == 0:
                catchName.append('rlnGroupNumber')
        count += 1
        #pick remaing colName
        if len(remName) == 0:
            remName = [i for i in colName if i not in catchName]
        for singleName in remName:
            particleStruct[singleName].append(example_info[singleName])
    #make a dataframe
    particleStruct['rlnMicrographName'] = tomoName
    appendRibo = pd.DataFrame(particleStruct)
    
    return appendRibo,idxExtendPoly
            
        
            

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
    