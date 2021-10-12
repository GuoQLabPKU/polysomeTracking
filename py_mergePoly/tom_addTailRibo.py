import numpy as np
import pandas as pd
from ast import literal_eval
from alive_progress import alive_bar 
import random
import os
import dill
import multiprocessing as mp
import shutil
import gc

from py_log.tom_logger import Log
from py_io.tom_starread import tom_starread, generateStarInfos
from py_io.tom_starwrite import tom_starwrite
from py_io.tom_extractData import tom_extractData
from py_mergePoly.tom_extendPoly import tom_extendPoly
from py_cluster.tom_A2Odist import tom_A2Odist
from py_transform.tom_calcPairTransForm import tom_calcPairTransForm
from py_transform.tom_eulerconvert_xmipp import tom_eulerconvert_xmipp
from py_stats.tom_calcPvalues import tom_calcPvalues

def tom_addTailRibo(statePolyAll_list, pairList, pairClass, avgRot, avgShift,
                    cmbDistMaxMeanStd, oriPartList, pruneRad, 
                    tranListOutput = '',  particleOutput = '',                 
                    numAddRibo = 1, verbose=1, method = 'max',accept_threshold = 0.05,
                    worker_n = 1, gpu_list = None, 
                    xyzborder = None, cmb_metric = 'scale2Ang'):
    '''
    TOM_ADDTAILRIBO put one/two/.. ribosomes at the end of each polysome to try 
    to link shorter polysome.
    
    EXAMPLE
    transList = tom_addTailRibo();
    
    PARAMETERS
    
    INPUT
        pairList        
        pairClass        the polysome cluster class to process
        avgRot           the avg euler angles from ribo1 ==> ribo2 (ZXZ)
                         np.array([phi, psi, theta])
        avgShift         the avg shifts from ribo1 ==> ribo2 
                         np.array([x,y,z])
        cmbDistMaxMeanStd     the max,mean,std of forward  distance 
                                       

                       
        oriPartList      starfile of particles for update (add fillup ribos) 
        pruneRad         check if two ribsomes are close to each other(also for
                         angle distance normalization)
        
        transListOutput  ('', Opt)the pathway to store the transList
        particleOutput   ('', Opt)the pathway to store the particle.star
        NumAddRibo       (1,Opt)# of ribosomes to add at the end of each polysome
        verbose          (1,Opt) if print the information of filled up ribosomes
        method            ('max',opt)the method to check if filled up ribos are in the same 
                          cluster class,now only 'max' & 'lognorm'  & 'genFit'are offered
                          extreme:[min, max]//lognorm:pvalue based on lognorm distribution//genFit:pvalue based on kde fitting
        worker_n/gpu_list    computation for distance(vect/angle) calculation
        
        xyzborder        the xmax/ymax/zmax of the tomo
                         np.array([xmax,ymax,zmax])
        cmb_metric        ('scale2Ang', Opt)the methods to combine vect&angle distance. 
                          Now only 'scale2Ang' & 'scale2AngFudge' arer offered
        accept_threshold  only useful when lognrom/genFit is switched on
                           
     
    OUTPUT
        transList        (dataframe) transList with fillup ribosomes transList
    
    '''
    randN = random.randint(0,1000)
    log = Log('polysome filling up %d'%randN).getlog()
    
    if isinstance(pairList, str):
        pairList = tom_starread(pairList)
        pairList = pairList['data_particles']
    if isinstance(oriPartList, str):
        particleStar = tom_starread(oriPartList, pairList['pairPixelSizeAng'].values[0])
        particleStar_data = particleStar['data_particles']
        particleSt = tom_extractData(oriPartList)
        particleStar_data['if_fillUp'] = [-1]*particleStar_data.shape[0]
   
    polyStateU = statePolyAll_list[statePolyAll_list['pairClass'] == pairClass]
    polyU = np.unique(polyStateU['pairLabel'].values) #unique polysome with branches
    if len(polyU) == 0:
        log.warning('No polysomes detected in class%d! Check your transList!'%pairClass)
        return pairList
    if len(polyU) == 1:
        log.warning('Only one polysome detected, no need link polys!')
        return pairList
   
    #get the head/tail ribo position & angle information
    tailRiboIdx = statePolyAll_list[statePolyAll_list['ifWoOut'] == 1]['pairIDX'].values
    headRiboIdx = statePolyAll_list[statePolyAll_list['ifWoIn'] ==  1]['pairIDX'].values
 
    #colllect the information of the tail& head ribosomes of each polysome
    tailRiboInfo  = np.zeros((len(tailRiboIdx), 8))
    headRiboInfo = np.zeros((len(headRiboIdx), 8)) 
    
    tailRiboInfo[:,0] = tailRiboIdx
    headRiboInfo[:,0] = headRiboIdx  
    tailRiboInfo[:,1] = statePolyAll_list[statePolyAll_list['ifWoOut'] == 1]['pairLabel'].values
    headRiboInfo[:,1] = statePolyAll_list[statePolyAll_list['ifWoIn'] == 1]['pairLabel'].values
    tailRiboInfo[:,2:5] =  particleSt["p1"]["positions"][tailRiboIdx, ]   
    headRiboInfo[:,2:5] =  particleSt["p1"]["positions"][headRiboIdx, ]
    tailRiboInfo[:,5:] =   particleSt["p1"]["angles"][tailRiboIdx, ]   
    headRiboInfo[:,5:] =   particleSt["p1"]["angles"][headRiboIdx, ]                
    #add ribosome(s) to the end of each polysome
    #fillUpRiboInfos store the information of filled up ribosomes which link another polysome
    #fillUpMiddleRiboInfos store the information of filled up ribsomes when we added more than one ribos
    fillUpRiboInfos, fillUpMiddleRiboInfos = tom_extendPoly(tailRiboInfo, avgRot, avgShift, particleSt, pruneRad, 
                                                            numAddRibo, xyzborder)
    if fillUpRiboInfos.shape[0] == 0:
        log.warning('''Warning: can not extend polysomes! This is because filledup ribos are 
                       already in the tomo OR out of the tomo border!''')
        return pairList
    
    #calculate angle /vector distance between hypothetical trans and head ribos of other polysomes
    transListAct = genTransList(worker_n, fillUpRiboInfos, headRiboInfo, statePolyAll_list)
    if transListAct.shape[0] == 0:
        log.warning('''Can not link polys! This may be polys are in different tomos or 
                       from the same polysome.''')
        return pairList
    
    _,_,distsCN = tom_A2Odist(transListAct[:, 4:7], 
                              transListAct[:, 7:10],
                              avgShift, avgRot,
                              1, gpu_list,
                              cmb_metric, pruneRad)
    
    if method == 'max':
        index = np.argwhere(distsCN <= cmbDistMaxMeanStd[0]).reshape(1,-1)[0]      
        fillUpProb_list = [1]*len(index)
    if method == 'lognorm':
        #load fitparams data
        path, _ = os.path.split(tranListOutput)
        fitData = pd.read_csv('%s/vis/fitDistanceDist/distanceDistFit_c%d.csv'%(path, pairClass), sep = ",")
        fitParam = fitData[fitData['distribution'] == 'lognorm']['fit_params'].values[0]       
        fitParam = literal_eval(fitParam)
        distCNNorm = (distsCN - cmbDistMaxMeanStd[1])/cmbDistMaxMeanStd[2]
        pvalues = tom_calcPvalues(distCNNorm, 'lognorm', fitParam)
        index = np.argwhere(pvalues > accept_threshold).reshape(1,-1)[0]
        fillUpProb_list = pvalues[index]
    if method == 'genFit':
        path, _ = os.path.split(tranListOutput)
        distCNNorm = (distsCN - cmbDistMaxMeanStd[1])/cmbDistMaxMeanStd[2]
        with open('%s/vis/fitDistanceDist/c%d_dill.pkl'%(path, pairClass), 'rb') as f:
            kde = dill.load(f)
        
        kde_cdf = 1-np.array([kde.integrate_box_1d(-np.inf, x) for x in distCNNorm])
        index = np.argwhere(kde_cdf > accept_threshold).reshape(1,-1)[0]
        fillUpProb_list = kde_cdf[index]
    fillUpProb_info = np.zeros((len(index), 2))
    fillUpProb_info[:,0] = fillUpProb_list
    fillUpProb_info[:,1] = transListAct[index, 0]
    
    transAct_filter = transListAct[index]
    if transAct_filter.shape[0] == 0:
        log.warning('''Can not link  polys. This is because filluped ribos form different transform class''')
        return pairList
    #debug for filled up ribosome infos output
    if verbose:
        debug_output(transAct_filter, distsCN[index])   
        
    ##################################################
    ##################################################

    #update the transList as well as particle.starfile
    keepTailRiboIdxForFillUpRibo, index = np.unique(transAct_filter[:,0], return_index = True)
    fillRiboCoords = transAct_filter[index, 18:21]
    fillRiboAngles = transAct_filter[index, 21:24]
    tomoNamesOfFillUpRibos = [statePolyAll_list[statePolyAll_list['pairIDX'] == i]['pairTomoName'].values[0] \
                              for i in keepTailRiboIdxForFillUpRibo]
    tomoNamesOfDupFillUpRibos = [statePolyAll_list[statePolyAll_list['pairIDX'] == i]['pairTomoName'].values[0] \
                                for i in transAct_filter[:,0]]
            
    appendRiboStruct = updateParticle( fillRiboCoords, fillRiboAngles, 
                                       particleStar_data.iloc[0,:], tomoNamesOfFillUpRibos, 
                                       particleStar_data.shape[0],  particleStar['type'])
    appendRiboStruct['if_fillUp'] = [1]*len(index)
    idxOfFillUpRibos = { } #this is the idx of filled up ribsomes in the particles.star
    for i,j in enumerate(keepTailRiboIdxForFillUpRibo):
        idxOfFillUpRibos[j] = particleStar_data.shape[0] + i
    idxOfDupFillUpRibos = np.array([idxOfFillUpRibos[i] for i in transAct_filter[:, 0]])
    
    particleStar_data = pd.concat([particleStar_data, appendRiboStruct], axis = 0)
    particleStar_data.reset_index(drop = True, inplace = True)
    
    #update transList and append into transList  
    transAct_filter[:,1] = idxOfDupFillUpRibos
    transListOfFillUpRibo = updateTransList(transAct_filter, 
                                            tomoNamesOfDupFillUpRibos, 
                                            particleStar_data, particleStar['pixelSize'],
                                            particleStar['type'],
                                            pruneRad, oriPartList, pairClass) 
    transListOfFillUpRibo['fillUpProb'] = fillUpProb_info[:,0]
    pairList = pd.concat([pairList, transListOfFillUpRibo], axis = 0)
    pairList.reset_index(drop = True, inplace = True)

    #####################################################
    #####################################################  
    
    #update transList & particleStar for middle fill up ribos(if NumAddRibo > 1)
    if fillUpMiddleRiboInfos.shape[0] > 0: 
        transActFillupMiddleRibo, tomoNameFillUpMiddleRibo, idxFillUpMiddleRibo = updateTransOfMiddleFillupRibos(
                                                                                fillUpMiddleRiboInfos, 
                                                                                fillRiboCoords,fillRiboAngles,
                                                                                keepTailRiboIdxForFillUpRibo,
                                                                                particleStar_data.shape[0],
                                                                                avgShift, avgRot,
                                                                                statePolyAll_list,                                                                        
                                                                                idxOfFillUpRibos)  
        
        
        riboCoords =  transActFillupMiddleRibo[:, 18:21]
        riboAngles =  transActFillupMiddleRibo[:, 21:24]
        fillUpRiboStruct = updateParticle(riboCoords, riboAngles,   particleStar_data.iloc[0,:], 
                                          tomoNameFillUpMiddleRibo, particleStar_data.shape[0],
                                          particleStar['type'])
        fillUpRiboStruct['if_fillUp'] = [1.1]*fillUpRiboStruct.shape[0]
        #update particlesStar
        particleStar_data = pd.concat([particleStar_data, fillUpRiboStruct], axis = 0)
        particleStar_data.reset_index(drop = True, inplace = True)   
        #update transList
        transListFillupMiddleRibo = updateTransList(transActFillupMiddleRibo, 
                                                    tomoNameFillUpMiddleRibo, 
                                                    particleStar_data, 
                                                    particleStar['pixelSize'],
                                                    particleStar['type'],
                                                    pruneRad, oriPartList, pairClass)
        transListFillupMiddleRibo['fillUpProb'] = [1.1]*len(transListFillupMiddleRibo)
        pairList = pd.concat([pairList, transListFillupMiddleRibo], axis = 0)
        pairList.reset_index(drop = True, inplace = True)    
       
        #update the translist: each tail ribo of one poly -> first filluped ribo of the same poly
        transActT2F, tomoNameT2F = genTransTailToFillUp(keepTailRiboIdxForFillUpRibo, 
                                                        tailRiboInfo, fillUpMiddleRiboInfos, 
                                                        statePolyAll_list, idxFillUpMiddleRibo, 
                                                        avgShift, avgRot)
        transListT2F = updateTransList(transActT2F, tomoNameT2F, 
                                      particleStar_data, particleStar['pixelSize'],
                                      particleStar['type'],
                                      pruneRad, oriPartList,pairClass)
        transListT2F['fillUpProb'] = [1.1]*len(transListT2F)
        pairList = pd.concat([pairList, transListT2F], axis = 0)
        pairList.reset_index(drop = True, inplace = True)
    else:
        
        transActT2F, tomoNameT2F = genTransTailToFillUp(keepTailRiboIdxForFillUpRibo, 
                                                        tailRiboInfo,fillUpRiboInfos, 
                                                        statePolyAll_list, idxOfFillUpRibos, 
                                                        avgShift,avgRot)
    
        transListT2F = updateTransList(transActT2F, tomoNameT2F, 
                                      particleStar_data, particleStar['pixelSize'],
                                      particleStar['type'],
                                      pruneRad, oriPartList,pairClass)
        transListT2F['fillUpProb'] = [1.1]*len(transListT2F)
        pairList = pd.concat([pairList, transListT2F], axis = 0)
        pairList.reset_index(drop = True, inplace = True)        
    #save the transList and particlStar file
    particleStar['data_particles'] = particleStar_data
    saveStruct(particleOutput, particleStar)
    return pairList

def genTransList(worker_n, fillUpRiboInfos, headRiboInfo, polyInfoList):
    if (worker_n == None) | (worker_n == 1):
        transList = genTransListSub(-1, fillUpRiboInfos, headRiboInfo, polyInfoList, '')
        return transList
    else:
        #make temp directory for data saving
        temp_dir = "tempTrans"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
        else:
            os.mkdir(temp_dir)
        #parallel calculation
        processes = dict()
        npr = worker_n
        avail_cpu = mp.cpu_count()
        npr = min(npr, avail_cpu)
        print('use %d cpus to calculate fillingUp trans pairs'%npr)
        spl_ids = np.array_split(np.arange(fillUpRiboInfos.shape[0]),npr) 
        spl_ids = [i for i in spl_ids if len(i) > 0]
        for pr_id, spl_id in enumerate(spl_ids):
            pr = mp.Process(target = genTransListSub, args=(pr_id, fillUpRiboInfos[spl_id,:], 
                                                             headRiboInfo, polyInfoList, temp_dir))
            pr.start()
            processes[pr_id] = pr
        for pr_id, pr in zip(processes.keys(), processes.values()):
            pr.join()
            if pr_id != pr.exitcode:
                errorInfo = 'the process %d ended usuccessfully [%d]'%(pr_id, pr.exitcode)
                raise RuntimeError(errorInfo)
                
        gc.collect() #free the memory
        #combined the npy data
        transList = np.array([]).reshape(0, 30)
        for pr_id, pr in zip(processes.keys(), processes.values()):
            transListAct = np.load('%s/trans_%d.npy'%(temp_dir, pr_id))
            transList = np.concatenate((transList, transListAct),axis=0)
        #delete the temp files       
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir) 
    return transList

def genTransListSub(pr_id, fillUpRiboInfos, headRiboInfo, polyInfoList, temp_dir):
    transListAct  =  np.array([]).reshape(-1, 30)
    with alive_bar(fillUpRiboInfos.shape[0], title="calculate trans pairs for fillingUp") as bar:
        for i in range(fillUpRiboInfos.shape[0]):
            for j in range(headRiboInfo.shape[0]):
                polyId1 = fillUpRiboInfos[i,1];polyId2 = headRiboInfo[j,1]
                tomo1 = polyInfoList[polyInfoList['pairLabel'] == polyId1]['pairTomoID'].values[0]
                tomo2 = polyInfoList[polyInfoList['pairLabel'] == polyId2]['pairTomoID'].values[0]
                if (polyId1 == polyId2) | (tomo1 != tomo2):
                    #the first condition is whether two ribosomes are from the same polysome. 
                    #the second condition is whether two ribosome are from the same tomogram
                    continue
                
                pos1 = fillUpRiboInfos[i,2:5]
                ang1 = fillUpRiboInfos[i,5:]
                
                pos2 = headRiboInfo[j,2:5]
                ang2 = headRiboInfo[j,5:]  
                posTr1, angTr1, lenPosTr1, lenAngTr1 = tom_calcPairTransForm(pos1,ang1,pos2,ang2,'exact')
                transListAct = np.concatenate((transListAct, np.array([[fillUpRiboInfos[i,0],
                                           fillUpRiboInfos[i,1], headRiboInfo[j,0], 
                                           tomo1, posTr1[0], posTr1[1], posTr1[2], 
                                           angTr1[0], angTr1[1], angTr1[2],                                        
                                           -1, -1, -1, -1, -1, -1,
                                           lenPosTr1, lenAngTr1,
                                           pos1[0],pos1[1],pos1[2],
                                           ang1[0],ang1[1],ang1[2],
                                           pos2[0],pos2[1],pos2[2],
                                           ang2[0],ang2[1],ang2[2]]])),
                                           axis = 0)  
            bar()
    if pr_id == -1:
        return transListAct   
    else:
        np.save('%s/trans_%d.npy'%(temp_dir, pr_id), transListAct)
        return os._exit(pr_id)
            
     
def saveStruct(filename,starfile):  
    if len(filename) == 0:
        return
    if isinstance(starfile, dict):
        tom_starwrite(filename, starfile)
    else:    
        starInfo = generateStarInfos()
        starInfo['data_particles'] = starfile
        tom_starwrite(filename, starInfo)


def genTransTailToFillUp(riboIdxs, tailInfo, fillUpInfo, 
                         polyInfoList, fillUpIdx, shift, rot):
    
    transList = np.zeros([len(riboIdxs), 30])
    tomoNames = [ ]
    i = 0
    for riboIdx in riboIdxs:      
        tailRibo = tailInfo[tailInfo[:,0] == riboIdx][0] #1D array data 
        fuRibo = fillUpInfo[fillUpInfo[:,0] == riboIdx][0] #1D array data 
        transList[i,1] = riboIdx
        transList[i,2] = fillUpIdx[riboIdx]
        transList[i,3] = polyInfoList[polyInfoList['pairIDX'] == riboIdx]['pairTomoID'].values[0]
        transList[i,4:7] = shift
        transList[i,7:10] = rot
        transList[i,10:13] = -1
        transList[i,13:16] = -1
        transList[i,16:18] = -1
        transList[i,18:21] = tailRibo[2:5]
        transList[i,21:24] = tailRibo[5:]
        transList[i,24:27] = fuRibo[2:5]
        transList[i,27:30] = fuRibo[5:]      
        i+=1
        tomoNames.append(polyInfoList[polyInfoList['pairIDX'] == riboIdx]['pairTomoName'].values[0])
    return transList, tomoNames 


def updateTransOfMiddleFillupRibos(fillUpMiddleRiboInfos,fillRiboCoords,fillRiboAngles, 
                                keepTailIdxForFillUpRibo, particleN, shift, rot,
                                polyInfoList, idxOfFillUpRibos):
   
    keepIdx = np.where(fillUpMiddleRiboInfos[:,0] == keepTailIdxForFillUpRibo[:,None])[-1]
    tomoNames = []
    fillUpMiddleRiboIdx = { } #this dict store the idx of each fillupmiddle ribo,but only one ribo for each poly
    fillUpMiddleRiboKeep = fillUpMiddleRiboInfos[keepIdx,:] #only keep fillupmiddle ribos of successfully filup polys
    #generate tranList
    transListFillupMiddle = np.array([]).reshape(-1, 30)
    for i, riboIdx in enumerate(keepTailIdxForFillUpRibo):
        tomoId = polyInfoList[polyInfoList['pairIDX'] == riboIdx]['pairTomoID'].values[0]
        tomoName = polyInfoList[polyInfoList['pairIDX'] == riboIdx]['pairTomoName'].values[0]    
        begin = 0
        fillUpMiddleRiboIdx[riboIdx] = particleN        
        fillUpMiddleRibosPerPoly = fillUpMiddleRiboKeep[fillUpMiddleRiboKeep[:,0] == riboIdx]
        transList_singlePoly = np.zeros([fillUpMiddleRibosPerPoly.shape[0],30])     
        for ii in range(fillUpMiddleRibosPerPoly.shape[0] - 1): #this cycle can be replaced,but for looks okay
            transList_singlePoly[ii,1] = particleN + begin #idx of thie ribo
            transList_singlePoly[ii,2] = particleN + begin + 1 #idx of next ribo
            transList_singlePoly[ii,3] = tomoId
            transList_singlePoly[ii,4:7] = shift
            transList_singlePoly[ii,7:10] = rot
            transList_singlePoly[ii,10:13] = -1;transList_singlePoly[ii,13:16] = -1
            transList_singlePoly[ii,16:18] = -1
            transList_singlePoly[ii,18:21] = fillUpMiddleRibosPerPoly[ii,2:6] #pos
            transList_singlePoly[ii,21:24] = fillUpMiddleRibosPerPoly[ii,5:] #angle 
            transList_singlePoly[ii,24:27] = fillUpMiddleRibosPerPoly[ii+1,2:6] #pos 
            transList_singlePoly[ii,27:30] = fillUpMiddleRibosPerPoly[ii+1,5:]#angle 
            begin += 1
            tomoNames.append(tomoName)
        #fillup the final row: from middle filled up ribosome => filled up ribosome     
        transList_singlePoly[-1,1:4] =  particleN + begin, idxOfFillUpRibos[riboIdx],tomoId
        transList_singlePoly[-1,4:7] = shift
        transList_singlePoly[-1,7:10] = rot
        transList_singlePoly[-1,10:13] = -1;transList_singlePoly[-1,13:16] = -1
        transList_singlePoly[-1,16:18] = -1
        transList_singlePoly[-1,18:21] = fillUpMiddleRibosPerPoly[-1,2:5]
        transList_singlePoly[-1,21:24] = fillUpMiddleRibosPerPoly[-1,5:]
        transList_singlePoly[-1,24:27] = fillRiboCoords[i,:]
        transList_singlePoly[-1,27:30] = fillRiboAngles[i,:]
        
        transListFillupMiddle = np.concatenate((transListFillupMiddle, transList_singlePoly), 
                                               axis = 0)
        particleN += (begin+1)
        tomoNames.append(tomoName)
    return transListFillupMiddle, tomoNames, fillUpMiddleRiboIdx



def updateTransList(transList, tomoName12, particleStar, pixelSize, starType, 
                    maxDist, oriPartList, pairClass):
    idx1 = transList[:, 1].astype(np.int)
    idx2 = transList[:, 2].astype(np.int)
    header = { }
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

    if (starType == 'relion2') | (starType == 'relion3'):
        classesPart1 = particleStar['rlnClassNumber'].values[idx1]
        classesPart2 = particleStar['rlnClassNumber'].values[idx2]
        psfsPart1 = particleStar['rlnCtfImage'].values[idx1]
        psfsPart2 = particleStar['rlnCtfImage'].values[idx2]
    elif starType == 'stopgap':
        classesPart1 = particleStar['class'].values[idx1]
        classesPart2 = particleStar['class'].values[idx2]
        psfsPart1 =  ['-1']*len(idx1)
        psfsPart2 =  ['-1']*len(idx2)
        
    neighPMPart = np.tile(['-1:-1','-1:-1'],(transList.shape[0],1))
    posInPolyPart = np.repeat(-1,transList.shape[0])
    
    #make the final startSt dict, which is differen with st dict
    #transform the array into dataframe
    startSt_data = pd.DataFrame(idxTmp, columns = header["fieldNames"][0:2])
    startSt_data[header["fieldNames"][2]] = transList[:,3].astype(np.int)
    transform_data = pd.DataFrame(transList[:,4:24],columns = header["fieldNames"][3:23])
    startSt_data = pd.concat([startSt_data,transform_data],axis = 1)
    startSt_data[header["fieldNames"][23]] = classesPart1
    startSt_data[header["fieldNames"][24]] = psfsPart1
    startSt_data[header["fieldNames"][25:27]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][27]] = posInPolyPart
    startSt_data[header["fieldNames"][28:34]] = pd.DataFrame(transList[:,24:30])
    startSt_data[header["fieldNames"][34]] = classesPart2
    startSt_data[header["fieldNames"][35]] = psfsPart2
    startSt_data[header["fieldNames"][36:38]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][38]] = posInPolyPart
    startSt_data[header["fieldNames"][39]] = tomoName12
    startSt_data[header["fieldNames"][40]] = np.repeat([pixelSize], transList.shape[0])
    startSt_data[header["fieldNames"][41]] = np.repeat([oriPartList],transList.shape[0])
    startSt_data[header["fieldNames"][42]] = np.repeat([maxDist],transList.shape[0])
    startSt_data[header["fieldNames"][43]] = pairClass
    startSt_data[header["fieldNames"][44]]  = np.repeat(['100.00-100.00-100.00'],transList.shape[0]) #100&100&100 --> represents added Ribos
    startSt_data[header["fieldNames"][45:47]] = pd.DataFrame(np.tile([-1,-1],(transList.shape[0],1)))
    
    return startSt_data
    

def updateParticle(riboCoords, riboAngles, exampleInfo, tomoNames, particleN, starType): 
    #deal with the colname of dataframe 
    colNames =  list(exampleInfo.index)
    if (starType == 'relion2')  |  (starType == 'relion3'):
        processedColNames = ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ',
                             'rlnAngleRot',   'rlnAngleTilt','rlnAnglePsi',
                             'rlnMicrographName','rlnImageName','rlnCtfImage']
        toProcessColNames = [ ]
        particleStruct = { } #store the infotmation of filled up ribosomes
        for single_name in colNames:
            particleStruct[single_name] = [ ]
        
        particleStruct['rlnCoordinateX'] = riboCoords[:,0]
        particleStruct['rlnCoordinateY'] = riboCoords[:,1]
        particleStruct['rlnCoordinateZ'] = riboCoords[:,2]
        
        count = 0
        for i in range(riboCoords.shape[0]):
            _,angles = tom_eulerconvert_xmipp(riboAngles[i,0], riboAngles[i,1], riboAngles[i,2], 'tom2xmipp')
            particleStruct['rlnAngleRot'].append(angles[0])
            particleStruct['rlnAngleTilt'].append(angles[1])
            particleStruct['rlnAnglePsi'].append(angles[2])
            particleStruct['rlnImageName'].append('notImplemented.mrc')
            particleStruct['rlnCtfImage'].append('notImplemented.mrc')
            if 'rlnOriginX' in colNames:
                particleStruct['rlnOriginX'].append(0)
                particleStruct['rlnOriginY'].append(0)
                particleStruct['rlnOriginZ'].append(0)
                if count == 0:
                    processedColNames.append('rlnOriginX')
                    processedColNames.append('rlnOriginY')
                    processedColNames.append('rlnOriginZ')
                    
            if 'rlnClassNumber' in colNames:
                particleStruct['rlnClassNumber'].append(-100) #class == -100 ==>represents added Ribos
                if count == 0:
                    processedColNames.append('rlnClassNumber')
            if 'rlnGroupNumber' in colNames:
                particleStruct['rlnGroupNumber'].append(-1) #groupNumber == -1 ==> represents added Ribos
                if count == 0:
                    processedColNames.append('rlnGroupNumber')
            count += 1
            #pick remaing colName
            if len(toProcessColNames) == 0:
                toProcessColNames = [i for i in colNames if i not in processedColNames]
            for singleName in toProcessColNames:
                particleStruct[singleName].append(exampleInfo[singleName])
        particleStruct['rlnMicrographName'] = tomoNames  
        
    elif starType == 'stopgap':
        processedColNames = ['orig_x','orig_y','orig_z','phi','psi','the']
        toProcessColNames = [ ]
        particleStruct = { } #store the infotmation of filled up ribosomes
        for single_name in colNames:
            particleStruct[single_name] = []
        
        particleStruct['orig_x'] = riboCoords[:,0]
        particleStruct['orig_y'] = riboCoords[:,1]
        particleStruct['orig_z'] = riboCoords[:,2]
        
        particleStruct['phi'] = riboAngles[:,0]
        particleStruct['psi'] = riboAngles[:,1]
        particleStruct['the'] = riboAngles[:,2]
        
        count = 0
        for i in range(riboCoords.shape[0]):
            
            if 'x_shift' in colNames:
                particleStruct['x_shift'].append(0)
                particleStruct['y_shift'].append(0)
                particleStruct['z_shift'].append(0)
                if count == 0:
                    processedColNames.append('x_shift')
                    processedColNames.append('y_shift')
                    processedColNames.append('z_shift')
                    
            if 'class' in colNames:
                particleStruct['class'].append(-100) #class == -100 ==>represents added Ribos
                if count == 0:
                    processedColNames.append('class')
                    
            if 'subtomo_num' in colNames:
                particleStruct['subtomo_num'].append(-1)
                if count == 0:
                    processedColNames.append('subtomo_num')
                    
            if 'tomo_num' in colNames:
                particleStruct['tomo_num'].append(-1)
                if count == 0:
                    processedColNames.append('tomo_num')  
                    
            if 'halfset' in colNames:
                particleStruct['halfset'].append(-1)
                if count == 0:
                    processedColNames.append('halfset') 
                    
            count += 1
            #pick remaing colName
            if len(toProcessColNames) == 0:
                toProcessColNames = [i for i in colNames if i not in processedColNames]
            for singleName in toProcessColNames:
                particleStruct[singleName].append(exampleInfo[singleName])  
        
    #make a dataframe    
    appendRiboInfo = pd.DataFrame(particleStruct)
    
    return appendRiboInfo


def debug_output(transList_filter, dists):
    print('Sucessfully fill up these ribosomes:')
    print('euler angle:Rot      Tilt       Psi, position:X          Y          Z ')
    for i in range(transList_filter.shape[0]):
        _,angles = tom_eulerconvert_xmipp(transList_filter[i,21], transList_filter[i,22], 
                                          transList_filter[i,23], 'tom2xmipp')
        print('\t   %.3f   %.3f   %.3f      %.3f   %.3f   %.3f'%(angles[0],angles[1],angles[2], 
                                                          transList_filter[i,18],
                                                          transList_filter[i,19],
                                                          transList_filter[i,20]))
    print('trans angle:Phi    Psi      Theta, trans shift:X         Y          Z')
    for i in range(transList_filter.shape[0]):
        print('\t   %.3f   %.3f   %.3f      %.3f   %.3f   %.3f'%( transList_filter[i,7],
                                                          transList_filter[i,8],
                                                          transList_filter[i,9], 
                                                          transList_filter[i,4],
                                                          transList_filter[i,5],
                                                          transList_filter[i,6]))   
    print('the filled up ribosomes have distance with the average transform:')
    print(dists)
       