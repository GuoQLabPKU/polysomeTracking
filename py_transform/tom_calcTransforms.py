import numpy as np
import time
import os
import copy 
import pandas as pd
import multiprocessing as mp
import gc
from py_io.tom_starread import tom_starread
from py_io.tom_extractData import tom_extractData
from py_transform.tom_calcPairTransForm import tom_calcPairTransForm
from py_io.tom_starwrite import tom_starwrite


def tom_calcTransforms(posAng,  maxDist, tomoNames='', dmetric='exact', outputName='', verbose=1, worker_n = 1):
    '''
    TOM_CALCTRANSFORMS calcutes pair transformations for all combinations use
    pruneRad(maxDist) to reduce calc (particles with distance > maxDist will not paired)

    transList=tom_calcTransforms(pos,pruneRad,dmetric,verbose)

    PARAMETERS

    INPUT
        posAng              positions and angles of particles/observation nx6 or List .star               
        maxDist             (inf) max particle/observation distance 
                                  pairs above this distance will be discarted
                                  usually a good value is 2.5 particleRad 
        tomoNames           ('') tuple with tomogramName to seperate tomograms
        dmetric             ('exact') at the moment only exact implemented
        outputName          ('') name of pair output starFile 
        verbose             (1) verbose flag use 0 for no output and no progress display
        worker_n            (1) number of CPUs to process

    OUTPUT
        starSt              star dataframe  containing the transformations and input Data


    EXAMPLE
    
        datMat = tom_calcTransforms(np.array([0 0 0;1 1 1;2 2 2]),np.array([0 0 0; 0 0 10; 10 20 30]),10,verbose = 0);

    REFERENCES
    
    '''
    oriPartList = 'noListStarGiven'
    type_posAng = type(posAng)
    
    if type_posAng.__name__ == 'str':
        oriPartList = posAng
        posAng = tom_starread(posAng)
    if type_posAng.__name__ == 'dict':
        if "pairTransVectX" in posAng.keys():
            print("The file already has transformation information! No need to transform.")
            return 
    # read the star file into a dict/st
    st = tom_extractData(posAng,0)
    uTomoId = np.unique(st["label"]["tomoID"])
    #uTomoNames = np.unique(st["label"]["tomoName"])
    len_tomo = len(uTomoId) #how many tomo in this starfile
    #transDict: store the information of transformation
    #allTomoNames: the name of tomo
    #idxOffset: 
    transList = np.array([],dtype = np.float).reshape(0, 29)
    allTomoNames = st["label"]["tomoName"]
    #idxOffSet = 0

    for i in range(len_tomo):
        time1 = time.time()
        print("####################################################")
        print("Calculating transformations for tomo %d.........."%i)
        idx = list(np.where(st["label"]["tomoID"] == uTomoId[i])[0])
        idxAct = copy.deepcopy(idx)
        posAct = st["p1"]["positions"][idx,:]
        anglesAct = st["p1"]["angles"][idx,:]
        if worker_n == 1:
            transListAct = calcTransforms_linear(posAct, anglesAct, maxDist, dmetric, uTomoId[i], idxAct, verbose)
        else:
            transListAct = calcTransforms(posAct, anglesAct, maxDist, dmetric, uTomoId[i], idxAct, verbose,  worker_n)     
        #transListAct[:,0:2] = transListAct[:,0:2] + idxOffSet
        transList = np.concatenate((transList, transListAct),axis=0)
        #idxOffSet = idxOffSet + posAct.shape[0]
        time2 = time.time()
        time_gap = (time2-time1)
        print("Finish calculating transformations for tomo %d with %d pairs, %.5f seconds consumed."%(i,transListAct.shape[0],
                                                                                                      time_gap))
    
    if outputName == '':
        return transList
    else:
        startSt = genStarFile(transList, allTomoNames, st, maxDist, oriPartList, outputName)  
        return startSt 

def calcTransforms_linear(pos, angles, pruneRad, dmetric, tomoID, idxAct, verbose):
    #jobList: store three columns: 1.the index of the pair1 in pos_array 2. the index of the pair2 in pos_array
    jobList = np.zeros([pos.shape[0]*pos.shape[0], 2], dtype = np.int)
    zz = 0
    for i in range(pos.shape[0]):
        for ii in range(i+1, pos.shape[0]):
            pos1 = pos[i,:]
            pos2 = pos[ii,:]
            if (np.linalg.norm(pos1 - pos2)) > pruneRad: #discard the pairs with euler distance > pruneRad
                continue
            jobList[zz,0] = i
            jobList[zz,1] = ii
            zz += 1        
    jobListN = jobList[0:zz,:]
    jobList = jobListN
    
    if jobList.shape[0] == 0:
        print("Error: the distances between ribosomes are bigger than %d pixels! Set bigger maxDist and try again!"%pruneRad)
        os._exit(1)    
    transListAct_inner = np.zeros([jobList.shape[0],29], dtype = np.float)#this will store the transformation results   
    for i in range(jobList.shape[0]):             
        icmb0,icmb1 = jobList[i,:]
        pos1 = pos[icmb0,:]
        pos2 = pos[icmb1,:]
        ang1 = angles[icmb0,:]
        ang2 = angles[icmb1,:]
        posTr1, angTr1, lenPosTr1, lenAngTr1 = tom_calcPairTransForm(pos1,ang1,pos2,ang2,dmetric)
        posTr2, angTr2, _, _ = tom_calcPairTransForm(pos2,ang2,pos1,ang1,dmetric)
        transListAct_inner[i,:] = np.array([idxAct[icmb0], idxAct[icmb1],tomoID,
                                             posTr1[0], posTr1[1], posTr1[2], angTr1[0], angTr1[1], angTr1[2],                                        
                                             posTr2[0], posTr2[1], posTr2[2], angTr2[0], angTr2[1], angTr2[2],
                                             lenPosTr1, lenAngTr1,
                                             pos1[0],pos1[1],pos1[2],ang1[0],ang1[1],ang1[2],
                                             pos2[0],pos2[1],pos2[2],ang2[0],ang2[1],ang2[2]])
        if verbose == 1:
            if i%50 == 0:
                print("Calculating transfromation for tomo %d with %d pairs..........."%(tomoID,i))
                
    return transListAct_inner
      
def calcTransforms(pos, angles, pruneRad, dmetric, tomoID, idxAct, verbose, worker_n):
    #jobList: store three columns: 1.the index of the pair1 in pos_array 2. the index of the pair2 in pos_array
    jobList = np.zeros([pos.shape[0]*pos.shape[0], 2], dtype = np.int)
    zz = 0
    
    for i in range(pos.shape[0]):
        for ii in range(i+1, pos.shape[0]):
            pos1 = pos[i,:]
            pos2 = pos[ii,:]
            if (np.linalg.norm(pos1 - pos2)) > pruneRad: #discard the pairs with euler distance > pruneRad
                continue
            jobList[zz,0] = i
            jobList[zz,1] = ii
            zz += 1        
    jobListN = jobList[0:zz,:]
    jobList = jobListN
    
    if jobList.shape[0] == 0:
        print("Error: the distances between ribosomes are bigger than %d pixels! Set bigger maxDist and try again!"%pruneRad)
        os._exit(1)    
    transListAct_inner = np.zeros([jobList.shape[0],29], dtype = np.float)#this will store the transformation results   
    transListAct_inner[:,0] = np.array([ idxAct[i] for i in jobList[:,0]])
    transListAct_inner[:,1] = np.array([ idxAct[i] for i in jobList[:,1]])
    transListAct_inner[:,2] = np.repeat([tomoID], jobList.shape[0])
    
    npr = worker_n
    avail_cpu = mp.cpu_count()
    if worker_n == -1:
        npr = avail_cpu
        print("Use %d CPUs."%npr)
    elif avail_cpu < worker_n:
        npr = avail_cpu
        print("Warning: No enough CPUs are available! Use %d CPUs instead."%npr)
    processes = [ ]
    #create the list on indices to split
    npart = jobList.shape[0]
    sym_ids = np.arange(npart)
    spl_ids = np.array_split(range(len(sym_ids)),npr)
    shared_ncc = mp.Array('f', npart*26)
    for pr_id in range(npr):
        pr = mp.Process(target = tom_calcPairTransForm_parallel,
                        args = (pr_id, spl_ids[pr_id],jobList, pos, angles, dmetric, shared_ncc))
        pr.start()
        processes.append(pr)
    pr_results = [ ]
    for pr in processes:
        pr.join()
        pr_results.append(pr.exitcode)
    #check the exit stats
    for pr_id in range(len(processes)):
        if pr_id != pr_results[pr_id]:
            print("Error: process %d exited unexpectedly."%pr_id)
            
    return_results = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(npart,26)
    transListAct_inner[:,3:] = return_results
    gc.collect()
    
    return transListAct_inner
        


def tom_calcPairTransForm_parallel(pr_id,rows_id, jobList, pos, angles, dmetric, shared_ncc):

    for single_row in rows_id:             
        icmb0,icmb1 = jobList[single_row,:]
        pos1 = pos[icmb0,:]
        pos2 = pos[icmb1,:]
        ang1 = angles[icmb0,:]
        ang2 = angles[icmb1,:]
        posTr1, angTr1, lenPosTr1, lenAngTr1 = tom_calcPairTransForm(pos1,ang1,pos2,ang2,dmetric)
        posTr2, angTr2, _, _ = tom_calcPairTransForm(pos2,ang2,pos1,ang1,dmetric)
        if single_row == 0:
            print(single_row)
            shared_ncc[0:26] = np.array([posTr1[0], posTr1[1], posTr1[2], angTr1[0], angTr1[1], angTr1[2],                                        
                                     posTr2[0], posTr2[1], posTr2[2], angTr2[0], angTr2[1], angTr2[2],
                                     lenPosTr1, lenAngTr1,
                                     pos1[0],pos1[1],pos1[2],ang1[0],ang1[1],ang1[2],
                                     pos2[0],pos2[1],pos2[2],ang2[0],ang2[1],ang2[2]]  )
        else:
            shared_ncc[single_row*26:single_row*26+26] = np.array([posTr1[0], posTr1[1], posTr1[2], angTr1[0], angTr1[1], angTr1[2],                                        
                                     posTr2[0], posTr2[1], posTr2[2], angTr2[0], angTr2[1], angTr2[2],
                                     lenPosTr1, lenAngTr1,
                                     pos1[0],pos1[1],pos1[2],ang1[0],ang1[1],ang1[2],
                                     pos2[0],pos2[1],pos2[2],ang2[0],ang2[1],ang2[2]])
           
    
    os._exit(pr_id)   
    
def genStarFile(transList, allTomoNames, st, maxDist, oriPartList, outputName):
    classes = copy.deepcopy(st["p1"]["classes"]) #make sure the newborn transList doesn't impact the st 
    psfs = copy.deepcopy(st["p1"]["psfs"])
    pixs = copy.deepcopy(st["p1"]["pixs"])
    #store the header information 
    header = { }
    header["is_loop"] = 1
    header["title"] = "data_"
    header["fieldNames"] = ['_pairIDX1','_pairIDX2','_pairTomoID',
                                  '_pairTransVectX','_pairTransVectY','_pairTransVectZ',
                                  '_pairTransAngleZXZPhi', '_pairTransAngleZXZPsi','_pairTransAngleZXZTheta',
                                  '_pairInvTransVectX','_pairInvTransVectY','_pairInvTransVectZ',
                                  '_pairInvTransAngleZXZPhi', '_pairInvTransAngleZXZPsi','_pairInvTransAngleZXZTheta',
                                  '_pairLenTrans','_pairAngDist',
                                  '_pairCoordinateX1','_pairCoordinateY1','_pairCoordinateZ1',
                                  '_pairAnglePhi1','_pairAnglePsi1','_pairAngleTheta1',
                                  '_pairClass1','_pairPsf1',
                                   '_pairNeighPlus1','_pairNeighMinus1', '_pairPosInPoly1',
                                  '_pairCoordinateX2','_pairCoordinateY2','_pairCoordinateZ2',
                                  '_pairAnglePhi2','_pairAnglePsi2','_pairAngleTheta2',
                                  '_pairClass2','_pairPsf2',
                                   '_pairNeighPlus2','_pairNeighMinus2', '_pairPosInPoly2',
                                  '_pairTomoName','_pairPixelSizeAng',
                                  '_pairOriPartList',
                                  '_pairMaxDist','_pairClass','_pairClassColour','_pairLabel','_pairScore']
    idxTmp = transList[:,0:2].astype(np.int)
    idxTmp1 = list(idxTmp[:,0].reshape(1,-1)[0])
    idxTmp2 = list(idxTmp[:,1].reshape(1,-1)[0])
    classesPart1 = classes[idxTmp1]
    classesPart2 = classes[idxTmp2]
    psfsPart1 = psfs[idxTmp1]
    psfsPart2 = psfs[idxTmp2]
    neighPMPart = np.tile(['-1:-1','-1:-1'],(transList.shape[0],1))
    posInPolyPart = np.repeat(-1,transList.shape[0])
    
    tomoName12 = allTomoNames[idxTmp1]
    #make the final startSt dict, which is differen with st dict
    #transform the array into dataframe
    startSt_data = pd.DataFrame(idxTmp, columns = header["fieldNames"][0:2])
    startSt_data[header["fieldNames"][2]] = transList[:,2].astype(np.int)
    transform_data = pd.DataFrame(transList[:,3:23],columns = header["fieldNames"][3:23])
    startSt_data = pd.concat([startSt_data,transform_data],axis = 1)
    startSt_data[header["fieldNames"][23]] = classesPart1
    startSt_data[header["fieldNames"][24]] = psfsPart1
    startSt_data[header["fieldNames"][25:27]] = neighPMPart
    startSt_data[header["fieldNames"][27]] = posInPolyPart
    startSt_data[header["fieldNames"][28:34]] = transList[:,23:29]
    startSt_data[header["fieldNames"][34]] = classesPart2
    startSt_data[header["fieldNames"][35]] = psfsPart2
    startSt_data[header["fieldNames"][36:38]] = neighPMPart
    startSt_data[header["fieldNames"][38]] = posInPolyPart
    startSt_data[header["fieldNames"][39]] = tomoName12
    startSt_data[header["fieldNames"][40]] = np.repeat([pixs[0]], transList.shape[0])
    startSt_data[header["fieldNames"][41]] = np.repeat([oriPartList],transList.shape[0])
    startSt_data[header["fieldNames"][42]] = np.repeat([maxDist],transList.shape[0])
    startSt_data[header["fieldNames"][43]] = np.repeat([-1],transList.shape[0])
    startSt_data[header["fieldNames"][44]]  = np.repeat(['0-0-0'],transList.shape[0])
    startSt_data[header["fieldNames"][45:47]] = np.tile([-1,-1],(transList.shape[0],1))
    
    #store the starSt 
    tom_starwrite(outputName, startSt_data, header) ###header should begin with "_"
    #load the saved starfile
    startSt = tom_starread(outputName)
   
    return startSt 
    


if __name__ == '__main__':
    startSt  = tom_calcTransforms('./simOrderRandomized.star', 100, tomoNames='', 
                                      dmetric='exact', outputName='allTransforms.star', verbose=1, worker_n = 2)
    #tom_starread("ILV")
    



    
                 
    
    
    
    
    
    
    
    
    
    
       
        
        
        
            
            
        
        