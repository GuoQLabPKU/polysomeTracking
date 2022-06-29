import numpy as np
import os
import pandas as pd
import multiprocessing as mp
import shutil
import gc
from alive_progress import alive_bar 

from nemotoc.py_io.tom_starread import generateStarInfos
from nemotoc.py_io.tom_extractData import tom_extractData
from nemotoc.py_io.tom_starwrite import tom_starwrite
from nemotoc.py_transform.tom_calcPairTransForm import tom_calcPairTransForm
from nemotoc.py_log.tom_logger import Log


def tom_calcTransforms(posAng, pixS, maxDist, tomoNames='', dmetric='exact', outputName='', verbose=1, worker_n = 1):
    '''
    TOM_CALCTRANSFORMS calcutes pair transformations for all combinations use
    pruneRad(maxDist) to reduce calc (particles with distance > maxDist will not paired)

    transList=tom_calcTransforms(pos,pruneRad,dmetric,verbose)

    PARAMETERS

    INPUT
        posAng              positions and angles of particles/observation nx6 or List .star               
        maxDist             (inf) max particle/observation distance 
                                  pairs above this distance will be discarded
                                  usually a good value is 2.5 particleRad 
        tomoNames           ('') tuple with tomogramName to seperate tomograms
        dmetric             ('exact') at the moment only exact implemented
        outputName          ('') name of pair output starFile 
        verbose             (1) verbose flag use 0 for no output and no progress display
        worker_n            (1) number of CPUs to process

    OUTPUT
        starSt              star dataframe  containing the transformations and input data information


    EXAMPLE
    
        datMat = tom_calcTransforms(np.array([0 0 0;1 1 1;2 2 2]),np.array([0 0 0; 0 0 10; 10 20 30]),10,verbose = 0);

    REFERENCES
    
    '''
    log = Log('CalcTransform').getlog()
    oriPartList = 'noListStarGiven'
    
    if isinstance(posAng, str):
        oriPartList = posAng
    if isinstance(posAng, dict):
        if "pairTransVectX" in posAng.keys():
            log.info(" Input files has transformation information! Skip transform.")
            return 
    # read the star file into a dict/st
    st = tom_extractData(posAng, pixS)
    uTomoId = np.unique(st["label"]["tomoID"])
    len_tomo = len(uTomoId) 

    transList = np.array([],dtype = np.float).reshape(0, 29)
    allTomoNames = st["label"]["tomoName"]
  
    if worker_n == 1:
        for i in range(len_tomo):
            log.info('Calculate transformations for tomo %s'%uTomoId[i])
            idx = list(np.where(st["label"]["tomoID"] == uTomoId[i])[0])
            idxAct = idx
            posAct = st["p1"]["positions"][idx,:]
            anglesAct = st["p1"]["angles"][idx,:]
            transListAct = calcTransforms(posAct, anglesAct, maxDist, dmetric, uTomoId[i], idxAct, verbose)
            transList = np.concatenate((transList, transListAct),axis=0)
            log.info('Transforms for tomo%d done'%(i))
    else:
        #make temp directory to store the transListAct and then reload merge them!
        temp_dir = "%s/tempTrans"%os.path.split(outputName)[0]
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
        else:
            os.mkdir(temp_dir)
                   
        npr = worker_n
        avail_cpu = mp.cpu_count()
        if worker_n == -1:
            npr = avail_cpu
            log.info("Use %d CPUs to calculate transformations"%npr)
        elif avail_cpu < worker_n:
            npr = avail_cpu
            log.warning('Not enough CPUs are available! Use %d CPUs instead.'%npr)
            
        #using parallel
        log.info('Use %d cpus to calculate transforms'%npr)
        processes = dict()
        spl_ids = np.array_split(uTomoId,npr) #one cpu/one tomogram
        #remove the empty spl_ids
        spl_ids = [i for i in spl_ids if len(i) > 0]
        
        for pr_id, spl_id in enumerate(spl_ids):
            pr = mp.Process(target = pr_worker, args=(pr_id, st, spl_id, maxDist, dmetric, temp_dir, verbose))
            pr.start()
            processes[pr_id] = pr
        for pr_id, pr in zip(processes.keys(), processes.values()):
            pr.join()
            if pr_id != pr.exitcode:
                errorInfo = 'The process %d ended usuccessfully:[%d]'%(pr_id, pr.exitcode)
                log.error(errorInfo)
                raise RuntimeError(errorInfo)
                
        gc.collect() #free the memory      
        for single_id in uTomoId:
            transListAct = np.load('%s/tomo%d_trans.npy'%(temp_dir, single_id))
            transList = np.concatenate((transList, transListAct),axis=0)
        #delete the temp files       
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        log.info('Calculate transformations done')
        
    if outputName == '':
        return transList
    else:
        startSt = genStarFile(transList, allTomoNames, st, maxDist, oriPartList, outputName)  
        return startSt 

def pr_worker(pr_id, st, tomo_ids, maxDist, dmetric, temp_dir,verbose):
    for i in tomo_ids:
        idx = list(np.where(st["label"]["tomoID"] == i)[0])
        idxAct = idx
        posAct = st["p1"]["positions"][idx,:]
        anglesAct = st["p1"]["angles"][idx,:]
        transListAct = calcTransforms(posAct, anglesAct, maxDist, dmetric, i, idxAct, verbose)
        np.save('%s/tomo%d_trans.npy'%(temp_dir, i), transListAct)
        print("Transforms for tomo%d done"%(i))
                                                                                                       
    os._exit(pr_id)
              
    
def calcTransforms(pos, angles, pruneRad, dmetric, tomoID, idxAct, verbose):
    #jobList: store three columns: 
    #1.the index of the pair1 in pos_array 2. the index of the pair2 in pos_array
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
        raise RuntimeError("The distances between neighbors are bigger than %d pixels! Set bigger maxDist!"%pruneRad)  
        
    transListAct_inner = np.zeros([jobList.shape[0],29], dtype = np.float) 
    with alive_bar(int(np.floor(jobList.shape[0]/100)+1), title="calculate transforms") as bar:
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
            if (verbose == 1) & (i%100 == 0):
                bar()
                bar.text("calculate transfroms done with %d pairs"%(i))
    return transListAct_inner
      
    
def genStarFile(transList, allTomoNames, st, maxDist, oriPartList, outputName):
    classes = st["p1"]["classes"] 
    psfs = st["p1"]["psfs"]
    pixs = st["p1"]["pixs"]
    #store the header information 
    header = { }
    header["fieldNames"] =       ['pairIDX1','pairIDX2','pairTomoID',
                                  'pairTransVectX','pairTransVectY','pairTransVectZ',
                                  'pairTransAngleZXZPhi','pairTransAngleZXZPsi','pairTransAngleZXZTheta',
                                  'pairInvTransVectX','pairInvTransVectY','pairInvTransVectZ',
                                  'pairInvTransAngleZXZPhi','pairInvTransAngleZXZPsi','pairInvTransAngleZXZTheta',
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
                                  'pairOriPartList','pairMaxDist','pairClass','pairClassColour',
                                  'pairLabel','pairScore']
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
    startSt_data[header["fieldNames"][25:27]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][27]] = posInPolyPart
    startSt_data[header["fieldNames"][28:34]] = pd.DataFrame(transList[:,23:29])
    startSt_data[header["fieldNames"][34]] = classesPart2
    startSt_data[header["fieldNames"][35]] = psfsPart2
    startSt_data[header["fieldNames"][36:38]] = pd.DataFrame(neighPMPart)
    startSt_data[header["fieldNames"][38]] = posInPolyPart
    startSt_data[header["fieldNames"][39]] = tomoName12
    startSt_data[header["fieldNames"][40]] = np.repeat([pixs[0]], transList.shape[0])
    startSt_data[header["fieldNames"][41]] = np.repeat([oriPartList],transList.shape[0])
    startSt_data[header["fieldNames"][42]] = np.repeat([maxDist],transList.shape[0])
    startSt_data[header["fieldNames"][43]] = np.repeat([-1],transList.shape[0])
    startSt_data[header["fieldNames"][44]]  = np.repeat(['0-0-0'],transList.shape[0])
    startSt_data[header["fieldNames"][45:47]] = pd.DataFrame(np.tile([-1,-1],(transList.shape[0],1)))
    
    #store the starSt 
    starInfos = generateStarInfos()
    starInfos['data_particles'] = startSt_data
    tom_starwrite(outputName, starInfos)      
    return startSt_data 
    