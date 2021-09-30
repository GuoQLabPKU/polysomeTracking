import cupy as cp
import numpy as np
import os
import gc
import shutil
import multiprocessing as mp
from alive_progress import alive_bar 
import random

from py_cluster.tom_calc_packages import tom_calc_packages
from py_transform.tom_sum_rotation import tom_sum_rotation
from py_log.tom_logger import Log

#@profile
def tom_pdist(in_Fw, maxChunk ,worker_n = 1, gpu_list = None,
              dmetric = 'euc', in_Inv = '', jobListSt = None, 
              tmpDir = '',  cleanTmpDir = 1, verbose = 1):
    '''
    dists=tom_pdist(in_Fw,dmetric,in_Inv,maxChunk)

    PARAMETERS

    INPUT
       in_Fw                inputdata nxdimension for angles OR distance nx3 in zxz 
       dmetric          ('euc') for euclidean distance metric or  
                            'ang'   for angles
       in_Inv           ('')  inverse data neede for needed for transformations   
       makeJob          if make joblist (1,make; 0 no. If 0 is input, must provide jobList)
       maxChunk         max chunk size  #shoud modity accoring to the cpus/gpus & memory
       worker_n        # of cpus (not used in this function)
       gpu_list        gpus (not used in this function)
       

    OUTPUT
       dists             distances in the same order as pdist from matlab (single array)

    EXAMPLE
  

    dd=tom_pdist(np.array([[0, 0, 0],[0, 0, 10], [10, 20, 30]]),'ang');
    '''   
    randN = random.randint(0,500)
    log = Log('transform pairs distance %d'%randN).getlog()
    
    in_Fw = in_Fw.astype(np.single)
    if len(in_Inv) > 0:
        log.debug('Use inverse transforms')
        #print("Using inverse transforms")
        in_Inv = in_Inv.astype(np.single)
     #since the number of combination of pairs can be large 
    if jobListSt is None:
        lenJobs = np.uint64(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2)
        tmpDir = 'tmpPdistgpu'
        jobListSt = genJobList(in_Fw.shape[0], tmpDir, maxChunk) #jobList store each dict for each node
    else:
        lenJobs = np.uint64(in_Fw.shape[0] - 1)
       
    dists = np.zeros(lenJobs, dtype = np.single) # the distance between pairs of ribosomes , one dimention array
    log.info('Calculate %s for %d transforms'%(dmetric, in_Fw.shape[0]))
    #print("Start calculating %s for %d transforms"%(dmetric, in_Fw.shape[0]))
    job_n = len(jobListSt) #number of cores to use
    if dmetric == 'euc':
        log.debug("Load data into %d gpus..."%job_n)
        #print("Loading data into %d gpus..."%job_n)            
        shared_ncc = mp.Array('f', int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2))            
        processes = [ ]
        for pr_id, gpu_id in enumerate(jobListSt.keys()):
            jobList = jobListSt[gpu_id]
            pr = mp.Process(target = calcVectDist_mp,
                    args = (pr_id, jobList, in_Fw, in_Inv, shared_ncc,gpu_id))
            
            pr.start()
            processes.append(pr)
        pr_results = [ ]
        for pr in processes:
            pr.join()
            pr_results.append(pr.exitcode)
        #check the exit stats
        for pr_id in range(len(processes)):
            if pr_id != pr_results[pr_id]:
                errorInfo = "Process %d exited unexpectedly."%pr_id
                log.error(errorInfo)
                raise RuntimeError(errorInfo)
        dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
        dists = dists[0]
        gc.collect()
            
        log.info('Calculate euc transforms distance done!')    
        #print("Finishing calculating euc transforms distance!")      
      
             
    elif dmetric == 'ang':
        #calculate ration matrix
        Rin= calcRotMatrices(in_Fw, verbose)
        if len(in_Inv) > 0:
            Rin_Inv= calcRotMatrices(in_Inv, verbose)        
        else:
            Rin_Inv = ''
        
        log.debug("Load data into %d gpus..."%job_n)
        #print("Loading data into %d gpus..."%job_n)
        shared_ncc = mp.Array('f', int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2))   
        processes = [ ]
        for pr_id, gpu_id in enumerate(jobListSt.keys()):
            jobList = jobListSt[gpu_id]
            pr = mp.Process(target = calcAngDist_mp,
                            args = (pr_id, jobList, Rin, Rin_Inv,shared_ncc, gpu_id))        
            pr.start()
            processes.append(pr)
        pr_results = [ ]
        for pr in processes:
            pr.join()
            pr_results.append(pr.exitcode)
        #check the exit stats
        for pr_id in range(len(processes)):
            if pr_id != pr_results[pr_id]:
                errorInfo = "Process %d exited unexpectedly."%pr_id
                log.error(errorInfo)
                raise RuntimeError(errorInfo)

        dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
        dists = dists[0]
        gc.collect()       

        log.info('Calculate ang transforms distance done!')                        
        #print("Finishing calculating ang transforms distance!")  
        
    if  cleanTmpDir == 1:
        shutil.rmtree(tmpDir) #remove the dirs 
    return dists  # one dimension array           
            
#@profile  
def calcVectDist_mp(pr_id, jobList, in_Fw, in_Inv, shared_ncc, gpu_id):
    cp.cuda.Device(gpu_id).use()
    in_Fw = cp.asarray(in_Fw) #move array into different GPUs
    if len(in_Inv) > 0:
        in_Inv = cp.asarray(in_Inv)
    with alive_bar(len(jobList), title="euc distances") as bar:  
        for jobList_single in jobList:
            jobListChunk = cp.load(jobList_single["file"],allow_pickle=True)
            g1 = in_Fw[jobListChunk[:,0],:]
            g2 = in_Fw[jobListChunk[:,1],:]
            if len(in_Inv)  == 0:
                g1Inv = ''
                g2Inv = ''
            else:
                g1Inv = in_Inv[jobListChunk[:,0],:]
                g2Inv = in_Inv[jobListChunk[:,1],:]
            dtmp = calcVectDist(g1,g2,g1Inv,g2Inv)
            dtmp = cp.asnumpy(dtmp)        
            shared_ncc[jobList_single["start"]:jobList_single["stop"]] = dtmp
            del jobListChunk, g1, g2, g1Inv, g2Inv, dtmp
            gc.collect()
            bar()
        
    cp.get_default_memory_pool().free_all_blocks()   #free the blocked memory 
    cp.get_default_pinned_memory_pool().free_all_blocks() #free the blocked memory

    os._exit(pr_id)

#@profile    
def calcVectDist(g1,g2,g1Inv,g2Inv):
    dv = g2-g1
    dtmp =  cp.linalg.norm(dv, axis = 1)
    if len(g1Inv) > 0:
        dv = g2-g1Inv
        distsInv = cp.linalg.norm(dv, axis = 1)
        
        dv = g1-g2Inv
        distsInv2 = cp.linalg.norm(dv, axis = 1)
        
        dv = g1Inv - g2Inv
        distsInv3 = cp.linalg.norm(dv, axis = 1)
               
        dists_allpart2 = cp.array([dtmp, distsInv, distsInv2, distsInv3])
        dtmp = cp.min(dists_allpart2, axis = 0)
    return dtmp
 
def calcRotMatrices(in_angs, verbose):
    if verbose:
        print("Start calculating rotation matrices for each transform")
    Rin = np.zeros([in_angs.shape[0], 3,6 ], dtype = np.single)
    
    for i in range(in_angs.shape[0]):
        _,_, Rin[i,:,0:3] = tom_sum_rotation(in_angs[i,:], np.array([0,0,0]))
        Rin[i,:,3:6] = np.linalg.inv(Rin[i,:,0:3])
    
    if verbose:    
        print("Calculate rotation matrices for each transform done")
    
    
    return  Rin
    
#@profile   
def calcAngDist_mp(pr_id, jobList, Rin, Rin_Inv,shared_ncc,gpu_id):
    cp.cuda.Device(gpu_id).use()
    Rin = cp.asarray(Rin) #move array into different GPUs
    if len(Rin_Inv) > 0:
        Rin_Inv = cp.asarray(Rin_Inv)   
    with alive_bar(len(jobList), title="ang distances") as bar:
        for singlejobs in jobList: 
            jobListChunk = cp.load(singlejobs["file"])
            #Rs = Rin[jobListChunk[:,0],:,0:3]
            #RsInv = Rin[jobListChunk[:,1],:,3:6]           
            dtmp = calcAngDist(Rin[jobListChunk[:,0],:,0:3], Rin[jobListChunk[:,1],:,3:6])
            if len(Rin_Inv) > 0:
                #Rs_Inv = Rin_Inv[jobListChunk[:,0],:,0:3]
                #Rs_Inv_Inv = Rin_Inv[jobListChunk[:,1],:,3:6]
                dtmpInv = calcAngDist(Rin_Inv[jobListChunk[:,0],:,0:3], Rin[jobListChunk[:,1],:,3:6])             
                dtmpInv2 = calcAngDist(Rin[jobListChunk[:,0],:,0:3], Rin_Inv[jobListChunk[:,1],:,3:6])
                dtmpInv3 = calcAngDist(Rin_Inv[jobListChunk[:,0],:,0:3], Rin_Inv[jobListChunk[:,1],:,3:6] )
    
                dists_all = cp.array([dtmp, dtmpInv, dtmpInv2, dtmpInv3])
                dtmp = cp.min(dists_all, axis = 0)
                del  dtmpInv, dtmpInv2, dtmpInv3, dists_all
                
            shared_ncc[singlejobs["start"]:singlejobs["stop"]] = dtmp  
            del jobListChunk, dtmp
            gc.collect() 
            bar()
                            
    cp.get_default_memory_pool().free_all_blocks()   #free the blocked memory 
    cp.get_default_pinned_memory_pool().free_all_blocks() #free the blocked memory
    os._exit(pr_id)        
    
#@profile    
def calcAngDist(Rs,RsInv):
    #multiple the two matrices       
    Rp = cp.matmul(Rs, RsInv)   
    tr_Rp = (cp.trace(Rp, axis1=1, axis2=2) - 1)/2 
    #calculate the angle distance       
    tr_Rp = cp.clip(tr_Rp, a_min = -1, a_max =1)
    dists = cp.arccos(tr_Rp)/cp.pi*180
    dists = (dists.real).astype(cp.single)
    return dists #one dimention arrsy float32)
    
    
    
def genJobList(szIn, tmpDir, maxChunk): #maxChunk is one dict 
    lenJobs = np.uint64(szIn*(szIn-1)/2)
    jobList = np.zeros([lenJobs,2], dtype = np.uint32) #expand the range of positive int save memory(no negative int)
    startA = 0  
    
    with alive_bar(int(np.floor(szIn/100)+1), title="jobList generation") as bar:
        for i in range(szIn):
            v2 = np.arange(i+1,szIn, dtype = np.uint32)
            v1 = np.repeat(i, len(v2)).astype(np.uint32)
            endA = startA+len(v2)
            jobList[startA:endA,0] = v1
            jobList[startA:endA,1] = v2
            startA = endA 
            if (i%100) == 0:
                bar()      
        
    #split the jobsList into different GPUs
    gpu_list, startSiteList, fileSizeList  = fileSplit(maxChunk, lenJobs)  
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir) #remove the .npy anyway
    os.mkdir(tmpDir)
    jobsListSt_dict = { }
    for gpu_id, startsite, filesize in zip(gpu_list, startSiteList, fileSizeList):
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
        
def fileSplit(maxChunk, lenJobs):
    gpulist = [ ]
    file_size = [ ]
    start_site = [ ]
    for key in maxChunk.keys():
        gpulist.append(key)
        file_size.append(maxChunk[key]) 
    sumF = np.sum(file_size)
    file_size = [np.uint64(i/sumF*lenJobs) for i in file_size]
    #give the start sites of lenJobs for each gpu
    start_site.append(np.uint64(0) )
    forward_site = np.uint64(0) 
    for file_len in file_size[:-1]:        
        site = forward_site + file_len
        start_site.append(site)
        forward_site = site
    file_size[-1] = lenJobs - start_site[-1]
    return gpulist, start_site, file_size    
        
    
def genjobsList_oneGPU(startsite, lenJobs, maxChunk):
    numOfPackages = int(np.ceil(lenJobs/maxChunk))       
    packages = tom_calc_packages(numOfPackages, lenJobs, startsite) #split the jobList into different size, the packages is one array
    return packages
  
    
    
        
    