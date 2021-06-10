import numpy as np
import os
import gc
import shutil
import multiprocessing as mp
from alive_progress import alive_bar 
from py_cluster.tom_calc_packages import tom_calc_packages
from py_transform.tom_sum_rotation import tom_sum_rotation

#from py_transform.tom_calcTransforms import tom_calcTransforms

#@profile
def tom_pdist(in_Fw, maxChunk ,worker_n, gpu_list,dmetric = 'euc', in_Inv = '', verbose=1 ):
    '''
    dists=tom_pdist(in_Fw,dmetric,in_Inv,maxChunk)

    PARAMETERS

    INPUT
       in_Fw                inputdata nxdimension for angles OR distance nx3 in zxz 
       dmetric        ('euc') for euclidean distance metric or  
                            'ang'   for angles
       in_Inv           ('')  inverse data neede for needed for transformations    
       maxChunk         max chunk size  #shoud modity accoring to the cpus/gpus & memory
       worker_n        # of cpus 
       gpu_list        gpus (not used in this function)
       verbose        (1) 0 for no output

    OUTPUT
       dists             distances in the same order as pdist from matlab (single array)

    EXAMPLE
  

    dd=tom_pdist(np.array([[0, 0, 0],[0, 0, 10], [10, 20, 30]]),'ang');
    '''
    #change into single 
 
    in_Fw = in_Fw.astype(np.single)
    if len(in_Inv) > 0:
        in_Inv = in_Inv.astype(np.single)
        print("Using inverse transforms")
   
    tmpDir = 'tmpPdistcpu' #since the number of combination of pairs can be large 
    lenJobs = np.uint64(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2)
    jobList = genJobList(in_Fw.shape[0], tmpDir, maxChunk) #jobList store each dict for each node
    dists = np.zeros(lenJobs, dtype = np.single) # the distance between pairs of ribosomes , one dimention array
    print("Start calculating %s for %d transforms"%(dmetric, in_Fw.shape[0]))
    job_n = len(jobList) #number of chunks of joblists to process
    if dmetric == 'euc':
        if (worker_n > 1) & (job_n > 1):
            if job_n < worker_n:
                worker_n = job_n               
            print("Start %d processors..."%worker_n)            
            shared_ncc = mp.Array('f', np.int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2)) #store the distance results for all cpus
            spl_ids = np.array_split(range(job_n),worker_n) #split the jobs into workers
            processes = [ ]
            for pr_id,  jobs_split in enumerate(spl_ids):            
                jobList_single = [jobList[i] for i in jobs_split]
                pr = mp.Process(target = calcVectDist_mp,
                        args = (pr_id, jobList_single, in_Fw, in_Inv, shared_ncc))               
                pr.start()
                processes.append(pr)
            #collect the running results
            pr_results = [ ]
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
            #check the exit stats
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    raise RuntimeError("Process %d exited unexpectedly."%pr_id)
            dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
            dists = dists[0]
            del shared_ncc
            gc.collect()
                
                
        if (worker_n == 1) | (job_n == 1):          
            print("Using single cpu")
            #never change a changable variant in one function
            dists = calcVectDist_mp(-1, jobList, in_Fw, in_Inv, dists) 
        
        print("Finishing calculating euc transforms!")   
        
             
    elif dmetric == 'ang':
        #calculate rotation matrix
        Rin= calcRotMatrices(in_Fw)
        if len(in_Inv) > 0:
            Rin_Inv= calcRotMatrices(in_Inv)           
        else:
            Rin_Inv = ''
        if (worker_n > 1) & (job_n > 1):
            if job_n < worker_n:
                worker_n = job_n               
            print("Start %d processors..."%(worker_n))
            shared_ncc = mp.Array('f', np.int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2))   
            #start parallel computation
            spl_ids = np.array_split(range(job_n),worker_n) #split the jobs into workers
            processes = [ ]
            for pr_id,  jobs_split in enumerate(spl_ids):            
                jobList_single = [jobList[i] for i in jobs_split]           
                pr = mp.Process(target = calcAngDist_mp,
                                args = (pr_id, jobList_single, Rin, Rin_Inv,shared_ncc))        
                pr.start()
                processes.append(pr)
            #collect the running results
            pr_results = [ ]
            for pr in processes:
                pr.join()
                pr_results.append(pr.exitcode)
                #check the exit stats
            for pr_id in range(len(processes)):
                if pr_id != pr_results[pr_id]:
                    raise RuntimeError("Error: process %d exited unexpectedly."%pr_id)

            dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
            dists = dists[0]
            del shared_ncc
            gc.collect()  
            
            
        if (worker_n == 1) | (job_n == 1):          
            print("Using single cpu")
            #never change a changable variant in one function
            dists = calcAngDist_mp(-1, jobList, Rin,Rin_Inv, dists) 
                 
        print("Finishing calculating ang transforms distance!")  
        
    shutil.rmtree(tmpDir) #remove the dirs  
    return dists  # one dimension array float 32          
            
#@profile  
def calcVectDist_mp(pr_id, jobList_single, in_Fw, in_Inv, shared_ncc):
    with alive_bar(len(jobList_single), title="euc distances") as bar:
        for single_job in jobList_single:
           
            jobListChunk = np.load(single_job["file"],allow_pickle=True)
           
            g1 = in_Fw[jobListChunk[:,0],:]
            g2 = in_Fw[jobListChunk[:,1],:]
            if len(in_Inv)  == 0:
                g1Inv = ''
                g2Inv = ''
            else:
                g1Inv = in_Inv[jobListChunk[:,0],:]
                g2Inv = in_Inv[jobListChunk[:,1],:]
        
            dtmp = calcVectDist(g1,g2,g1Inv,g2Inv)
            
            shared_ncc[single_job["start"]:single_job["stop"]] = dtmp
            del jobListChunk, g1, g2, g1Inv, g2Inv, dtmp
            gc.collect()            
            bar()
            
    if pr_id == -1:
        return shared_ncc    
    os._exit(pr_id)

#@profile    
def calcVectDist(g1,g2,g1Inv,g2Inv):
    dv = g2-g1
    dtmp =  np.linalg.norm(dv, axis = 1)
    if len(g1Inv) > 0:
        dv = g2-g1Inv
        distsInv = np.linalg.norm(dv, axis = 1)
        
        dv = g1-g2Inv
        distsInv2 = np.linalg.norm(dv, axis = 1)
        
        dv = g1Inv - g2Inv
        distsInv3 = np.linalg.norm(dv, axis = 1)
        
        dists_all = np.array([dtmp, distsInv,distsInv2, distsInv3])
        dtmp = np.min(dists_all, axis = 0).astype(np.single)
        
        del dv, distsInv, distsInv2, distsInv3, dists_all, g1,g2,g1Inv,g2Inv
        gc.collect()
        
     
           
    return dtmp
 
def calcRotMatrices(in_angs):
    print("Starting calculating rotation matrices for each transform")
  
    Rin = np.zeros([in_angs.shape[0], 3,6 ], dtype = np.single)
    
    for i in range(in_angs.shape[0]):
        _,_, Rin[i,:,0:3] = tom_sum_rotation(in_angs[i,:], np.array([0,0,0]))
        Rin[i,:,3:6] = np.linalg.inv(Rin[i,:,0:3])
        
    print("Finishing calculating rotation matrices for each transform")
    
    
    return  Rin
    
#@profile   
def calcAngDist_mp(pr_id, jobList_single, Rin, Rin_Inv,shared_ncc):
    with alive_bar(len(jobList_single), title="ang distances") as bar:
        for singlejobs in jobList_single:
           
            jobListChunk = np.load(singlejobs["file"])
           
            dtmp = calcAngDist(Rin[jobListChunk[:,0],:,0:3], Rin[jobListChunk[:,1],:,3:6])
            if len(Rin_Inv) > 0:
                #Rs_Inv = Rin_Inv[jobListChunk[:,0],:,0:3]
                #Rs_Inv_Inv = Rin_Inv[jobListChunk[:,1],:,3:6]
                dtmpInv = calcAngDist(Rin_Inv[jobListChunk[:,0],:,0:3], Rin[jobListChunk[:,1],:,3:6])               
                dtmpInv2 = calcAngDist(Rin[jobListChunk[:,0],:,0:3], Rin_Inv[jobListChunk[:,1],:,3:6])
                dtmpInv3 = calcAngDist(Rin_Inv[jobListChunk[:,0],:,0:3], Rin_Inv[jobListChunk[:,1],:,3:6] )              
                dists_all = np.array([dtmp, dtmpInv, dtmpInv2,dtmpInv3 ])
                dtmp = np.min(dists_all, axis = 0)
               
            shared_ncc[singlejobs["start"]:singlejobs["stop"]] = dtmp  
            del jobListChunk, dtmp, dtmpInv,dtmpInv2,dtmpInv3,dists_all
            gc.collect()                            
            bar()
            
    if pr_id == -1:
        return shared_ncc
    os._exit(pr_id)        
    
#@profile    
def calcAngDist(Rs,RsInv):
    #multiple the two matrices 
        
    Rp = np.matmul(Rs, RsInv)   
    tr_Rp = (np.trace(Rp, axis1=1, axis2=2) - 1)/2 
    #calculate the angle distance 
    dists = np.lib.scimath.arccos(tr_Rp)/np.pi*180
    dists = (dists.real).astype(np.single)
   
    return dists #one dimention arrsy float32)
    
    
    
def genJobList(szIn, tmpDir, maxChunk):
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
        jobListSt[i]["stop"] = packages[i,1]
        np.save(jobListSt[i]["file"], jobListChunk)  #will waste a long time for writing and reading!  
    return jobListSt
        

