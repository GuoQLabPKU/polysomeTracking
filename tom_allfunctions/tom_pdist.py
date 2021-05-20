import numpy as np
import os
import gc
import shutil
import multiprocessing as mp
from tom_functions.tom_calc_packages import tom_calc_packages
from tom_functions.tom_sum_rotation import tom_sum_rotation
#from tom_functions.tom_calcTransforms import tom_calcTransforms


def tom_pdist(in_Fw, dmetric = 'euc', in_Inv = '', maxChunk = 600000000 ,verbose=1):
    '''
    dists=tom_pdist(in_Fw,dmetric,in_Inv,maxChunk)

    PARAMETERS

    INPUT
       in_Fw                inputdata nxdimension for angles OR distance nx3 in zxz 
       dmetric        ('euc') for euclidean distance metric or  
                            'ang'   for angles
       in_Inv           ('')  inverse data neede for needed for transformations    
       maxChunk   (10000) max chunk size
       verbose        (1) 0 for no output

    OUTPUT
       dists             distances in the same order as pdist from matlab

    EXAMPLE
  

    dd=tom_pdist(np.array([0 0 0;0 0 10; 10 20 30]),'ang');
    '''
    tmpDir = 'tmpPdist' #since the number of combination of pairs can be large 
    jobList = genJobList(in_Fw.shape[0], tmpDir, maxChunk) #jobList store each dict for each node
    dists = np.zeros(np.int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2), dtype = np.single ) # the distance between pairs of ribosomes , one dimention array
    print("Start calculating %s for %d transforms"%(dmetric, in_Fw.shape[0]))
    worker_n = len(jobList) #number of cores to use
    if dmetric == 'euc':
        if worker_n != 1:
            print("staring %d processors..."%worker_n)
            shared_ncc = mp.Array('f', np.int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2))            
            avail_cpu = mp.cpu_count()
            if avail_cpu < worker_n:
                worker_n = avail_cpu
                print("Warning: No enough CPUs are available! Use %d CPUs instead."%worker_n)
            processes = [ ]
            for pr_id in range(worker_n):
                jobList_single = jobList[pr_id]
                pr = mp.Process(target = calcVectDist_mp,
                        args = (pr_id, jobList_single, in_Fw, in_Inv, shared_ncc))
                
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
                    os._exit(-1)
            dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
            dists = dists[0]
            gc.collect()
                
                
        else:
            print("using single node")
            for i in range(worker_n):
                #never change a changable variant in one function
                dists[jobList[i]["start"]:jobList[i]["stop"]] = calcVectDist_mp(-1, jobList[i], in_Fw, in_Inv, dists) 
        print("Finishing calculating transforms!")   
        
             
    elif dmetric == 'ang':
        Rin, RinInv = calcRotMatrices(in_Fw)
        if in_Inv != '':
            Rin_Inv, _ = calcRotMatrices(in_Inv)
            print("using inverse transforms")
        else:
            Rin_Inv = ''
        
        if worker_n != 1:
            print("staring %d processors..."%(worker_n))
            shared_ncc = mp.Array('f', np.int(in_Fw.shape[0]*(in_Fw.shape[0]-1)/2))   
            avail_cpu = mp.cpu_count()
            if avail_cpu < worker_n:
                worker_n = avail_cpu
                print("Warning: No enough CPUs are available! Use %d CPUs instead."%worker_n)
            processes = [ ]
            for pr_id in range(worker_n):
                jobList_single = jobList[pr_id]
                pr = mp.Process(target = calcAngDist_mp,
                                args = (pr_id, jobList_single, Rin, RinInv,Rin_Inv, shared_ncc))        
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
                    os._exit(-1)
            dists = np.frombuffer(shared_ncc.get_obj(), dtype=np.float32).reshape(1,-1)
            dists = dists[0]
            gc.collect()       
        else:
            print("using single node")
            for i in range(worker_n):
                #never change a changable variant in one function
                dists[jobList[i]["start"]:jobList[i]["stop"]] = calcAngDist_mp(-1, jobList[i], Rin, RinInv,Rin_Inv, dists) 
        print("Finishing calculating transforms distance!")  
        
    shutil.rmtree(tmpDir) #remove the dirs  
    return dists  # one dimension array           
            
   
def calcVectDist_mp(pr_id, jobList_single, in_Fw, in_Inv, shared_ncc):
    jobListChunk = np.load(jobList_single["file"])
    g1 = in_Fw[jobListChunk[:,0],:]
    g2 = in_Fw[jobListChunk[:,1],:]
    if in_Inv  == '':
        g1Inv = ''
        g2Inv = ''
    else:
        print("using inverse transforms")
        g1Inv = in_Inv[jobListChunk[:,0],:]
        g2Inv = in_Inv[jobListChunk[:,1],:]
    dtmp = calcVectDist(g1,g2,g1Inv,g2Inv)
    if pr_id == -1:
        return dtmp
    else:
        shared_ncc[jobList_single["start"]:jobList_single["stop"]] = dtmp
        os._exit(pr_id)
    
def calcVectDist(g1,g2,g1Inv,g2Inv):
    dv = g2-g1
    dtmp =  np.sqrt(np.sum(dv*dv,axis = 1))
    if g1Inv != '':
        dv = g2-g1Inv
        distsInv = np.sqrt(np.sum(dv*dv,axis = 1))
        #new
        dv = g1-g2Inv
        distsInv2 = np.sqrt(np.sum(dv*dv,axis = 1))
        
        dv = g1Inv - g2Inv
        distsInv3 = np.sqrt(np.sum(dv*dv,axis = 1))
        
        dists_all = np.array([dtmp, distsInv, distsInv2, distsInv3])
        dtmp = np.min(dists_all, axis = 0)
    
    return dtmp
 
def calcRotMatrices(in_angs):
    print("Starting calculating rotation matrices for each transforms")
    Rin = np.zeros([in_angs.shape[0], 3,3 ], dtype = np.single)
    RinInv = np.zeros([in_angs.shape[0], 3,3 ], dtype = np.single)
    
    for i in range(in_angs.shape[0]):
        _,_, Rin[i,:,:] = tom_sum_rotation(in_angs[i,:], np.array([0,0,0]))
        RinInv[i,:,:] = np.linalg.inv(Rin[i,:,:])
        
    print("Finishing calculating rotation matrices for each transforms")
    
    return  Rin, RinInv
    
    
def  calcAngDist_mp(pr_id, jobList_single, Rin, RinInv,Rin_Inv, shared_ncc):
    jobListChunk = np.load(jobList_single["file"])
    Rs = Rin[jobListChunk[:,0],:,:]
    RsInv = RinInv[jobListChunk[:,1],:,:]
    dtmp = calcAngDist(Rs, RsInv)
    if Rin_Inv != '':
        Rs_Inv = Rin_Inv[jobListChunk[:,0],:,:]
        dtmpInv = calcAngDist(Rs_Inv, RsInv)
        dists_all = np.array([dtmp, dtmpInv])
        dtmp = np.min(dists_all, axis = 0)
    if pr_id == -1:
        return dtmp
    else:
        shared_ncc[jobList_single["start"]:jobList_single["stop"]] = dtmp
        os._exit(pr_id)        
    
    
def calcAngDist(Rs,RsInv):
    #multiple the two matrices
    tr_Rp = np.zeros(Rs.shape[0], dtype = np.single)
    for i in range(Rs.shape[0]):
        Rp = np.dot(Rs[i,:,:], RsInv[i,:,:])
        tr_Rp[i] = np.trace(Rp)
    #calculate the angle distance 
    dists = np.array([np.arccos(i)/np.pi*180 for i in (tr_Rp-1)/2])
    #extract the real part of the dists and single them
    dists = np.single(dists.real)
    
    return dists #one dimention arrsy float32
    
    
    
def genJobList(szIn, tmpDir, maxChunk):
    lenJobs = np.uint32(szIn*(szIn-1)/2)
    jobList = np.zeros([lenJobs,2], dtype = np.uint32) #expand the range of positive int save memory(no negative int)
    startA = 0   
    for i in range(szIn):
        v2 = np.arange(i+1,szIn, dtype = np.uint32).reshape(-1,1)
        v1 = np.repeat(i, len(v2)).astype(np.uint32).reshape(-1,1)
        jobListTmp = np.concatenate((v1,v2),axis = 1).astype(np.uint32) 
        endA = startA+jobListTmp.shape[0]
        jobList[startA:endA,:] = jobListTmp
        startA = endA  
    numOfPackages = np.int(np.floor(jobList.shape[0]/maxChunk))
    if numOfPackages < 1:
        numOfPackages = 1
    packages = tom_calc_packages(numOfPackages, jobList.shape[0]) #split the jobList into different size, the packages is one array
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
        np.save(jobListSt[i]["file"], jobListChunk)
        
    return jobListSt
        
 

#if __name__ == '__main__':
#    startSt  = tom_calcTransforms('./simOrderRandomized.star', 100, tomoNames='', 
#                                      dmetric='exact', outputName='allTransforms.star', verbose=1, worker_n = 2)
#    stest = np.array([startSt["pairTransAngleZXZPhi"].values, startSt["pairTransAngleZXZPsi"].values, startSt["pairTransAngleZXZTheta"].values])
#    stestinv = np.array([startSt["pairInvTransAngleZXZPhi"].values, startSt["pairInvTransAngleZXZPsi"].values, 
#                         startSt["pairInvTransAngleZXZTheta"].values])
# 
#    stest = np.array([startSt["pairTransVectX"].values, startSt["pairTransVectY"].values, startSt["pairTransVectZ"].values])
#    stestinv = np.array([startSt["pairInvTransVectX"].values, startSt["pairInvTransVectY"].values, 
#                         startSt["pairInvTransVectZ"].values])    
#    stest = stest.T
#    stestinv = stestinv.T
#    pdist = tom_pdist(stest, 'ang', in_Inv = stestinv, maxChunk = 50000)
    
    
    
        
        
        
    