import numpy as np
import os 
import shutil

from nemotoc.py_memory.tom_memalloc import tom_memalloc
from nemotoc.py_cluster.tom_pdist_cpu import fileSplit,genjobsList_oneGPU
from nemotoc.py_cluster.tom_calc_packages import tom_calc_packages

def tom_A2Odist(transVect, transAng, shift, rot, 
                worker_n = 1, gpu_list = None, 
                cmb_metric = 'scale2Ang', pruneRad = 100):
    '''
    TOM_A2DIST is aimed to calculate the distance between 
    one group of transforms with another one transform
    
    Input 
    transVect     (nx3)the vect trans of one group of transforms (2D-array)
    transAng      (nx3)the angle trans of one group of transforms (2D-array)
    shift         the vect trans of one transforms(1D-array) [1,3]
    rot           the angle trans of one transform(1D-array)
    worker_n   
    gpu_list
    cmb_metric
    pruneRad
    
    Output 
    
    distVect     1D array
    distAng      1D array
    distCombine  1D array
    
    '''
    
    transVect = np.append(transVect, shift.reshape(-1,3), axis  = 0) 
    transAng = np.append(transAng,rot.reshape(-1,3), axis = 0)
    distVect, distAng, distCombine = getDist(transVect, transAng, 
                                              worker_n, gpu_list, 
                                              cmb_metric, pruneRad)
    return distVect, distAng, distCombine
              
def getDist(transVect, transAng,worker_n, gpu_list, cmb_metric, pruneRad): 
    if gpu_list is not None:
        worker_n = None
    maxChunk = tom_memalloc(None, worker_n, gpu_list) #mallocal the memory
    if isinstance(worker_n, int):
        from nemotoc.py_cluster.tom_pdist_cpu import tom_pdist
        tmpDir = 'tmpA2Odistcpu' 
        jobListdict = genJobListCPU(transVect.shape[0] - 1, tmpDir, maxChunk)
    else:        
        if (isinstance(gpu_list,list)) & (len(gpu_list) == 1):
            from nemotoc.py_cluster.tom_pdist_gpu2 import tom_pdist                
        else:
            from nemotoc.py_cluster.tom_pdist_gpu import tom_pdist
        tmpDir = 'tmpA2Odistgpu' 
        jobListdict = genJobListGpu(transVect.shape[0] - 1, tmpDir, maxChunk)

    distsVect = tom_pdist(transVect, maxChunk, worker_n, gpu_list, 'euc', 
                          '', jobListdict, tmpDir, 0, 0)
    distsAng =  tom_pdist(transAng,  maxChunk ,worker_n, gpu_list,'ang',
                          '',jobListdict,tmpDir, 1, 0)

    if cmb_metric == 'scale2Ang':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+distsVect)/2
    elif cmb_metric == 'scale2AngFudge':
        distsVect = distsVect/(2*pruneRad)*180
        distsCN = (distsAng+(distsVect*2))/2    
    return distsVect, distsAng, distsCN


def genJobListGpu(lenJobs, tmpDir, maxChunk):
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
            jobListSt[i]["file"] = "%s/jobListChunk%d_gpu%d.npy"%(tmpDir, i, gpu_id)
            jobListSt[i]["start"] = packages[i,0]
            jobListSt[i]["stop"] = packages[i,1]
            np.save(jobListSt[i]["file"], jobListChunk)  #will waste a long time for writing and reading!  
        jobsListSt_dict[gpu_id] = jobListSt
    return jobsListSt_dict
   
    
def genJobListCPU(lenJobs, tmpDir, maxChunk):
    jobList = np.zeros([lenJobs,2], dtype = np.int) 
    jobList[:,0] = np.arange(lenJobs)
    jobList[:,1] = np.repeat(-1, lenJobs) 
    numOfPackages = np.int(np.ceil(lenJobs/maxChunk)) 
    packages = tom_calc_packages(numOfPackages, lenJobs)  
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