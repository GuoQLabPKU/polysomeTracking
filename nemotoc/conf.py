import numpy as np
import time

from nemotoc.py_run.py_run import runPoly

#####BASIC PARAMTERS SETTING########
input_star =  'test.star' #the star file
project_folder = 'projTest' #the folder to store all runs.
run_folder = 'run0' #the folder storing the results of each run
pixel_size = 3.42 #in Ang, the pixel size of input starfile
particle_radius = 170 #in Ang, the radius of the input particle
cluster_threshold = 25 #the threshold to cut-off the dendrogram tree for clustering
minNumTransform_ratio = -1  #select clusters with at least minNumTransformRatio transforms. 0:keep any cluster regardless of the #transforms/-1:default ratio:1%
remove_branches = 1 #1:branch removal; 0:switch off. If remove the branches of linear asemblies
average_particles = 1 #average particles from each cluster. 1:switch on, 0:switch off
search_radius = particle_radius*2 #in Ang. The searching range for neighbors. Two neighbors will be linked within this range.

#####PREPROCESS PARAMETERS####
min_dist = particle_radius/pixel_size*0.8 #in pixeles, the minmum distance to remove repeat particles
if_stopgap = 1 #if input a stopgap starfile, then transfer into relion2. 0:input is not stopgap file type
subtomo_path = 'subtomograms' #if input is stopgap file, need specify the path pointing to subtomograms
ctf_file =  'miss30Wedge.mrc' #if input is stopgap file, need specify the missing wedge file

####PARALLEL COMPUTATION PARAMETERS####
cpuN = 15 #number of CPU for parallel computation. 1:switch off parallel computation
gpu_list = None  #leave None if no gpu available. Else offer gpuID list like [0]//[0,1]
avg_cpuN = 35 #the number of CPUs when average particles by relion if average_particles == 1

#####VISUAL PARAMETERS######
vectorfield_plotting = 'basic' #advance:show detailed information of polysomes in the vector field figure
show_longestPoly = 1 #plot and save the longest polysome in each cluster 0:switch off

#####AVERAGE PARAMETERS#####
if_avg = 1  #if average particles 0:switch off
avg_pixS = 3.42  #the pixel size of particles for relion averaging
avg_minPart = 50 #the minimal number of particles requirement for average of each cluster
avg_maxRes = 20 #the maximum resolution for relion averaging
avg_callByPython = 0 #if use python to call relion_reconstruct command 0: generate linux scripts to run relion_reconstruct, 1:use python to call relion
                     #if set to 1, you should make sure the relion_reconstruct command is searchable in the system pathway

#####ADVANCED PARAMETERS SETTING######
#for huge dataset
transNr = 100000 #if the total number of transformation >= transNr. Remainning transformation will be assigned into clusters by k-means methods
transNr_initialCluster = 10000 #the amount of transformations used for the initial hirachical clustering
iterationNr = 1 #the cycle number for k-means for cluster assignment
#for gap fillingup
link_depth = 2 #the searching depth for linking adjacent transforms into longer polysomes. 0:remove branches
fillUpPoly_addN = 1 #number of particles added in each tail of polysome to fill up gaps   0:switch off filling up step
fillUpPoly_model = 'lognorm' #the type of fitted distribution for filling up step(genFit:based on experimental data; lognorm:base on lognorm model; max:no model fitting)
fillUpPoly_threshold = 0.05 #threshold to accept filled up particles. The smaller, the more convinced of accepted interpolated particles


if __name__ == '__main__': 

    fillUpPoly = { }
    fillUpPoly['addNum'] = fillUpPoly_addN
    fillUpPoly['fitModel'] = fillUpPoly_model
    fillUpPoly['threshold'] = fillUpPoly_threshold

    avg = { }
    avg['filt'] = { }
    avg['filt']['minNumPart'] = avg_minPart
    avg['filt']['maxNumPart'] = np.inf
    avg['pixS'] = avg_pixS
    avg['maxRes'] = avg_maxRes
    avg['cpuNr'] = avg_cpuN
    avg['callByPython'] = avg_callByPython
    
    begin = time.time()
    runPoly(input_star, run_folder, project_folder,
            pixel_size, min_dist, if_stopgap, subtomo_path, ctf_file,
            search_radius, link_depth, cluster_threshold,
            minNumTransform_ratio, fillUpPoly, cpuN, gpu_list, remove_branches, vectorfield_plotting,
            show_longestPoly, if_avg, average_particles, avg,transNr,
            transNr_initialCluster,iterationNr)
    end = time.time()
    print('Successfully finish NEMO-TOC with %.2fs'%(end-begin))
