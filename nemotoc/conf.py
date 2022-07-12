import numpy as np
import os

from nemotoc.polysome_class.polysome import Polysome

#####BASIC PARAMTERS SETTING########
input_star =  'all_particles_neuron_warp.star' #the star
project_folder = 'cluster-all_particles_neuron_warp' #the folder to store all runs.
run_time = 'threshold20_relink01percent' #the folder storing the results of each run
pixel_size = 3.42 #in Ang, the pixel size of input starfile
particle_radius = 50*pixel_size #in Ang, the radius of the input particle
cluster_threshold = 20 #the threshold to cut-off the dendrogram tree for clustering
minNumTransform_ratio = 0.01  #select clusters with at least minNumTransformRatio transforms. -1:keep any cluster regardless of the #transforms
remove_branches = 0 #1:branch removal; 0:switch off
average_particles = 1 #average particles from each cluster. 1:switch on, 0:switch off
search_radius = particle_radius*2 #in Ang. The searching range for neighbors. Two neighbors will be linked within this range.

#####PREPROCESS PARAMETERS####
min_dist = particle_radius/pixel_size #in pixeles, the minmum distance to remove repeat particles
if_stopgap = 0 #if input a stopgap starfile, then transfer into relion2. 0:input is not stopgap file type
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
avg_minPart = 50 #the minmual number of particles requirement for average of each cluster
avg_maxRes = 20 #the maximum resolution for relion averaging
avg_callByPython = 0 #if use python to call relion_reconstruct command 0: generate linux scripts to run relion_reconstruct, 1:use python to call relion
                     #if set to 1, you should make sure the relion_reconstruct command is searchable in the system pathway

#####ADVANCED PARAMETERS SETTING######
link_depth = 2 #the searching depth for linking adjacent transforms into longer polysomes. 0:remove branches
fillUpPoly_addN = 1 #number of particles added in each tail of polysome to fill up gaps   0:switch off filling up step
fillUpPoly_model = 'lognorm' #the type of fitted distribution for filling up step(genFit:based on experimental data; lognorm:base on lognorm model; max:no model fitting)
fillUpPoly_threshold = 0.05 #threshold to accept filled up particles. The smaller, the more convinced of accepted interpolated particles


def runPoly(input_star, run_time, project_folder, pixel_size, min_dist, if_stopgap, subtomo_path, ctf_file,
            search_radius, link_depth, cluster_threshold, minNumTransform_ratio, fillUpPoly, cpuN, gpu_list, remove_branches,
            vectorfield_plotting, show_longestPoly, if_avg, average_particles, avg):
    #check the type of input parameters
    assert isinstance(input_star, str)
    assert isinstance(run_time, str)
    assert isinstance(project_folder, str)
    assert isinstance(pixel_size, (int, float))
    assert isinstance(min_dist, (int, float))
    assert isinstance(if_stopgap, (int, float))
    assert isinstance(subtomo_path, str)
    assert isinstance(ctf_file, str)
    assert isinstance(search_radius, (int, float))
    assert isinstance(link_depth, (int, float))
    assert isinstance(cluster_threshold, (int, float))
    assert isinstance(minNumTransform_ratio, (int, float))
    assert isinstance(fillUpPoly, dict)
    assert isinstance(cpuN, int)
    assert isinstance(gpu_list, (list, type(None)))
    assert isinstance(remove_branches, int)
    assert isinstance(vectorfield_plotting, str)
    assert isinstance(show_longestPoly, int)
    assert isinstance(if_avg, (int, float))
    assert isinstance(average_particles, int)
    assert isinstance(avg, dict)
    #check if the project_folder exist
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)

    polysome1 = Polysome(input_star = input_star, run_time = run_time)
    #calculate transformations
    polysome1.transForm['pixS'] = pixel_size
    polysome1.transForm['maxDist'] = search_radius
    polysome1.transForm['branchDepth'] = link_depth
    #do clustering and filtering
    polysome1.classify['clustThr'] = cluster_threshold
    polysome1.sel[0]['minNumTransform'] = minNumTransform_ratio

    polysome1.creatOutputFolder()  #create folder to store the result
    polysome1.preProcess(if_stopgap, subtomo_path, ctf_file, min_dist) #preprocess
    polysome1.calcTransForms(worker_n = cpuN) #calculate transformations
    polysome1.groupTransForms(worker_n = cpuN, gpu_list = gpu_list)  #cluster transformations
    transListSel, selFolds = polysome1.selectTransFormClasses() #filter clusters
    polysome1.genOutputList(transListSel, selFolds) #save the filtered clusters
    polysome1.alignTransforms() #align the transformationsto the same direction
    polysome1.analyseTransFromPopulation('','',1, 0)  #summary the clusters but w/o any polysome information
    polysome1.fillPoly = fillUpPoly #fill up the gaps
    polysome1.link_ShortPoly(remove_branches, cpuN) #link transforms into a long linear chain
    polysome1.analyseTransFromPopulation('','',0, 1) #summary the clusters
    polysome1.noiseEstimate() #estimate the purity of each cluster

    polysome1.vis['vectField']['type'] = vectorfield_plotting
    polysome1.vis['longestPoly']['render'] = show_longestPoly
    polysome1.visResult()
    polysome1.visLongestPoly()

    #average particles subset using relion_reconstruct
    if if_avg:
        polysome1.avg = avg
        polysome1.generateTrClassAverages()

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

    runPoly(input_star, run_time, project_folder,
            pixel_size, min_dist, if_stopgap, subtomo_path, ctf_file,
            search_radius, link_depth, cluster_threshold,
            minNumTransform_ratio, fillUpPoly, cpuN, gpu_list, remove_branches, vectorfield_plotting,
            show_longestPoly, if_avg, average_particles, avg)
    print('Successfully finish NEMO-TOC')
