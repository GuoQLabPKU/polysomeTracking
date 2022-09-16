#####BASIC PARAMTERS ########
input_star =  'test.star' #the input star file 
project_folder = 'projTest' #the folder to store all run_folders.
run_folder = 'run0' #the folder storing the results of each run
pixel_size = 3.42 #in Ang, the pixel size of input starfile
cluster_threshold = 25 #the threshold to cut-off the dendrogram tree for clustering
minNumTransform_ratio = -1  #select clusters with at least minNumTransformRatio transforms. 0:keep any cluster regardless of the #transforms/-1:default ratio:1%
remove_branches = 1 #1:branch removal; 0:switch off. If remove the branches of linear assemblies
do_avg = 1 #average particles from each cluster. 1:switch on, 0:switch off
search_radius = 170*2 #in Ang. The searching range for neighbors. Two neighbors would be linked within this range.
do_errorEstimate = 1 #if estimate the probaility that one transformation belongs to another clusters

#####PREPROCESS STAR FILE PARAMETERS####
min_dist = 170*0.8 #in Ang, the minmum distance to remove repeat particles. Default:raduis of particles*0.8
subtomo_path = 'subtomograms' #if input is stopgap file, need specify the path pointing to subtomograms
ctf_file =  'miss30Wedge.mrc' #if input is stopgap file, need specify the missing wedge file

####PARALLEL COMPUTATION PARAMETERS#####
cpuN = 15 #number of CPU for parallel computation. 1:switch off parallel computation
gpu_list = None  #leave None if no gpu available. Else provide gpuID list like [0]//[0,1]
avg_cpuN = 35 #the number of CPUs for averaging particles by relion if do_avg == 1

#####VISUAL PARAMETERS######
vectorfield_plotting = 'basic' #advance:show detailed information of polysomes in the vector field figure/None:switch off
show_longestPoly = 1 #plot and save the longest polysome in each cluster 0:switch off

#####AVERAGE PARAMETERS#####
avg_pixS = 3.42  #the pixel size of particles for relion averaging
avg_maxPart = 20000 #the maximum number of particles to average particles from each cluster
avg_minPart = 50 #the minimal number of particles requirement for averaging from each cluster
avg_maxRes = 20 #the maximum resolution for relion averaging
avg_callByPython = 0 #if use python to call relion_reconstruct command 0: generate linux scripts to run relion_reconstruct, 1:use python to call relion
                     #if set to 1, you should make sure the relion_reconstruct command is searchable in the system pathway
avg_command = 'mpirun -np XXX_cpuNr_XXX `which relion_reconstruct_mpi` --i XXX_inList_XXX --maxres XXX_maxRes_XXX --angpix  XXX_pix_XXX --ctf --3d_rot --o XXX_outAVG_XXX'
              #generic (relion) call template
#####ADVANCED PARAMETERS ######
#for huge dataset
transNr_initialCluster = 10000000000 #the amount of transformations used for the initial hirachical clustering
iterationNr = 1 #the cycle times for k-means for cluster assignment
#for gap fillingup
link_depth = 2 #the searching depth for linking adjacent transforms into longer polysomes.
fillUpPoly_addN = 1 #number of particles added in each tail of polysome to fill up gaps   0:switch off filling up step
fillUpPoly_model = 'lognorm' #the type of fitted distribution for filling up step(genFit:based on experimental data; lognorm:base on lognorm model; max:no model fitting)
fillUpPoly_threshold = 0.05 #threshold to accept filled up particles. The smaller, the more convinced of accepted interpolated particles

