# Neighboring particles' conformation clustering  based on particles' positions and euler angles and the polyribosome tracking
![This is an image](https://github.com/werhoog/polysomeTracking/blob/main/image/concept2.PNG)

this folder contains the scripts for describing quantitavely polysomes high order conformations 
## Concept
NEMO-TOC can classify the relative spatial arrangement of neighbors based on the positions and Euler angles. 
Features to track ordered linear assemblies are also added.
Reference:
## Scripts description
- **polysome_class:** folder with function class written by python
- **py_align:** folder with scripts alignning the direction of particle pairs transformation
- **py_cluster:** folder with scripts classifing the neighbors' pairs transformation
- **py_io:** folder with scripts for star files/tomogram MRC files I/O
- **py_link:** folders with scripts linking neighbors' pairs transformations to longer linear assemblies
- **py_log:** folders with scripts making log file
- **py_memory:** folder with scripts allocating memory of CPU/GPU
- **py_mergePoly:** folder with scripts filling up inferred particles to merge shorter linear assemblies to longer ones
- **py_simulation:** folder with scripts creating simulation datasets 
- **py_stats:** folder with scripts fitting distribution and estimating the errors of transformation clusters assignment
- **py_summary:** folder with scripts summaring transformation clusters
- **py_test:** folder with scripts testing branches cleaning & linear assemblies tracking & particles' fillingup
- **py_transform:** folder with scripts calculating the transformation of neighbor pairs
- **py_vis:** folder with scripts for visulization
## Install
1. install miniconda
    https://docs.conda.io/en/latest/miniconda.html
2. create enviroment
    pathToConda/bin/conda create --name nemotoc
3. activate the enviroment
    source pathToConda/etc/profile.d/conda.sh
    pathToConda/conda activate nemotoc
4. install dependencies 
    git clone https://github.com/GuoQLabPKU/polysomeTracking.git
    cd polysomeTracking/
    bash install_conda_pkgs.sh
## Platform
Any platform (linux/windows/macos)
## Usage
Modify suitable parameters in main.py, then `python main.py`
## Debug
```
pyest py_test/test_*
```
This will test three functions:
- test_polysome.py: test if track right  linear assemblies(polysomes) from simulation data (uncomment the last line if want to keep and check the output)
  > test_polysome.py can receive a parameter ***particleStar***. Modify this parameter to your real particle.star file. Then the euler angles
  > of simulation data will come from the particle.star
- test_fillupRibo.py: test if fill up right particles after manully delete two particles 
- test_branchClean.py: test if clean the branches created manully in simulation data

***NOTICE: thoes test_XXX.py need modify the sys.path.append('THE PATH YOU DOWNLOADED THESE SCRIPTS') in the head line***
### If meet any memory error, please reduce the input variable ***Chunk*** in function ***tom_memalloc*** **(py_memory/tom_memalloc.py)**
## Output 
Given an input starfile named 'particles.star' in ***main.py***, for example, several folders will be created like below,
cluster-particle
  - run0   (*change the folder name using ***run_time*** in **main.py**)*
    - allTransformsb4Relink.star  
      > *transforms starfiles before relinking*
    - allTransforms.star 
      > *transforms starfiles after relinking*
    - allTransformsFillUp.star 
      > *transforms starfiles after branches cleaning and particle fillingup*
      
      | column name | description | 
      | ---- | ---- |
      | fillUpProb | the probability of the transform. -1:none fill up/1.1: transforms with middle fillup particles(fillUoPoly_addNum>1)/other:transforms with last filled up particles| 
    - avg 
      - exp
      - model
    - clusters
      > *transList for each transform cluster*
      - c1/transList.star
      - c2/transList.star
      - cXXX/transList.star
    - scores 
      > *transform clusters assignment scores*
      - treeb4Relink.npy
      - tree.npy
    - stat
      > *summary of each transform cluster and each linear assemblies*
      - statePerClass.star
      - statePerPoly.star   
    - vis
      - averages
      - clustering
        - linkLevel.png
        - tree.png
      - innerClusterDist
        > *the distribution of distances between each transform and the average transform from one transform cluster*
        - cluster1.png
        - clusterXXX.png
      - fitInnerClusterDist
        > *using different models to fit distribution of distances between each transform and the average transform from one transform cluster*
        - cXXX_dill.pkl 
        > *the fitting results of gauss KDE*
        - cXXX_fitDist.png
        > *the fitting results figures using gauss KDE and lognorm*
        - distanceDistFit_cXXX.csv 
        > *the parameters for lognorm fitting*
      - noiseEstimate
        > *compare the distribution of 1.distance between each transform and the average transform from the same transform cluster; 2.distance between transforms from other clusters and the average transform from cluster of 1*
      - vectfields
        - tomoID.png 
          > *polysomes from the same tomogram*
        - cXXX_longestPoly.png
          > *the longest polysome from clusterXXX*
        - cXXX_polyLengthDist.png
          > *the distribution of length of polysomes from clusterXXX*
     - particlesFillUp.star
        > *the particlesFillUp.star with filled up particles as well as from particles.star
        
       | column name | description | 
       | ---- | ---- |
       | fillUpProb | the state of particles. -1:none fill up/1.1: middle fillup particles(fillUoPoly_addNum>1)/1:last filled up particles |
       
