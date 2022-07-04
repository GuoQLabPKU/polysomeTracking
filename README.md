# Neighboring particles' conformation clustering  based on particles' positions and euler angles and the polyribosome tracking
![This is an image](https://github.com/werhoog/polysomeTracking/blob/main/image/concept2.PNG)
## Concept
NEMO-TOC can classify the relative spatial arrangement of neighbors based on the positions and Euler angles. 
Features to track ordered linear assemblies are also added.
Reference:
## Scripts description (located at nemotoc/)
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
    ```
    pathToConda/bin/conda create --name nemotoc
    ```
3. activate the enviroment
    ```
    source pathToConda/etc/profile.d/conda.sh
    pathToConda/conda activate nemotoc
    ```
4. install dependencies 
    ```
    git clone https://github.com/GuoQLabPKU/polysomeTracking.git
    cd polysomeTracking/
    bash install_conda_pkgs.sh
    ```
    > *Replace the cudatoolkit version with the appropriate version of CUDA installed with the GPU drivers (you can check the CUDA version with nvidia-smi)*
5. install NEMO-TOC
    ```
    python setup.py build
    python setup.py sdist
    pip install dist/nemotoc-1.0.0b0.tar.gz
    ```
## Platform
Any platform (linux/windows/macos) was tested
## Usage
1. make one directory to do the analysis 
   ```
   mkdir myNemoProj
   cd myNemoProj
   ```
2. generate a configure file. Running the command below will generate a configure file named conf.py
    ```
    nemotocGen --getTestData 1 #copies or fetches the testData
    nemotocGen --getConf 1 #copies or fetches the configure script
    ```
3. modify suitable parameters in the generated conf.py and run the command below  
    ```
    nemotocRun -c conf.py
    ```
## Debug
```
pytest nemotoc_test/test_*
```
This will test three functions:
- test_polysome.py: test if track right  linear assemblies(polysomes) from the simulation dataset 
- test_fillupRibo.py: test if fill up right particles after manully delete two particles 
- test_branchClean.py: test if clean the branches created manully in the simulation dataset

### If meet any memory error, please reduce the input variable ***Chunk*** in function ***tom_memalloc*** **(nemotoc/py_memory/tom_memalloc.py)**
## Output 
Given an input starfile named 'particles.star' in ***main.py***, for example, several folders will be created like below,

cluster-particles\
  - run0   (*change the folder name by  changing ***run_time*** in **main.py**)*
    - allTransformsb4Relink.star  
      > *transforms starfiles before filtering*
    - allTransforms.star 
      > *transforms starfiles after filtering*
    - allTransformsFillUp.star 
      > *transforms starfiles after branches removal and particle fillingup*
      
      | column name | description | 
      | ---- | ---- |
      | fillUpProb | the probability of the transform. -1:none fill up/1.1: transforms with middle fillup particles(fillUoPoly_addNum>1)/other:transforms with last filled up particles| 
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
          > *linear assemblies (polysomes) from the same tomogram*
        - cXXX_longestPoly.png
          > *the longest linear assemblies (polysomes) from clusterXXX*
        - cXXX_polyLengthDist.png
          > *the distribution of length of linear assemblies (polysomes) from clusterXXX*
     - particlesFillUp.star
        > *the particlesFillUp.star with filled up particles as well as from particles.star
        
       | column name | description | 
       | ---- | ---- |
       | if_fillUp| the state of particles. -1:none fill up/1.1: middle fillup particles(fillUoPoly_addNum>1)/1:last filled up particles |
       
