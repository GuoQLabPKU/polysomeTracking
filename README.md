# polysomes tracking based on ribosome positions and euler angles
this folder contains the scripts for describing quantitavely polysomes high order conformations 
## Scripts description
- **polysome_class:** folder with polysome class written by python
- **py_align:** folder with scripts alignning the direction of ribosome pairs transformation
- **py_cluster:** folder with scripts classifing the ribosome pairs transformation
- **py_io:** folder with scripts for star files/tomogram MRC files I/O
- **py_link:** folders with scripts linking ribosome pairs transformation to longer polysomes
- **py_log:** folders with scripts making log file
- **py_memory:** folder with scripts allocating memory of CPU/GPU
- **py_mergePoly:** folder with scripts filling up inferred ribosomes to merge shorter polysomes to longer one
- **py_simulation:** folder with scripts creating polysome simulation data
- **py_stats:** folder with scripts fitting distribution and estimating the errors of transform class assignment
- **py_summary:** folder with scripts summaring polysomes and transform classes
- **py_test:** folder with scripts testing branches-cleaning/polysome-tracking/ribosome-fillingup
- **py_transform:** folder with scripts calculating the transformation of ribosome pairs
- **py_vis:** folder with scripts for visulization
## Dependencies
- python: 3.6.7(3.7.9/3.8.3)
- numpy: 1.16.1(1.18.5/1.19.2)
- pandas: 1.0.5(1.1.1/1.2.1)
- jupyter: 1.0.0
- matplotlib: 3.3.3
- cupy: 9.0.0(for GPU calculation)
- cudatoolkit: 11.0.3(for GPU calculation)
- pytest: 6.2.4
- networkx:2.6.2
- seaborn:0.11.2
- dill:0.3.4
## Flatform
Any platform (linux/windows/macos)should be fine. But linux is largely tested to run these scripts.
## Usage
Modify suitable parameters in main.py, then `python main.py`

## Debug
```
pyest py_test/test_*
```
If meet any memory error, please reduce the input variable ***Chunk*** in function ***tom_memalloc*** **(py_memory/tom_memalloc.py)**
## Output 
Given an input starfile named 'particles.star' in ***main.py***, for example, several folders will be created like below,
cluster-particle
  - run0   (*change the folder name using ***run_time*** in **main.py**)*
    - allTransformsb4Relink.star  (*transforms starfiles before relinking*)
    - allTransforms.star (*transforms starfile after relinking*)
    - allTransformsFillUp.star (*transforms starfiles after branches cleaning and ribosome fillingup*)
    - avg 
      - exp
      - model
    - classes (*transList for each transform class*)
      - c1
      - c2
      - cXXX
    - scores (*transform classes assignment scores*)
      - treeb4Relink.npy
      - tree.npy
    - stat (*summary of each transform class and each polysome*)
      - statePerClass.star
      - statePerPoly.star   
    - vis
      - averages
      - clustering
        - linkLevel.png
        - tree.png
      - distanceDist (*the distribution of distances between each transform and the average transform from one transform class*)
        - class1.png
        - classXXX.png
      - fitDistanceDist (*using different models to fit distribution of distances between each transform and the average transform from one transform class*)
        - cXXX_dill.pkl (*the fitting results of gauss KDE*)
        - cXXX_fitDist.png (*the fitting results using gauss KDE and lognorm*)
        - distanceDistFit_cXXX.csv (*the parameters for lognorm fitting*)
      - noiseEstimate (*compare the distribution of 1.distance between each transform and the average transform from the same transform class; 2.distance between transforms from other classes and the average transform from 1*)
        - cXXX_otherClasses_angle distance.png 
        - cXXX_otherClasses_vect distance.png
        - cXXX_otherClasses_combined distance.png
      - vectfields
        - tomoID.png (*polysomes from the same tomogram*)
        - cXXX_longestPoly.png(*the longest polysome from classXXX*)
        - cXXX_polyLengthDist.png(*the distribution of length of polysome from classXXX*)
       
