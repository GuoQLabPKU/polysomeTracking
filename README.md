codes for track polysomes from tomograms

**py_***: the functions for polysomes tracing***

#tutorial_readme.ipynb: an example to call functions

#polysome_class: the polysome Class

Requirement:
Modules:

python: 3.6.7 (3.7.9/3.8.3)

numpy: 1.16.1 (1.18.5/1.19.2)

pandas: 1.1.1 (1.0.5/1.2.1)

jupyter: 1.0.0

matplotlib: 3.3.3

cupy: 9.0.0

cudatoolkit: 11.0.3 

pytest: 6.2.4

networkx:2.6.2

seaborn:0.11.2

dill:0.3.4

Flatform:
Any platform (linux/windows/macos)should be fine. But linux is largely tested to run these codes.

Test:
For test, please run pytest py_simulation/run_simPoly.py in the terminal 

Debugging:
If you meet an memory error from tom_pdist function, please reduce the input variable Chunk in function tom_memalloc (py_memory/tom_memalloc.py)

Visualization：
1. tree/linkage:  vis/clustering
2. transform pairs distance distribution: vis/distVSavg
3. noise estimation: vis/noiseEstimate
4. polysomes: vis/vectfields

