#!/bin/bash

## Install dependent packages 
lspci |grep -i nvidia
if [ $? -eq 0 ]; then
    `which conda` install --yes \
    cupy \
    cudatoolkit \
    

`which conda` install --yes \
--channel conda-forge \
--channel anaconda \
python==3.7 \
numpy \
pandas \
matplotlib \
pytest==6.0.0 \
networkx \
seaborn \
dill \
alive-progress \
psutil \
pynvml \

`which conda` clean --yes --all
