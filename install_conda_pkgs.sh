#!/bin/bash

## Install dependent packages 
`which conda` install --yes \
--channel conda-forge \
--channel anaconda \
python==3.7 \
numpy \
pandas \
matplotlib \
cupy \
cudatoolkit \
pytest==6.0.0 \
networkx \
seaborn \
dill \
alive-progress \
psutil \
pynvml \

`which conda` clean --yes --all
