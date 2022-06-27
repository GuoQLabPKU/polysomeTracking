#!/bin/bash

## Install dependent packages 
`which conda` install --yes \
--channel conda-forge \
--channel anaconda \
python=3 \
numpy \
pandas \
matplotlib \
cupy \
cudatoolkit \
pytest \
networkx \
seaborn \
dill \
alive-progress \
psutil \

`which conda` clean --yes --all
