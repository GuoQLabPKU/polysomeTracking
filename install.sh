#!/bin/bash

## Installing conda packages (requires Anaconda with Python >= 3.6 )
conda install --yes \
numpy \
pandas \
matplotlib \
cupy \
cudatoolkit \
pytest \
networkx \
seaborn \
dill \

conda clean --yes --all
