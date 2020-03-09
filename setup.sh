#!/bin/bash

# Setup a conda environment and install packages
conda create -n nopg python=3.6.8
conda activate nopg

python setup.py develop

git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients quanser
cd quanser
python setup.py develop
cd ..

conda deactivate
