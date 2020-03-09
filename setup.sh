#!/bin/bash

# Setup a conda environment and install packages
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda create -n nopg python=3.6.8
conda activate nopg

pip install -e .

git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients quanser
cd quanser
pip install -e .
cd ..

conda deactivate
