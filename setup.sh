# Setup a python virtual environment and install packages
rm -rf venv
virtualenv -p python3.6 venv
source venv/bin/activate

python setup.py develop

git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients quanser
cd quanser
python setup.py develop
cd ..


deactivate
