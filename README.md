# Running the training

## Clone repo
```
cd <PATH-TO-WORKING-AREA>
git clone git@github.com:StealthStop/StandAloneNN.git
cd StandAloneNN
```

## Install anaconda (keeps needed python environment seperate)
Install the appropriate anaconda3 version for your machine.
You can see all options here: https://repo.anaconda.com/archive/

Here is an example for a Mac running python 3:
```
cd <PATH-TO-WORKING-AREA>/StandAloneNN
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
bash Anaconda3-2020.11-MacOSX-x86_64.sh <<< $'\nyes\n$PWD/anaconda3/\nyes\n'
rm Anaconda3-2020.11-MacOSX-x86_64.sh
source $PWD/anaconda3/bin/activate
#cat ~/.bash_profile #Note anaconda likes to update this file. However, these updates are not needed.
```

Setup a new python area with tensorflow 1.10 for python 2.7.15.
* Note that this will not overwrite your base python area or the tensorflow version you may currently have.
* This will create a local python environment called tf that can be used solely for running this code
```
conda update -n base -c defaults conda <<< $'y\n'
conda create -n tf python=2.7.15 anaconda <<< $'y\n'
conda activate tf
conda install -n tf libgcc pandas "tensorflow==1.10" numpy protobuf pydot <<< $'y\n'
pip install uproot
pip install awkward
```

Download training files for each year
```
./getNNCfg.sh -t Keras_Tensorflow_2016_v1.2 -o -s 2016 -Q
./getNNCfg.sh -t Keras_Tensorflow_2017_v1.2 -o -s 2017 -Q
./getNNCfg.sh -t Keras_Tensorflow_2018pre_v1.2 -o -s 2018pre -Q
./getNNCfg.sh -t Keras_Tensorflow_2018post_v1.2 -o -s 2018post -Q
```

Running the NN example
```
python2 runNN.py
```

Deactivate the local python setup
```
conda deactivate
```
