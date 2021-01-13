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

Expected output using test.root
```
Loading model...
Reading config file
MVA bin: 2      NN Score: 0.40781268
MVA bin: 3      NN Score: 0.84303075
MVA bin: 2      NN Score: 0.777397
MVA bin: 4      NN Score: 0.96657974
MVA bin: 2      NN Score: 0.5619648
MVA bin: 2      NN Score: 0.46352914
MVA bin: 4      NN Score: 0.9934535
MVA bin: 2      NN Score: 0.7719398
MVA bin: 3      NN Score: 0.8518663
MVA bin: 4      NN Score: 0.97128975
MVA bin: 4      NN Score: 0.9922706
MVA bin: 4      NN Score: 0.963658
MVA bin: 4      NN Score: 0.98064995
MVA bin: 4      NN Score: 0.93071026
MVA bin: 1      NN Score: 0.2455785
MVA bin: 2      NN Score: 0.43037787
MVA bin: 4      NN Score: 0.99406576
MVA bin: 3      NN Score: 0.8494306
MVA bin: 4      NN Score: 0.97410357
MVA bin: 2      NN Score: 0.63021994
```


Deactivate the local python setup
```
conda deactivate
```
