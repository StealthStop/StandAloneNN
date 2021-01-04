## Running the training

#Install anaconda (keeps needed python environment seperate)
Install the appropriate anaconda3 version for your machine.
You can see all options here: https://repo.anaconda.com/archive/

Here is an example for a Mac running python 3:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
bash Anaconda3-2020.11-MacOSX-x86_64.sh <<< $'\nyes\n$PWD/anaconda3/\nyes\n'
rm Anaconda3-2020.11-MacOSX-x86_64.sh
source $PWD/anaconda3/bin/activate
```

Setup a new python area with tensorflow 1.10 for python 2.7.15 
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
./getNNCfg.sh -t Keras_Tensorflow_2016_v1.2 -o -s 2016
./getNNCfg.sh -t Keras_Tensorflow_2017_v1.2 -o -s 2017
./getNNCfg.sh -t Keras_Tensorflow_2018pre_v1.2 -o -s 2018pre
./getNNCfg.sh -t Keras_Tensorflow_2018post_v1.2 -o -s 2018post
```

