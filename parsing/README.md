# Parsing with Structured UNN

## Data

Download training, validation and test sets from Universal Dependencies. Create a folder `parsing/data` and copy the necessary `.conllu` files there.

## Requirements

In addition to the main requirements described in the main README, there are some specific steps that need to be performed in order to run the parsing experiments:

### Install `logdecomp`

* Clone the repository logdecomp

```
git clone https://github.com/ltl-uva/logdecomp/

cd logdecomp
```

* Change the branch 

```
git checkout -b 64bit
```

* Download [Eigen](https://gitlab.com/libeigen/eigen)

Set the enviroment variable to point to the Eigen installation.

```
export EIGEN_DIR=local_path_to_eigen
```


* Install logdecomp from source code:

```
cd logdecomp

pip install .
```


### Install LP-SparseMAP

* Go to https://github.com/deep-spin/lp-sparsemap and follow the installation instructions for compilatoin form source.

* Install Cython 

```
pip install Cython
```

* Download Eigen and set the environment variable `EIGEN_DIR` (se above).

* Install LP-SparseMAP from source:

```
cd lp-sparsemap

python setup.py build_clib

pip install -e . 
```


### Install Torch-Struct

* Go to https://github.com/harvardnlp/pytorch-struct

* Follow the installation instructions.

```
pip install -qU git+https://github.com/harvardnlp/pytorch-struct
```




## Config

In order to run the parsing experiments, you need to setup the configuration files. See examples of `.yaml` files in directory `parsing/parser/configs`. 
When running the script, pass a parameter `-baseconf=path_to_config_file`.
New configuration parameters can be added in `conf.py`.

## Saving the models

Create a folder `parsing/parser/saved_models`. The best checkpoint (the one with highest unlabeled attachment accuracy (UAS) on the validation set) will be saved for the wandb run.

## Running

In order to execute the parsing experiments, run the file `bert_dep.py` with the config parameters:

`python bert_dep.py baseconf=configs/$lang.yaml unn_iter=$k`

`basefconf` specifies the path to the default config file, other configuraiton parameters can be passed to override the baseconf ones.

See for example file `run_unn.sh`.


## Evaluation

The best checkpoint for each trained model is saved in directory `saved_models`. In order to evaluate the model on the test set, run the script `eval_from_checkpoint.py` and for all models - `eval_from_checkpoint_all.py`

