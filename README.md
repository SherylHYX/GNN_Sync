# GNN_Sync
Group synchronization with graph neural networks.

--------------------------------------------------------------------------------

## Environment Setup
The codebase is implemented in Python 3.7. package versions used for development are below.
```
networkx                        2.6.3
numpy                           1.20.3
scipy                           1.7.1
argparse                        1.1.0
scikit-learn                    1.0.1
torch                           1.10.1
pyg                             2.0.3 
torch-geometric-signed-directed 0.20.0
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use [GNU parallel](https://www.gnu.org/software/parallel/), which can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets.

- ./src/ stores files to train various models, utils and metrics.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./logs/ stores trained models and logs, as well as predicted clusters (optional). When you are in debug mode (see below), your logs will be stored in ./debug_logs/ folder.

## Options
<p align="justify">
GNNSync provides various command line arguments, which can be viewed in the ./src/param_parser.py. Some examples are:
</p>

```
  --epochs                INT         Number of GNNSync (maximum) training epochs.              Default is 1000. 
  --early_stopping        INT         Number of GNNSync early stopping epochs.                  Default is 200. 
  --num_trials            INT         Number of trials to generate results.                     Default is 2.
  --lr                    FLOAT       Learning rate.                                            Default is 0.005.  
  --weight_decay          FLOAT       Weight decay (L2 loss on parameters).                     Default is 5^-4. 
  --upset_coeff           FLOAT       Upset loss coefficient.                                   Default is 1.  
  --cycle_coeff           FLOAT       Cycle loss coefficient.                                   Default is 0.  
  --dropout               FLOAT       Dropout rate (1 - keep probability).                      Default is 0.5.
  --hidden                INT         Number of embedding dimension divided by 2.               Default is 8. 
  --no-cuda               BOOL        Disables CUDA training.                                   Default is False.
  --debug, -D             BOOL        Debug with minimal training setting, not to get results.  Default is False.
  --dataset               STR         Data set to consider.                                     Default is 'ERO/'.
  --all_methods           LST         Methods to use to generate results.                       Default is ['GNNSync'].
```


## Reproduce results
First, get into the ./execution/ folder:
```
cd execution
```
To reproduce baseline results.
```
bash baselines.sh
```
To reproduce the main results on angular synchronization with k=1.
```
bash GNNSync_sync.sh
```
To reproduce the main results on general k-synchronization with k>1.
```
bash GNNSync_ksync.sh
```

Note that if you are operating on CPU, you may delete the commands ``CUDA_VISIBLE_DEVICES=xx". You can also set you own number of parallel jobs, not necessarily following the j numbers in the .sh files, or use other GPU numbers.

You can also use CPU for training if you add ``--no-duca", or GPU if you delete this.

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```

Then, below are various options to try:

Creating a GNNSync model for ERO data with N=360 nodes, noise level eta=0.1, density parameter p=0.05 on angular synchronization.
```
python ./train.py --all_methods GNNSync --dataset ERO --N 360 --eta 0.1 --p 0.05
```
Creating a GNNSync model for RGGO data with N=360 nodes, noise level eta=0.5, density parameter p=0.15 on k-synchronization with k=3, with cycle loss only.
```
python ./ksync_train.py --all_methods GNNSync --dataset RGGO --N 360 --eta 0.5 --p 0.15 --k 3 --upset_coeff 0 --cycle_coeff 1
```
Creating a GNNSync model for BAO data with specific number of trials, hidden units and use CPU, on angular synchronization.
```
python ./train.py --dataset BAO --no-cuda --num_trials 5 --hidden 8 --all_methods GNNSync
```
--------------------------------------------------------------------------------