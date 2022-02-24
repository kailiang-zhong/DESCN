DESCN
-------
Implementation of our KDD2022 paper: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation


## Reproduce the experimental results
The execution process and results can be viewed in `DeepModels_real_data.ipynb` and `CasualForest_BART_real_data.ipynb`    
All experiments uses GPU for training and `CUDA Verson:11.4`
## Code usage
`main.py`: main process for all models, except X-learner.  
`x_learner_main.py`: main process for X-learner(NN based).  
`eval4real_data.py`: evaluation process for LAZADA real dataset.  
`eval.py`: evaluation process for ACIC2019 dataset.  
`search_parames.py`: A tool for scheduling training, prediction and evaluation.
## Configuration
all models's hyper-parameters are saved in `./conf4models`  
And `./conf` just contain a configuration temperature.

NOTE:
>- All **path related configuration options** must be set an absolute path.
>- `./results/lzd_real` and `./runs` should be created. 

## Dataset
`./data` contain the real-world Production Dataset from E-commerce platform Lazada.

## Python packages version
- python == 3.7

other python packages version are showed in `requirements.txt`.
