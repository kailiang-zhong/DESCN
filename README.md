DESCN
-------
Implementation of our KDD2022 paper: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation


## Reproduce the experimental results
The execution process and results can be viewed in `DeepModels_real_data.ipynb` and `CasualForest_BART_real_data.ipynb`
## Code usage
1. `main.py`: main process for all models, except X-learner.
2. `x_learner_main.py`: main process for X-learner(NN based).
3. `eval4real_data.py`: evaluation process for LAZADA real dataset.
4. `eval.py`: evaluation process for ACIC2019 dataset.
5. `search_parames.py`: A tool for scheduling training, prediction and evaluation.
## Configuration
all models's hyper-parameters is save in `./conf4models`. 
`./conf` just contain a configuration temperature.

>note:
>1. All path related configuration options must be set an absolute path.
>2. `./results/lzd_real` and `./runs` should be created. 


## Python packages version
- python == 3.7

other python packages version are showed in `requirements.txt`.
