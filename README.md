DESCN
-------
Implementation of paper [DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation](https://arxiv.org/abs/2207.09920), which is accepted in [SIGKDD 2022](https://kdd.org/kdd2022/) Applied Data Science Track.
![](images/ESN_Xnetwork_DESCN.jpg)

## Reproduce the experimental results
The execution process and results can be viewed in `DeepModels_real_data.ipynb` which contains the code for converting dataset from `.csv` format to `.npz` format. 
    
All experiments uses GPU for training and `CUDA Verson:11.4`
## Code usage
`main.py`: main process for all models, except X-learner.  
`x_learner_main.py`: main process for X-learner(NN based).  
`eval4real_data.py`: evaluation process for LAZADA real dataset.  
`eval.py`: evaluation process for ACIC2019 dataset.  
`search_parames.py`: A tool for scheduling training, prediction and evaluation.

A convenient way to call training and evaluation code serially:\
```python search_params.py main.py eval4real_data.py  ./conf4models/lzd_real_data/DESCN.txt 1 {path_to_train_npz} {path_to_test_npz}```\
More examples can be referred to code in `DeepModels_real_data.ipynb`


## Configuration
All models's hyper-parameters are saved in `./conf4models`.

NOTE:
>- All **path related configuration options** must be set an absolute path.
>- `./results/lzd_real` and `./runs` should be created. 

## Dataset
`./data` contain the real-world Production Dataset from E-commerce platform Lazada.

## Python packages version
- python == 3.7

other python packages version are showed in `requirements.txt`.
