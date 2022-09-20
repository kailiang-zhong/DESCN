## DESCN

这是关于论文 [DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation](https://arxiv.org/abs/2207.09920) 的实现, 该论文已被 [SIGKDD 2022](https://kdd.org/kdd2022/) Applied Data Science Track
接收。 

DESCN 要解决的问题是预估binary的干预对结果的增量影响，比如对一个用户给或不给优惠券，增量购买率会有多大影响。
#### 相关概念
- Treatment effect: 干预所带来的影响。比如给一个用户发优惠券，treatment effect就是优惠券这个干预带来的增量购买概率。其他常见的干预有某项政策实施、药物施加等。
- Control group: 对照组，表示无施加干预观测到的样本
- Treatment group: 干预组，表示有施加干预观测到的样本
- The counterfactual:我们永远只能观测的其中一种事实（要么是干预的结果、要么是无干预的结果）


DESCN的思路受启发于[ESMM](https://arxiv.org/abs/1804.07931) 和[X-learner](https://arxiv.org/abs/1706.03461) ：从端到端角度对treatment-effects进行建模， 通过多任务学习的方式捕获倾向性（propensity）、响应函数（response）以及潜在的treatment effect信号。

![](images/ESN_Xnetwork_DESCN.jpg)



## 复现实验
实验执行过程和结果都在 `DeepModels_real_data.ipynb` 文件里，该文件包含的数据预处理代码会把`.csv` 格式的数据集做了采样并以`.npz` 格式导出. 
    
所有实验都使用了GPU做训练和预测，版本： `CUDA Verson:11.4`
## 代码使用方法
`main.py`: 所有模型的main文件（除了X-learner）  
`x_learner_main.py`:  X-learner(NN based)模型的main文件  
`eval4real_data.py`:  Lazada真实评估代码.  
`eval.py`: ACIC 虚拟数据集的评估代码.  
`search_parames.py`: 一个串行调度工具，可以通过命令行指定模型参数配置文件、训练集、预测集、评估代码，方便地完成训练-预测-评估全流程。

search_parames.py 工具调用例子:\
```python search_params.py main.py eval4real_data.py  ./conf4models/lzd_real_data/DESCN.txt 1 {path_to_train_npz} {path_to_test_npz}```\
更多例子可以查看 `DeepModels_real_data.ipynb`


## 模型参数配置
所有模型超参数配置在 `./conf4models`文件夹中.

注意:
>- 所有关于路径的配置项必须填写绝对路径
>- `./results/lzd_real` and `./runs` should be created. 

## 数据集
`./data` 包含了Lazada 电商平台的真实数据集

## Python 包版本
- python == 3.7

其他python包版本记录在 `requirements.txt`.
