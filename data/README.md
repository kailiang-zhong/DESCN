
# The real-world Production Dataset from E-commerce platform Lazada

You could download the dataset using([Google Drive](https://drive.google.com/file/d/19iSXsbRXJWvuSFHdcLb0Vi9JCP9Fu41s/view?usp=sharing ) | [Baidu Drive](https://pan.baidu.com/s/1CKJvzow7UFGwrdXbkt1mQA) with code: 75hr | [Dropbox](https://www.dropbox.com/s/07r7592h9mfijsb/lzd_data_public.zip?dl=0))


## File structure:

```python
lzd_data_public/
    full_trainset.csv # the training set which contains treatment bias.
    full_testset.csv # the test set which is randomized.
```

## Information fingerprint of data files
To ensure that data files is not tampered with.
```shell
MD5 (lzd_data_public.zip) = 545bb0eae05cca58f0d813a990ed9cba 
MD5 (full_trainset.csv) = efffc70375e700bcb91a9d41eddd7f66
MD5 (full_testset.csv) = 9058fa57b984f0112a1ecf71dc52fb1e
```
## Data description
The trainning set: `full_trainset.csv`

| group        |   num   |     sum(label) |   E(label=1) |
|--------------|:-------:|---------------:|-------------:|
| treatment    | 205480  |          11638 |5.66%|
| no treatment | 721189  |           6810 |0.94%|
  
&nbsp;

The test set: `full_testset.csv`

| group        |   num   |     sum(label) | E(label=1) |
|--------------|:-------:|---------------:|-----------:|
| treatment    | 94682  |          3503 |      3.70%|
| no treatment | 86987  |           2893 |      3.33% |

