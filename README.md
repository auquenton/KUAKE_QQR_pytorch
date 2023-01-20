# Medical search Query relevance judgment

## Question description
Correlation between queries (i.e., search terms) evaluates how well two queries match the topics expressed by them, that is, whether and to what extent Query-A and Query-B are escaped. The topic of Query refers to the focus of query, and determining the correlation between two query terms is an important task, which is often used in the search quality optimization scenario of long-tail query. This task data set is generated under such background.
<div align=center>

![examples](./pic/1.png)
</div>

## Dataset introduction

[Download](https://tianchi.aliyun.com/competition/entrance/532001/information)

The correlation between Query and Title is divided into three levels (0-2). 0 is the worst correlation, and 2 is the best correlation.

2 points: indicates that A and B are equivalent, the expression is completely consistent

1 score: B is the semantic subset of A, and B refers to A scope less than A

0 score: B is the semantic parent set of A, B refers to A range greater than A; Or A has nothing to do with B semantics

## Environment

```shell
pip install gensim
pip install numpy
pip install tqdm
conda install torch
pip install transformer
```

## Prepare
Download corpus from Tencent AI Lab
```shell
wget https://ai.tencent.com/ailab/nlp/zh/data/tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz # v0.2.0 100 demention-Small
```
Decompress the corpus
```shell
tar -zxvf tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz
```

## Train

```python
python train.py --model_name {model_name} --datadir {datadir} --epochs 30 --lr 1e-4 --max_length 32 --batch_size 8 --savepath ./results --gpu 0 --w2v_path {w2v_path}
```
Or run the scripts

```shell
sh scripts/train.sh
```

## Inference
```python
python inference.py --model_name {model_name} --batch_size 8 --max_length 32 --savepath ./results --datadir {datadir} --model_path {model_path} --gpu 0 --w2v_path {w2v_path}
```
Or run the scripts

```shell
sh scripts/inference.sh
```

## Results

<div align=center>

| Model | Params(M) | Train Acc(%) |Val Acc(%)|Test Acc(%)|
| :----:| :----: | :----: |:----:|:----:|
| SemNN |  |  |||
| SemLSTM |  |  |||
| SemAttention | |  |||
| Bert |  |  |||

</div>