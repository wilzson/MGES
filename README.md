# MGES: Multi-Graph Collaborative Network for Chinese NER with Enhanced Semantic Dependency Feature Information 

Source code for Multi-Graph Collaborative Network for Chinese NER with Enhanced Semantic Dependency Feature Information.

# Requirements:

```
Python: 3.7
PyTorch: 1.12.1
HanLp: 2.1
```



Input format:
======

Input is in CoNLL format (We use BIO tag scheme), where each character and its label are in one line. Sentences are split with a null line.

```
叶 B-PER
嘉 I-PER
莹 I-PER
先 O
生 O
获 O
聘 O
南 B-ORG
开 I-ORG
大 I-ORG
学 I-ORG
终 O
身 O
校 O
董 O
。 O
```

Pretrained Embeddings:
====

Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec) can be downloaded in [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D).

Word embeddings (sgns.merge.word) can be downloaded in [Google Drive](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) or
[Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw).

Usage：
====

1.  Download the character embeddings and word embeddings and put them in the `data/embeddings` folder.
2. Modify the `config.py` by adding your train/dev/test file directory.
3. run `main.py`



Result：
====

For WeiboNER dataset, using the default hyperparameters in parameter.txt can achieve the state-of-art results (Test F1: 65.4%). 
