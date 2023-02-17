# Multi-document Summarization for News Articles Highlights Extraction

This repository contains the code for our Multi-News adaptation of paper [Transformer-based Highlights Extraction from scientific papers - THExt](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006931), realized for the Deep Natural Language Processing class (A.Y. 2022-2023).

The extraction of highlights is a process which consists in selecting the salient sentences within the body of a test, which well summarize the meaning of the text under examination. This paper focuses on the problem of extracting highlights from news articles using transformer-based techniques. 
Our task was to adapt THExt to another domain and extend it to Multi-Document summarization.

We also propose an ablation study, which aims to improve the baseline providing a generated context through the use of LED architecture. The results obtained in the multi-document and in the ablation study, achieved respectively on two distinct datasets, have demonstrated the effectiveness of the model in the field of multi-document summarization, and its adaptability to a different domain, as well as the possibility of its improvement by providing a broader context.

The pipeline we propose, and that you can reproduce, is the following:

- Data preprocessing
- Extractor, Abstractor Models training
- Model evaluation

### Dependencies
* Python 3 (tested on python 3.6)
* PyTorch
  * with GPU and CUDA enabled installation (though the code is not runnable on CPU)
* TensorFlow
* pyrouge (for evaluation)

### Dataset Download 
The dataset exploited are the following: 
1. [Multi-News dataset](https://github.com/Alex-Fabbri/Multi-News)
2. AIPubSumm, CSPubSumm, and BIOPubSumm. These datasets are not publicly available, so some directory folders may be created: plese refere to [this repo](https://github.com/arcangeloC-137/THExt) for further information.

### Execution Guide

Two examples to run our work are available on colab at the following links: 

1. *Multi-News Adaptation*: [![Multi-News Adaptation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1exznryjeKoObylxIuFAe0tV4qMtLle9U)

2. *New Context Generation*: [![New Contexts Generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fW9SRakKl3uGFOiYlUwaq2kTo96_Xl0s)

If the running is performed locally, please install the requirements in the file:

```bash
!pip install -r requirements.txt
```

## Pipeline Description

### 1. Multi-document Summarization Pipeline
Firstly, we preprocess the input news articles, from dataset [1], using techniques such as stopword removal, sentence segmentation, and separation of articles of the same cluster, which enhances the performance of the summarization model. During this stage a tokenized unique text is created merging the texts of all the articles related to the same news. Also a new context is defined, merging the first 20\% of sentences of each article for the same cluster of news.

<div align="center">
  <img src="https://github.com/arcangeloC-137/Multinews/blob/main/imgs/Multi-Document%20THExt%202.png" alt="Alt text" title="Preprocessing pipeline" width="500" height="300">
</div>

Next, we fine-tune the chosen models, BERT and LongFormer, on the Multi-News dataset, which helps obtain the optimized weights for the models. Finally, we use a fully connected layer-based regression to produce the resulting highlights for the cluster of articles.


### 2. Context Generator Pipeline

The LED model, which is an extension of the Longformer, was employed to create a new context from the dataset [2]. Specifically, the study performed an abstractive summarization of different sections of the papers, namely the abstract, introduction, and results

<div align="center">
  <img src="https://github.com/arcangeloC-137/Multinews/blob/main/imgs/Context_generation.png" alt="Alt text" title="Preprocessing pipeline" width="700" height="300">
</div>

<p>
 
 Please, for further information, and to see our results, refere to the relative [`paper`](https://github.com/arcangeloC-137/Multinews/blob/main/Multi-document%20Summarization%20for%20News%20Articles%20Highlights%20Extraction.pdf).
 
 ----
  
 Authors: [`Matteo Berta`](https://github.com/MatteoBerta), [`Arcangelo Frigiola`](https://github.com/arcangeloC-137), [`Francesco Marigioli`](https://github.com/FrancescoMarigioli98), [`Luca Varriale`]("")
