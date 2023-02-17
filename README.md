# Multi-News adaptation of THExt 



### Intro 

This repository contains the code for our Multi-News adaptation of THExt, realized for Deep Natural Language Process class(2022-2023). 
Our task was to adapt THExt to another domain and extend it to Multi-Document summarization. 
In this work, we will try to generate highlights for news articles, useful to be used as subtitles or key phrases to catch the attention of the reader. 
We also included an ablation study that consists in generate a different context from different sections of the paper, in order to extend the ones given to the original THExt. 

The pipeline we propose, and you can reproduce, is the following:

Data preprocessing
Extractor, Abstractor models training
Model evaluation

### Dependencies
* Python 3 (tested on python 3.6)
* PyTorch
  * with GPU and CUDA enabled installation (though the code is not runnable on CPU)
* TensorFlow
* pyrouge (for evaluation)

### Dataset Download 
You can download the complete Multi-News dataset at the following link: 
* https://github.com/Alex-Fabbri/Multi-News

### Execution Guide

If you like, two colab notebooks with the most salient steps is provided at: 

1. *Multi-News Adaptation*: [![Multi-News Adaptation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1exznryjeKoObylxIuFAe0tV4qMtLle9U)

2. *New Context Generation*: [![New Contexts Generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fW9SRakKl3uGFOiYlUwaq2kTo96_Xl0s)
