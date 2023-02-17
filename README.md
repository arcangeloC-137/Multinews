# Multi-document Summarization for News Articles Highlights Extraction

This repository contains the code for our Multi-News adaptation of paper Transformer-based Highlights Extraction - THExt, realized the Deep Natural Language Process class (2022-2023).

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
* [Multi-News dataset](https://github.com/Alex-Fabbri/Multi-News)
* AIPubSumm, CSPubSumm, and BIOPubSumm. These datasets are not publicly available: plese refere to [this repo](https://github.com/arcangeloC-137/THExt) for further information.

### Execution Guide

Two examples to run our work are available on colab at the following links: 

1. *Multi-News Adaptation*: [![Multi-News Adaptation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1exznryjeKoObylxIuFAe0tV4qMtLle9U)

2. *New Context Generation*: [![New Contexts Generation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fW9SRakKl3uGFOiYlUwaq2kTo96_Xl0s)

If the running is performed locally, please install the requirements in the file:

```bash
!pip install -r requirements.txt
```

## Pipeline Description

<p>
    <img src="https://github.com/arcangeloC-137/Multinews/blob/main/imgs/Multi-Document%20THExt%202.png" alt>
    <em>image_caption</em>
</p>

<div align="center">
  <img src="https://github.com/arcangeloC-137/Multinews/blob/main/imgs/Multi-Document%20THExt%202.png" alt="Alt text" title="Preprocessing pipeline" width="500" height="300">
</div>
