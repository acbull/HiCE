## Overview

HiCE (Hierarchical Context Encoding) is a model for learning accurate embedding of an OOV word with few occurrences. This repository is a pytorch implementation of HICE.

The basic idea is to train the model on a large scale dataset, masking some words out and use limited contexts to estimate their ground-truth embedding. The learned model can then be served to estimate OOV words in a new corpus. The model can be furthered improved by adapting to the new corpus with first order MAML (Model-Agnostic Meta-Learning).

You can see our ACL 2019 paper [“**Few-Shot Representation Learning for Out-Of-Vocabulary Words**”](https://arxiv.org/abs/1907.00505) for more details.

## Setup
This implementation is based on Pytorch We assume that you're using Python 3 with pip installed. To run the code, you need the following dependencies:

- [Pytorch 1.0](https://pytorch.org/)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [tqdm](https://github.com/tqdm/tqdm)

For fair comparison with earlier works, we utilize the same word embedding provided by [Herbelot & Baroni, 2017](https://drive.google.com/open?id=1Np4J0q_ZIYr74IIdLGlAgzwwx00KJJeE), which is a 259,376 word2vec embedding pre-trained on Wikipedia. After downloading it, unzip and put it into '/data/' directory.

To fit this word embedding, we use WikiText-103 as source corpus to train our model. Download [WikiText-103](https://drive.google.com/open?id=1Spu89EmhxKYJiyDVHfRa_P6aFoA-FuCd), unzip and put it into the '/data/' directory. 

## Usage
Execute the following scripts to train and evaluate the model:

```bash
python3 train.py --cuda 0 --use_morph --adapt  # Train HiCE with morphology feature and use MAML for adaptation
python3 train.py --cuda 0 --use_morph          # Train HiCE with morphology feature and no adaptation
python3 train.py --cuda 0 --adapt              # Train HiCE with context only without morphology and use MAML for adaptation
python3 train.py --cuda 0                      # Train HiCE with context only without morphology and no adaptation
```
There's also other hyperparameters to be tuned, which can be found in 'train.py' for details.

The model will parse the training corpus in a way that some words (which frequency is not too high or too low) are selected as OOV words, with the sentences containing these words as features and ground-truth embedding as the label. For each batch, the model will randomly select some words with K context sentences to estimate the ground-truth embedding. The model will be evaluated on ['Chimera dataset' (Lazaridou et al, 2017)](https://www.ncbi.nlm.nih.gov/pubmed/28323353). 

After finish training, the model can further be adapted to the target corpus with first order MAML. We also use the known words in the target corpus as OOV words and construct a target dataset. Then we use the better initialization get from source dataset to calculate the gradient on target dataset. Noted that this is not equivalent to the original definition of MAML, where there exist multiple tasks. If one can get access to multiple datasets in different domains, the model can also be trained in the original paper's style.

The trained model will be saved in a given directory (default in '/save' directory), which can be adopted to handle OOV in other downstream tasks.

### Citation

Please consider citing the following paper when using our code for your application.

```
@inproceedings{hice2019,
  title={Few-Shot Representation Learning for Out-Of-Vocabulary Words},
  author={Ziniu Hu and Ting Chen and Kai-Wei Chang and Yizhou Sun},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, ACL},
  year={2019}
}
```
