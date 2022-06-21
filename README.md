# MADA

This is an official pytorch implementation of our <strong>Interspeech, 2022</strong>  paper [Multilingual and Multimodal Abuse Detection](https://arxiv.org/pdf/2204.02263.pdf). 

In this repository, we provide the features and codebase for reproducing our work. 

## DOWNLOAD INSTRUCTIONS

### Dataset
Please download the train/validation splits of ADIMA dataset in the ```annotations``` folder from [here](https://drive.google.com/drive/folders/18U8penuxlxqMmeuatyqeLBZfkMHp2SjK?usp=sharing).

### Features
Please download the pre-extracted audio, emotion and textual features in the ```features``` folder from [here](https://drive.google.com/file/d/1DXTiCT9eWjoIrQEKs27JNPyfclM7Effi/view?usp=sharing).

## Training the models:

```
bash train.sh
```

## Installation:

```
conda env create -f environment.yml
```

## BibTeX Citation

If you find MADA useful in your research, please use the following BibTeX entry for citation.

```
@article{sharon2022multilingual,
  title={Multilingual and Multimodal Abuse Detection},
  author={Sharon, Rini and Shah, Heet and Mukherjee, Debdoot and Gupta, Vikram},
  journal={arXiv preprint arXiv:2204.02263},
  year={2022}
}
```
