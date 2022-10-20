# Selective compression learning of latent representations for variable-rate image compression
Repository of the paper "Selective compression learning of latent representations for variable-rate image compression"

>|![Samples](https://drive.google.com/file/d/1L6_xHBykTeOR_aLZozas7SBwIE7bi_6b/view?usp=sharing)|
>|:--:|
>| *Test results over the Tecnick SAMPLING imageset* |

## Introduction

This repository includes sample source codes (MSE-optimized on the ["Hyperprior"](https://arxiv.org/abs/1802.01436) model) of our paper ["Selective compression learning of latent representations for variable-rate image compression"](https://openreview.net/forum?id=xI5660uFUr). Please refer to [the paper](https://openreview.net/forum?id=xI5660uFUr) for the detailed information. If  [the paper](https://openreview.net/forum?id=xI5660uFUr) or this repository helps you, please cite our work as:

~~~
@inproceedings{
lee2022selective,
title={Selective compression learning of latent representations for variable-rate image compression},
author={Jooyoung Lee and Seyoon Jeong and Munchurl Kim},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=xI5660uFUr}
}
~~~

## Required packages (and their tested versions)
* python (3.6.8)
* tensorflow-gpu (1.15.0)
* tensorflow-compression (1.3)
* pillow (4.2.1)
* scipy (0.19.1)
* tqdm (4.47.0)

## Training
Our SCR model is trained in a step-wise manner as follows:
* Step 1: training of the base compression model
~~~
cd 1_hyperprior
python main.py --is_train True --quality_level 8
~~~

* Step 2: training of the SCR model without the selective compression components
~~~
cd ..
cd 2_SCR_wo_SC
python main.py --is_train True
~~~

* Step 3: training of the SCR full model
~~~
cd ..
cd 3_SCR_full
python main.py --is_train True
~~~
