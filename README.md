# Selective compression learning of latent representations for variable-rate image compression
Repository of the paper ["Selective compression learning of latent representations for variable-rate image compression"](https://openreview.net/forum?id=xI5660uFUr)

| ![Samples](./Kodim15_animation.gif) |
|:--:|
| *Sample reconstructions and their corresponding 3D binary masks* |

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
python main.py --is_train True --quality_level 8 --input_dataset [INPUT_DATASET_PATH]
~~~

* Step 2: training of the SCR model without the selective compression components
~~~
cd ..
cd 2_SCR_wo_SC
python main.py --is_train True --input_dataset [INPUT_DATASET_PATH]
~~~

* Step 3: training of the SCR full model
~~~
cd ..
cd 3_SCR_full
python main.py --is_train True --input_dataset [INPUT_DATASET_PATH]
~~~

INPUT_DATASET_PATH indicates the path of directory that contains the 256x256-sized patches of the trainset images.

## Test
* For the base compression model
~~~
cd 1_hyperprior
python main.py --is_test True --quality_level 8 --testset_path [TESTSET_PATH]
~~~

* For the SCR model without the selective compression components
~~~
cd ..
cd 2_SCR_wo_SC
python main.py --is_test True --quality_level [QUALITY_LEVEL] --testset_path [TESTSET_PATH]
~~~

* For the SCR full model
~~~
cd ..
cd 3_SCR_full
python main.py --is_test True --quality_level [QUALITY_LEVEL] --testset_path [TESTSET_PATH]
~~~

TESTSET_PATH indicates the path of the directory that contains the testset images such as the 24 Kodak imageset PNG files.
The bpp and PSNR(MS-SSIM) results are saved as corresponding CSV files in the "logs" directory. Note that the last rows in the CSV files indicate average values.

* TESTSET_PATH must end with '/'.
* QUALITY_LEVEL can be a floating point number between 1 to 8.

## To use the pretrained models
Please follow the following steps to use the pretrained models:
* Dowload the pretrained models (or one of the models) from [here](https://drive.google.com/drive/folders/1KbTZBcJggrnBaZddKrWiN7S60cEtD9U6?usp=sharing).
* Unpack the downloaded zip files.
* Move the unpacked "logs" directories into the corresponding model directories.
* Now you can test with the pretrained models or use them as pretrained models for the next step of the training.

## Pytorch version
The Pytorch version of our SCR is available at [https://github.com/swimmiing/SCR-Torch](https://github.com/swimmiing/SCR-Torch). We deeply appreciate their efforts.
