
**Cartoon-Texture Guided Network for Low-Light Image Enhancement**

## Environment

* NVIDIA GTX 3080Ti GPU.
* CUDA 11.3.1
* python 3.8
* torch  1.10.1


### Dataset
You can refer to the following links to download the datasets
[LOL](https://daooshee.github.io/BMVC2018website/), and
[VE-LOL](https://flyywh.github.io/IJCV2021LowLight_VELOL/).

You can refer to the following links to produce the datasets
[ground-truth cartoon images and ground-truth texture images] (https://github.com/Zhiyuan-Zhang510zg/CLRP), where ground-truth images are sourced from the LOL dataset.


###Testing

test.py

###Training
Step:

1. First, make normal exposure cartoon and texture images based on the LOL dataset.

2. Secondly, the pre-trained model of the cartoon recovery module is trained according to the cartoon dataset of normal exposure generated.

3. The generated cartoon pre-training model is imported into the texture recovery module to train the whole.