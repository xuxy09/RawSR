# Exploiting Raw Images for Real-Scene Super-Resolution

This repository is for the rawSR algorithm introduced in the TPAMI paper [*Exploiting raw images for real-scene super-resolution*](https://arxiv.org/pdf/2102.01579.pdf).

Conference version: [Towards real scene super-resolution with raw images, CVPR 2019](https://arxiv.org/abs/1905.12156)

[Paper](https://arxiv.org/pdf/2102.01579.pdf), [Project](https://sites.google.com/view/xiangyuxu/rawsr_pami)

## Contents

1. [Environment](#1)
2. [Introduction](#2)
3. [Train](#3)
4. [Test](#4)
5. [Results](#5)
6. [Reference](#6)


<h3 id="1">Environment</h3>
Our model is trained and tested through the following environment on Ubuntu:

* Python: v2.7.5 with following packages:

    * tensorflow with gpu: v1.9.0
    
    * rawpy: v0.12.0
    
    * numpy: v1.15.3
    
    * scipy: v1.1.0

<h3 id="2">Introduction</h3>
Super-resolution is a fundamental problem in computer vision which aims to overcome the spatial limitation of camera
sensors. While significant progress has been made for single image super-resolution, most existing algorithms only perform well on
unrealistic synthetic data, which limits their applications in real scenarios. In this paper, we study the problem of real-scene single
image super-resolution to bridge the gap between synthetic data and real captured images. Specifically, we focus on two problems of
existing super-resolution algorithms: first, lack of realistic training data; second, insufficient utilization of the information recorded by
cameras. To address the first issue, we propose a new pipeline to generate more realistic training data by simulating the imaging
process of digital cameras. For the second problem, we develop a two-branch convolutional neural network to exploit the
originally-recored radiance information in raw images. In addition, we propose a dense channel-attention block for better image
restoration as well as a learning-based guided filter network for more effective color correction. Our model is able to generalize to
different cameras without deliberately training on images from specific camera types. Extensive experiments demonstrate that the
proposed algorithm can help recover fine details and clear structures, and more importantly, achieve high-quality results for single
image super-resolution in real scenarios.

![branch1](http://ww1.sinaimg.cn/large/008cBKqfly1gjxxavsuqlj30y009tab3.jpg)
Fig. 5. The image restoration branch adopts the proposed DCA blocks in an encoder-decoder framework and reconstructs high-resolution linear
color measurements ![](http://latex.codecogs.com/gif.latex?\\widehat{X}_{lin}) from the degraded low-resolution raw input $`X_{raw}`$.


![branch2](http://ww1.sinaimg.cn/large/008cBKqfly1gjxysiy5epj30g804wglq.jpg)

Fig. 7. Network architecture of the color correction branch. Our model
predicts the pixel-wise transformations A and B with a reference color
image.


<h3 id="3">Train</h3>

* Prepare training data
    1. Download the raw super-resolution dataset (13040 training and 150 validation images) from [Google Drive][[x2](https://drive.google.com/file/d/1U0EvzwAB7Dq7bLeit595gNpEKU4ya0wl/view?usp=sharing)][[x4](https://drive.google.com/drive/folders/1JQN8rKEHiq19RFxzNGOa4SiasFR1vb4g?usp=sharing)]
    2. Place the downloaded dataset to './Dataset'

* Begin to train
    1. (optional) Download the pretrained  weights and place them in './log_dir'
    2. Run the following script to train our model:
        ```
        python train_and_test.py
        ```
    3. For different purposes, you can access './parameters.py' to change parameters according to the annotations. The default setting is to train the model from 0 epoch without pretrained weights, and the validation images will be leveraged to test model performance per 10 epochs. 

<h3 id="4">Test</h3>

* Prepare testing data
    * Synthetic data
        1. Download the synthetic testing dataset (150 images) from [Google Drive]
        [[x2](https://drive.google.com/open?id=1hoXGO_4vWRmRFoMIiQ32KwN_12kgNn7j)]
        [[x4](https://drive.google.com/drive/folders/1GB1QdPOQaW9iU-zdDXaTeesfEGvCpPj4?usp=sharing)]]  
        [BaiduNetdisk][[x2](https://pan.baidu.com/s/1z972Ic5X3zmMdwkMeOwA2w)]
        2. Place the downloaded dataset to './Dataset/' with folder name 'TESTING' or modify the 'TESTING_DATA_PATH' of parameters.py  to the corresponding path.
    
    * Real data
        1. If you wish to test real data, you can prepare the raw image yourself, or download some examples from 
        [[Google Drive](https://drive.google.com/drive/folders/1EfRQV0Cvn1JFl1XGG3r3r9iHPvplmB7L?usp=sharing)].
        2. Place the downloaded dataset or your prepared raw images (like .CR, .RAW, .NEF and etc.) to './Dataset/' with folder name 'REAL' or modify the 'REAL_DATA_PATH' of 'parameters.py' to corresponding path.
    
* Begin to test
    * Synthetic data
        1. Set 'TRAINING' and 'TESTING' of 'parameters.py' to be False and True respectively.
        2. Download the pretrained models through [Google Drive] [[x2](https://drive.google.com/drive/folders/1l91w51ou-p_2cVVbUCWDLGRUv_twnWcd?usp=sharing)], 
        [[x4](https://drive.google.com/drive/folders/1ZCp22cjZrKrQEoLC70YGnf8JbOGyw53P?usp=sharing)], and then place it to './log_dir'.
    
    * Real image
        1. Set 'REAL' of 'parameters.py' to be True.
        2. Download the pretrained models through [Google Drive] [[x2](https://drive.google.com/drive/folders/1l91w51ou-p_2cVVbUCWDLGRUv_twnWcd?usp=sharing)], [[x4](https://drive.google.com/drive/folders/1ZCp22cjZrKrQEoLC70YGnf8JbOGyw53P?usp=sharing)], and then place it to './log_dir'.
    
        And then, run the following script for testing:
            ```
            python train_and_test.py
            ```
        
        Notice, the testing results can be found in the path defined in 'RESULT_PATH' in 'parameters.py'.

<h3 id="5">Results</h3>

* Quantitative comparisons

    ![](http://ww1.sinaimg.cn/large/008cBKqfly1gjxwovx9i1j30cj05875o.jpg)

    TABLE 1
Quantitative evaluations on the synthetic dataset. “Blind” represents the
images with variable blur kernels, and “Non-blind” denotes fixed kernel.
    
* Visual comparisons
    * Synthetic data
    ![resSyn](http://ww1.sinaimg.cn/large/008cBKqfly1gjxwtfc9h8j30vk0hj1kx.jpg)
    Fig. 8. Results from the proposed synthetic dataset. References for the baseline methods including [SID[1]](#r1), [SRDenseNet[5]](#r5) and [RDN[6]](#r6), can be found in Table 1. “GT” represents ground truth.
    
    * Real data
    ![resReal](http://ww1.sinaimg.cn/large/008cBKqfly1gjxx8gdy2bj30uk0jmx2j.jpg)
    Fig. 11. Comparison with the state-of-the-arts on real-captured images. Since the outputs are of ultra-high resolution, spanning from 6048 × 8064 to
12416 × 17472, we only show image patches cropped from the tiny green boxes in (a). The input images from top to bottom are captured by Sony,
Canon, iPhone 6s Plus, and Nikon cameras, respectively.


All mentioned results of our algorithm can be found at [Google Drive]
[[Blind](https://drive.google.com/drive/folders/1BBCe157U3pBJWagJmTSYcvWzA5Iz2B1P?usp=sharing)], 
[[Non-blind](https://drive.google.com/drive/folders/1Atrstf9pLvezvs3iHWZXGDsTi74YTOYQ?usp=sharing)], 
[[Real](https://drive.google.com/drive/folders/17ldDW-CIiOXVh3lCwg4bIkVDmLNnp2r2?usp=sharing)]

    
<h3 id="6">Reference</h3>

<h id="r1">[1] C. Chen, Q. Chen, J. Xu, and V. Koltun. Learning to see in the dark. In CVPR, 2018.</h3>

<h id="r2">[2] C. Dong, C. C. Loy, K. He, and X. Tang. Learning a deep convolutional network for image super-resolution. In ECCV, 2014. </h3>

<h id="r3">[3] J. Kim, J. K. Lee, and K. M. Lee. Accurate image super-resolution using very deep convolutional networks. In CVPR, 2016.</h3>

<h id="r4">[4] E. Schwartz, R. Giryes, and A. M. Bronstein. Deepisp: Towards learning an end-to-end image processing pipeline. TIP, 2018.</h3>

<h id="r4">[5] T. Tong, G. Li, X. Liu, and Q. Gao. Image super-resolution using dense skip connections. In ICCV, 2017.</h3>

<h id="r4">[6] Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu. Residual dense network for image super-resolution. In CVPR, 2018.</h3>



&nbsp;
&nbsp;

Please consider citing this paper if you find the code and data useful in your research:
```
@inproceedings{xu2019towards,
  title={Towards real scene super-resolution with raw images},
  author={Xu, Xiangyu and Ma, Yongrui and Sun, Wenxiu},
  booktitle={CVPR},
  year={2019}
}

@article{xu2021exploiting,
  title={Exploiting Raw Images for Real-Scene Super-Resolution},
  author={Xu, Xiangyu and Ma, Yongrui and Sun, Wenxiu and Yang, Ming-Hsuan},
  journal={TPAMI},
  year={2021}
}
```
