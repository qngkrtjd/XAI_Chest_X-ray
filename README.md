# **XAI Chest X-ray**

## 1.1 About The Project
의료 분야에서 딥러닝 기술을 도입해 환자의 질병을 진단하는 연구가 다수 진행되었다. 의료 분야의 특성상 단순히 질병 진단으로 끝나는 것이 아니라, 어떠한 근거로 질병을 진단하였는지가 중요하다.
이 부분에 설명가능한 인공지능 XAI 기술을 접목시키는 연구가 진행되고 있다. 하지만 현재까지의 연구는 주로 CNN 기반의 딥러닝 모델에 XAI 분야를 적용시키고 있다. 최근 자연어 처리 분야의 Transformer 를 image classification 분야에 맞게 변형한 ViT 모델이 등장했는데, 본 논문에서는 ViT 모델을 기반으로 한 chest x-ray 질병 진단에도 CNN 모델에 사용해왔던 attribution 
method 를 적용할 수 있는지, 적용 가능하다면 어떠한 방식이 ViT 모델에서 좋은 성능을 나타내는 지 연구한다.

## 1.2 Goals
+ Gradient 방식의 SmoothGrad, Perturbation 방식의 Extremal Perturbation, Activation 방식의 Grad-CAM 3가지를 ViT 모델에 적용하여 chest x-ray disease classification 의 판단 근거를 시각화한다.

+ 우선 Masked Autoencoders(MAE)를 사용해 pre-training 한 ViT 모델의 fine-tuning 과정을 이해하고 Classification 분야에 ViT 모델을 적용한다.

+ 다음으로 CNN 에 사용되었던 3가지 attribution 기법을 ViT 모델에도 적용 가능한 지 연구한다.

+ 3가지 방법들 중 ViT 모델에서 최적의 성능을 보여주는 방식에 대해 연구한다.

## 2. Related Works
### 2.1 ViT for Thorax Disease Classification
<b>Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification</b> <br/>
[Junfei Xiao](https://lambert-x.github.io/), [Yutong Bai](https://scholar.google.com/citations?user=N1-l4GsAAAAJ&hl=en), [Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ&hl=en&oi=ao), [Zongwei Zhou](https://www.zongweiz.com/) <br/>
Johns Hopkins University <br/>
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023 <br/>
[paper](https://arxiv.org/abs/2210.12843) | [code](https://github.com/lambert-x/medical_mae)

### 2.2 SmoothGrad
<b>Smoothgrad: removing noise by adding noise.</b> <br/>
Smilkow, [Daniel], [Nikhil Thorat], [Been Kim], [Fernanda Viégas], [Martin Wattenberg]<br/>
ICML Workshop 2017<br/>
[paper](https://arxiv.org/abs/1706.03825)

### 2.3 Extremal Perturabtion
<b>Understanding deep networks via extremal perturbations and smooth masks.</b> <br/>
Fong, Ruth, Mandela Patrick, and Andrea Vedaldi<br/>
In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2950-2958. 2019<br/>
[paper] (https://arxiv.org/abs/1910.08485)

### 2.4 Grad-CAM
<b>Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.</b><br/>
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, DviParikh, Dhruv Batra<br/>
ICCV 2017<br/>
[paper] (https://arxiv.org/abs/1610.02391)

## 3. Dataset
<a href="[url](https://www.kaggle.com/datasets/nih-chest-xrays/data)" > <h3>NiH Dataset </h3> </a>
https://www.kaggle.com/datasets/nih-chest-xrays/data

## 4. Result

  
