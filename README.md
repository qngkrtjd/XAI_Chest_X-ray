# **XAI Chest X-ray**

## 1.1 About The Project
ViT 모델을 기반으로 한 chest x-ray 질병 진단에도 CNN 모델에 사용해왔던 attribution method 를 적용할 수 있는지, 적용 가능하다면 어떠한 방식이 ViT 모델에서 좋은 성능을 나타내는 지 연구한다.

## 1.2 Goals
+ Gradient 방식의 SmoothGrad, Perturbation 방식의 Extremal Perturbation, Activation 방식의 Grad-CAM 3가지를 Densenet과 ViT 모델에 적용하여 chest x-ray disease classification 의 판단 근거를 시각화한다.

+ Masked Autoencoders를 사용해 pre-training 한 ViT 모델의 fine-tuning 과정을 이해하고 Classification 분야에 ViT 모델을 적용한다.

+ CNN 에 사용되었던 3가지 attribution 기법을 ViT 모델에도 적용 가능한 지 연구한다.

+ ViT 모델에서 최적의 성능을 보여주는 방식에 대해 연구한다.

## 2. Related Works
### 2.1 ViT for Thorax Disease Classification
_<b>Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification</b>_<br/>
Junfei Xiao, Yutong Bai, Alan Yuille, Zongwei Zhou<br/>
Johns Hopkins University <br/>
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023 <br/>
[paper](https://arxiv.org/abs/2210.12843) | [code](https://github.com/lambert-x/medical_mae)

### 2.2 SmoothGrad
_<b>Smoothgrad: removing noise by adding noise._ </b> <br/>
Smilkow, Daniel, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg<br/>
ICML Workshop 2017<br/>
[paper](https://arxiv.org/abs/1706.03825)

### 2.3 Extremal Perturabtion
_<b>Understanding deep networks via extremal perturbations and smooth masks.</b>_ <br/>
Fong, Ruth, Mandela Patrick, and Andrea Vedaldi<br/>
In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2950-2958. 2019<br/>
[paper](https://arxiv.org/abs/1910.08485)

### 2.4 Grad-CAM
_<b>Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.</b>_ <br/>
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, DviParikh, Dhruv Batra<br/>
ICCV 2017<br/>
[paper](https://arxiv.org/abs/1610.02391)

## 3. Dataset
<a href="[url](https://www.kaggle.com/datasets/nih-chest-xrays/data)" > <h3>NiH Dataset </h3> </a>
https://www.kaggle.com/datasets/nih-chest-xrays/data

## 4. Result
<img src="https://github.com/qngkrtjd/XAI_Chest_X-ray/assets/98075749/63ea11bb-8e50-4821-b19c-2213b9ccdb44">
  
  
