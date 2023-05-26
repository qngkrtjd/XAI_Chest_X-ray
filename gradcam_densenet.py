from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import models_densenet,models_vit as models_vit
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

def preprocess_image(img, cuda=False):
    means=[0.5056, 0.5056, 0.5056]
    stds=[0.252, 0.252, 0.252]

    preprocessed_img = img.copy()[: , :, ::-1]#change BGR to RGB
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))#change H,W,C -> C,H,W
    preprocessed_img = torch.from_numpy(preprocessed_img)#img to Tensor
    preprocessed_img.unsqueeze_(0)#add extra dimension for batch
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img


# Load pre-trained model
is_vit=False
if(is_vit):
    model=models_vit.vit_small_patch16()
else:
    #model=models_densenet.DenseNet121()
    model=models.densenet121(num_classes=14)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
if(is_vit):
    _state_dict=torch.load("./finetuned_models/vit-s_CXR_0.3M_mae.pth",map_location=device)  
else:
    _state_dict=torch.load("C:/Users/boohacksung/jupyter/capstone/exercise/MAE_gradcam/pytorch-grad-cam-master/finetuned_models/densenet121_CXR_0.3M_mae.pth",map_location=device)  
a=_state_dict.get('model')
del _state_dict['optimizer']
del _state_dict['epoch']
del _state_dict['scaler']
del _state_dict['args']
b = { k.replace('fea','model.fea'): v for k, v in a.items() }    
c = { k.replace('classifier','model.classifier.0'): v for k, v in b.items() }
d = { k.replace('fc_norm','norm'): v for k, v in a.items() }
if(is_vit):
    model.load_state_dict(d)
else:
    model.load_state_dict(a)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5056, 0.5056, 0.5056],
        std=[0.252, 0.252, 0.252]
    )
])
    
image_path='./test_img/00014687_001.png'

# Prepare input image
img = cv2.imread(image_path, 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
preprocessed_img = preprocess_image(img)


if is_vit:
    target_layers=[model.blocks[-1].norm1]
else:
    target_layers = [model.features[-1]]
input_tensor = preprocessed_img


# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(0)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
print(grayscale_cam)
print(type(grayscale_cam))
# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()