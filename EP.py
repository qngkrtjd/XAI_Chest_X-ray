from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import get_device

import models_densenet,models_vit
import torch
import cv2
import numpy as np
from torch.autograd import Variable
from torchray.benchmark.label import NIH_LABELS
import os
import torchvision
from show_image import show_cam_on_image
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def preprocess_image(img, cuda=False):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img

is_vit=True

# Prepare input image
img = cv2.imread('./00014687_001.png', 1) # 1 for color, 0 for gray
img = np.float32(cv2.resize(img, (224, 224))) / 255
preprocessed_img = preprocess_image(img)


#model    
if(is_vit):
    model=models_vit.vit_small_patch16()
else:
    model=torchvision.models.densenet121(num_classes=14)
    #model=models_densenet.DenseNet121()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if(is_vit):
    _state_dict=torch.load("./finetuned_models/vit-s_CXR_0.3M_mae.pth",map_location=device)  
else:
    _state_dict=torch.load("./finetuned_models/densenet121_CXR_0.3M_mae.pth",map_location=device)  
    
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

# Prediction
output = model(preprocessed_img)
pred_index = int(np.argmax(output.detach().cpu().numpy()))
print(output)
print(pred_index)
print('Prediction: {}'.format(NIH_LABELS[pred_index]))

#out_dir='./result/'
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)

# Extremal perturbation backprop.
masks_1, _ = extremal_perturbation(
    model, preprocessed_img,pred_index,
    reward_func=contrastive_reward,
    debug=True,
    areas=[0.12],
)

#masks_2, _ = extremal_perturbation(
#    model, x, category_id_2,
#    reward_func=contrastive_reward,
#    debug=True,
#    areas=[0.05],
#)

# Plots.
#plot_example(preprocessed_img, masks_1, 'extremal perturbation', pred_index)
#plot_example(x, masks_2, 'extremal perturbation', category_id_2)



masks_1=masks_1.numpy()
print(type(masks_1))
print(masks_1.shape)

masks_1=masks_1[0,:]
print(masks_1.shape)

masks_1=masks_1[0,:]
print(masks_1.shape)


plt.clf()
visualization = show_cam_on_image(img, masks_1, use_rgb=True)
plt.imshow(visualization)
plt.show()