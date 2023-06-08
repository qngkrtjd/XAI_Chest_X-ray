"""

There are total 10,000 image fiels in E drive.
Nih provides Bounding box for 1,000 images.
This code search images which has bounding box data.
After searching, this code makes copy of images that Densenet and Vit model both predict as true label.


"""

import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv
import cv2
import torchvision

import models_vit

from lib.image_utils import preprocess_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lib.label import NIH_LABELS
import shutil


# Load pre-trained model

model_vit=models_vit.vit_small_patch16()
model_densenet=torchvision.models.densenet121(num_classes=14)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_vit.to(device)
model_densenet.to(device)

_state_dict_vit=torch.load("./finetuned_models/vit-s_CXR_0.3M_mae.pth",map_location=device)  

_state_dict_densenet=torch.load("./finetuned_models/densenet121_CXR_0.3M_mae.pth",map_location=device)  
    
a_vit=_state_dict_vit.get('model')
a_densenet=_state_dict_densenet.get('model')

del _state_dict_vit['optimizer']
del _state_dict_vit['epoch']
del _state_dict_vit['scaler']
del _state_dict_vit['args']
del _state_dict_densenet['optimizer']
del _state_dict_densenet['epoch']
del _state_dict_densenet['scaler']
del _state_dict_densenet['args']

b_vit = { k.replace('fea','model.fea'): v for k, v in a_vit.items() }    
b_densenet = { k.replace('fea','model.fea'): v for k, v in a_densenet.items() }    
c_vit = { k.replace('classifier','model.classifier.0'): v for k, v in b_vit.items() }
c_densenet = { k.replace('classifier','model.classifier.0'): v for k, v in b_densenet.items() }

d_Vit = { k.replace('fc_norm','norm'): v for k, v in a_vit.items() }
d_densenet = { k.replace('fc_norm','norm'): v for k, v in a_densenet.items() }

model_vit.load_state_dict(d_Vit)
model_densenet.load_state_dict(a_densenet)

model_vit.eval()
model_densenet.eval()


# Load class labels
labels=['Atelectasis','Cardiomegaly','Effusion','Infiltrate','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

# Define the root folder where the images are located on the E drive
root_folder = 'E:/nih/nih'

# Define the folder names where the images are located
folder_names = ['images_001/images', 'images_002/images', 'images_003/images', 'images_004/images', 'images_005/images', 'images_006/images',
                'images_007/images', 'images_008/images', 'images_009/images', 'images_010/images', 'images_011/images', 'images_012/images']

csv_file='./BBox_List_2017.csv'

 # Create the directory if it doesn't exist
output_dir = 'test_image/'
os.makedirs(output_dir, exist_ok=True)
            
# Read the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row

    # Keep track of the number of images processed
    cnt_correct = 0
    cnt_wrong = 0
    total = 0

    result_file=open("test_imgaes.txt","w")

    for row in reader:
        image_name = row[0]  # Assuming the image name is in the first column
        label = row[1]  # Assuming the label is in the second column

        # Search for the image in the folders within the root folder
        found = False
        for folder_name in folder_names:
            folder_path = os.path.join(root_folder, folder_name)
            image_path = os.path.join(folder_path, image_name)
            image_path=image_path.replace('\\','/',10)
            if os.path.isfile(image_path):
                # Image found
                found = True
                break

        if found:
            # Load and preprocess the image using OpenCV
            # Prepare input image
            img = cv2.imread(image_path, 1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            preprocessed_img = preprocess_image(img)

            # Prediction
            output_vit = model_vit(preprocessed_img)
            output_densenet = model_densenet(preprocessed_img)

            pred_index_vit = np.argmax(output_vit.detach().cpu().numpy())
            pred_index_densenet = np.argmax(output_densenet.detach().cpu().numpy())
    
            if (label == NIH_LABELS[pred_index_vit]) and (label == NIH_LABELS[pred_index_densenet]):
                cnt_correct += 1
                result_file.write('Image: {}\n'.format(image_path))
                result_file.write('True Label: {}\n'.format(label))
                result_file.write('Vit Predicted Label: {}\n'.format(NIH_LABELS[pred_index_vit]))
                result_file.write('Densenet Predicted Label: {}\n'.format(NIH_LABELS[pred_index_densenet]))
                result_file.write('---\n')

                # Specify the source and destination paths
                source_path = image_path
                destination_path = 'test_image/'+image_name

                # Copy the file
                shutil.copyfile(source_path, destination_path)

                # Check if the file was successfully copied
                if os.path.exists(destination_path):
                    print("File copied successfully.")
                else:
                    print("File copy failed.")                
            
            else:
                cnt_wrong += 1

            total += 1
 
        else:
            print(f"Image '{image_name}' not found in the specified folders.")
    
    # Close the result file
    result_file.close()
    
    # Print the overall statistics
    print('Correct:', cnt_correct)
    print('Wrong:', cnt_wrong)
    print('Total:', total)
