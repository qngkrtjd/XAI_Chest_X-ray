import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv
import cv2
import torchvision

import models_vit,models_densenet

is_vit=False

# Load pre-trained model

#model=models_densenet.DenseNet121()
if(is_vit):
    model=models_vit.vit_small_patch16()
else:
#    model=models_densenet.DenseNet121()
    model=torchvision.models.densenet121(num_classes=14)
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


# Load class labels
labels=['Atelectasis','Cardiomegaly','Effusion','Infiltrate','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

# Define the root folder where the images are located on the E drive
root_folder = 'E:/nih/nih'

# Define the folder names where the images are located
folder_names = ['images_001/images', 'images_002/images', 'images_003/images', 'images_004/images', 'images_005/images', 'images_006/images',
                'images_007/images', 'images_008/images', 'images_009/images', 'images_010/images', 'images_011/images', 'images_012/images']

csv_file='./BBox_List_2017.csv'

# Read the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row if present

    # Keep track of the number of images processed
    cnt_correct = 0
    cnt_wrong = 0
    total = 0

    # Create a text file to store the results
    if(is_vit):
        result_file = open("result_vit.txt","w")
    else:
        result_file = open("result_densenet.txt", "w")

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
            image = cv2.imread(image_path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_transformed = transform(image_pil)
            image_tensor = image_transformed.unsqueeze(0)  # Add batch dimension

            # Perform inference and print the results
            with torch.no_grad():
                output = model(image_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = labels[predicted_idx.item()]

            if label == predicted_label:
                cnt_correct += 1
                result_file.write('Image: {}\n'.format(image_path))
                result_file.write('True Label: {}\n'.format(label))
                result_file.write('Predicted Label: {}\n'.format(predicted_label))
                result_file.write('---\n')
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
