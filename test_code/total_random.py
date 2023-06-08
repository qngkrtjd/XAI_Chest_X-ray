"""

Testing code for EP.
984 images
Input images in random order.
Includes evaluation code.

"""

import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv
import cv2
import torchvision

import models_vit,models_densenet

from lib.gradients import SmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image, show_cam_on_image_SG
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lib.label import NIH_LABELS
from show_image import show_cam_on_image_EP
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import matplotlib.patches as patches
from evaluation.insertion_deletion import InsertionDeletion

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


is_vit=True
    
# Load class labels
labels=['Atelectasis','Cardiomegaly','Effusion','Infiltrate','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

# Define the root folder where the images are located on the E drive
root_folder = 'E:/nih/nih'

# Define the folder names where the images are located
folder_names = ['images_001/images', 'images_002/images', 'images_003/images', 'images_004/images', 'images_005/images', 'images_006/images',
                    'images_007/images', 'images_008/images', 'images_009/images', 'images_010/images', 'images_011/images', 'images_012/images']

csv_file='./BBox_List_2017.csv'

# Create the directory if it doesn't exist
output_dir = 'result_4'
os.makedirs(output_dir, exist_ok=True)
            
   


# Read the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row if present

    # Keep track of the number of images processed
    cnt_correct = 0
    cnt_wrong = 0
    total = 0


    for row in reader:
        image_name = row[0]  # Assuming the image name is in the first column
        label = row[1]  # Assuming the label is in the second column
        x=float(row[2])
        y=float(row[3])
        w=float(row[4])
        h=float(row[5])

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
            is_vit=False
            for i in [1,2]:
                
                if(is_vit):
                    mode='vit'
                    model=models_vit.vit_small_patch16()
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)

                    _state_dict=torch.load("./finetuned_models/vit-s_CXR_0.3M_mae.pth",map_location=device)  
                    
                else:
                    mode='densenet'
                    model=torchvision.models.densenet121(num_classes=14)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)
                
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
                    model.eval()
                    target_layers=[model.blocks[-1].norm1]
                    cam = GradCAM(model=model,
                        target_layers=target_layers,
                        use_cuda=False,
                        reshape_transform=reshape_transform)

                else:
                    model.load_state_dict(a)
                    model.eval()
                    target_layers =[model.features[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
           
                evaluator = InsertionDeletion(model,pixel_batch_size=25,sigma=10.)

                # Load and preprocess the image using OpenCV
                # Prepare input image
                img = cv2.imread(image_path, 1)
                img_bbox=img
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                preprocessed_img = preprocess_image(img)
                input_tensor = preprocessed_img

                # Prediction
                output = model(preprocessed_img)
                pred_index = int(np.argmax(output.detach().cpu().numpy()))
    
                if label == NIH_LABELS[pred_index]:
                    cnt_correct += 1
                    
                    output_dir = 'result_4/'+image_name
                    if is_vit:
                        output_eval=output_dir+'/evaluation_vit.txt'
                    else:
                        output_eval=output_dir+'/evaluation_densenet.txt'
                        
                    os.makedirs(output_dir, exist_ok=True)
#-----------------------------------------------------------------------------
                       # Create a text file to store the results
                
                    result_file = open(output_eval,"w")    
               
                   # Compute smooth gradient
                    smooth_grad = SmoothGrad(
                        pretrained_model=model,
                        n_samples=10,
                        magnitude=True)
                    smooth_saliency = smooth_grad(preprocessed_img)
                
                
                    os.environ['KMP_DUPLICATE_LIB_OK']='True'
                    visualization = show_cam_on_image_SG(img, smooth_saliency, use_rgb=True)
                    result_name=mode+'_SG_'+label+'_'+image_name
            
                    #evaluation
                    insertion_auc_SG = np.array([])
                    deletion_auc_SG = np.array([])
    
                    heatmap=torch.from_numpy(smooth_saliency)
                    res_single_SG = evaluator.evaluate(heatmap,preprocessed_img.squeeze().to(device), pred_index)
                    ins_auc_SG = res_single_SG['ins_auc']
                    insertion_auc_SG = np.append(insertion_auc_SG, np.array(ins_auc_SG))
                    del_auc_SG = res_single_SG['del_auc']
                    deletion_auc_SG = np.append(deletion_auc_SG, np.array(del_auc_SG))
                    mean_insertion_auc_SG = np.mean(insertion_auc_SG)
                    mean_deletion_auc_SG = np.mean(deletion_auc_SG)
    
                    result_file.write('Image: {}\n'.format(image_path))
                    result_file.write('True Label: {}\n'.format(label))
                    result_file.write('Predicted Label: {}\n'.format(NIH_LABELS[pred_index]))
                    
                    if is_vit:
                        result_file.write('SG_vit insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_SG, mean_deletion_auc_SG))
                    else:               
                        result_file.write('SG_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_SG, mean_deletion_auc_SG))

                    # Save the visualization as a file in the specified directory (e.g., "output.png")
                    output_path = os.path.join(output_dir,result_name)
                
                    plt.imshow(visualization)
                    plt.axis('off')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
              
                    # Optional: Close the figure to free up resources
                    plt.close()
#--------------------------------------------------------------------
                    masks, _ = extremal_perturbation(
                        model, preprocessed_img,pred_index,
                        reward_func=contrastive_reward,
                        debug=False,
                        areas=[0.12],
                    )

                    masks=masks.numpy()
                    masks=masks[0,:]
                    masks=masks[0,:]

                    os.environ['KMP_DUPLICATE_LIB_OK']='True'
                    visualization = show_cam_on_image_EP(img,masks, use_rgb=True)
                    result_name=mode+'_EP_'+label+'_'+image_name
            
                    #evaluation
                    insertion_auc_EP = np.array([])
                    deletion_auc_EP = np.array([])
    
                    heatmap=torch.from_numpy(masks)
                    res_single_EP = evaluator.evaluate(heatmap,preprocessed_img.squeeze().to(device), pred_index)
                    ins_auc_EP = res_single_EP['ins_auc']
                    insertion_auc_EP = np.append(insertion_auc_EP, np.array(ins_auc_EP))
                    del_auc_EP = res_single_EP['del_auc']
                    deletion_auc_EP = np.append(deletion_auc_EP, np.array(del_auc_EP))
                    mean_insertion_auc_EP = np.mean(insertion_auc_EP)
                    mean_deletion_auc_EP = np.mean(deletion_auc_EP)

                    if is_vit:
                        result_file.write('EP_vit insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_EP, mean_deletion_auc_EP))
                    else:
                        result_file.write('EP_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_EP, mean_deletion_auc_EP))


                    # Save the visualization as a file in the specified directory (e.g., "output.png")
                    output_path = os.path.join(output_dir,result_name)
                
                    plt.imshow(visualization)
                    plt.axis('off')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
              
                    # Optional: Close the figure to free up resources
                    plt.close()
#------------------------------------------------------------------

                    targets = [ClassifierOutputTarget(pred_index)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)


                    os.environ['KMP_DUPLICATE_LIB_OK']='True'
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
                    result_name=mode+'_gradcam_'+label+'_'+image_name

                    #evaluation
                    insertion_auc_grad = np.array([])
                    deletion_auc_grad = np.array([])
    
                    heatmap=torch.from_numpy(smooth_saliency)
                    res_single_grad = evaluator.evaluate(heatmap,preprocessed_img.squeeze().to(device), pred_index)
                    ins_auc_grad = res_single_grad['ins_auc']
                    insertion_auc_grad = np.append(insertion_auc_grad, np.array(ins_auc_grad))
                    del_auc_grad = res_single_grad['del_auc']
                    deletion_auc_grad = np.append(deletion_auc_grad, np.array(del_auc_grad))
                    mean_insertion_auc_grad = np.mean(insertion_auc_grad)
                    mean_deletion_auc_grad = np.mean(deletion_auc_grad)
                    
                    if is_vit:
                        result_file.write('GradCAM_vit insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_grad, mean_deletion_auc_grad))
                    else:
                        result_file.write('GradCAM_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_grad, mean_deletion_auc_grad))
                    

                   # Save the visualization as a file in the specified directory (e.g., "output.png")
                    output_path = os.path.join(output_dir,result_name)
                
                    plt.imshow(visualization)
                    plt.axis('off')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
              
                    # Optional: Close the figure to free up resources
                    plt.close()

#--------------------------------------------------------------------
                    os.environ['KMP_DUPLICATE_LIB_OK']='True'
            
                    fig, ax = plt.subplots(1)
                    image_rgb = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                    rectangle = patches.Rectangle((x,y), w, h, linewidth=3, edgecolor='yellow', facecolor='none')
                    ax.add_patch(rectangle)

                    result_name='bbox_'+label+'_'+image_name
            
                    # Save the visualization as a file in the specified directory (e.g., "output.png")
                    output_path = os.path.join(output_dir,result_name)
                
                    plt.axis('off')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                    
                    result_file.write('------\n')
                    # Optional: Close the figure to free up resources
                    plt.close()

                else:
                    cnt_wrong += 1

                total += 1
                is_vit=True
        else:
            print(f"Image '{image_name}' not found in the specified folders.")
    
            # Close the result file
            result_file.close()
    
        # Print the overall statistics
        print('Correct:', cnt_correct)
        print('Wrong:', cnt_wrong)
        print('Total:', total)
    is_vit=False