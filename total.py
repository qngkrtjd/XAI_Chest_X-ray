""" 
This is final code for colab using GPU.

Read 264 image files in "./test_image/" .
Image Labels are stored in "test_image_info.csv".
Model weights are stored in "./finetuned_models/".

1. Variable setting
2. load fine-tuned model
3. make prediction
4. perform SG , evaluation
            EP, evaluation
            Grad-CAM, evaluation
            for ViT
5. Do it again for Densenet

Result images and evaluation text files will be stored at each "image name" folder under "./output_dir/".

 """

import os
import torch
from torchvision import models, transforms
from PIL import Image
import csv
import cv2
import torchvision

import models.models_vit, models.models_densenet

from lib.gradients import SmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image, show_cam_on_image_SG, show_cam_on_image_EP
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lib.label import NIH_LABELS
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import matplotlib.patches as patches
from evaluation.insertion_deletion import InsertionDeletion

#reshape ViT for grad-cam
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


is_vit=True
    
# Load class labels
labels=['Atelectasis','Cardiomegaly','Effusion','Infiltrate','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural_Thickening','Hernia']

# Define the root folder where the images are located
image_folder = 'test_image/'

csv_file='./csv_files/test_image_info3.csv'

# Create the directory if it doesn't exist
output_dir = 'final_result'
os.makedirs(output_dir, exist_ok=True)
            
# List for evaluation
ins_densenet_SG=[]
del_densenet_SG=[]
ins_densenet_EP=[]
del_densenet_EP=[]
ins_densenet_CAM=[]
del_densenet_CAM=[]

ins_vit_SG=[]
del_vit_SG=[]
ins_vit_EP=[]
del_vit_EP=[]
ins_vit_CAM=[]
del_vit_CAM=[]


# Read the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row


    for row in reader:
        image_name = row[0]
        label = row[1]
        image_path=image_folder+image_name
        # for bounding box
        x=float(row[2])
        y=float(row[3])
        w=float(row[4])
        h=float(row[5])


        is_vit=False
        for i in [1,2]:
            
            # load models
            if(is_vit):
                mode='vit'
                model=models.models_vit.vit_small_patch16()
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
                    use_cuda=True,
                    reshape_transform=reshape_transform)

            else:
                model.load_state_dict(a)
                model.eval()
                target_layers =[model.features[-1]]
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        
            evaluator = InsertionDeletion(model,pixel_batch_size=25,sigma=10.)

            # Prepare input image
            img = cv2.imread(image_path, 1)
            print(image_path)
            img_bbox=img
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            preprocessed_img = preprocess_image(img,cuda=True)
            input_tensor = preprocessed_img

            # Prediction
            output = model(preprocessed_img)
            pred_index = int(np.argmax(output.detach().cpu().numpy()))

            # Output_dir
            output_dir = 'final_result/'+image_name
            if is_vit:
                output_eval=output_dir+'/evaluation_vit.txt'
            else:
                output_eval=output_dir+'/evaluation_densenet.txt'
                
            os.makedirs(output_dir, exist_ok=True)
#---------------------------Smooth Grad--------------------------------------------------
            # Create a text file to store the results
            result_file = open(output_eval,"w") # "result/image_name/evaluation.txt"    
        
            # Compute smooth gradient
            smooth_grad = SmoothGrad(
                pretrained_model=model,
                n_samples=10,
                magnitude=True,
                cuda=True)
            smooth_saliency = smooth_grad(preprocessed_img)
        
        
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            visualization = show_cam_on_image_SG(img, smooth_saliency, use_rgb=True)
            
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
                ins_vit_SG.append(mean_insertion_auc_SG)
                del_vit_SG.append(mean_deletion_auc_SG)
            else:               
                result_file.write('SG_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_SG, mean_deletion_auc_SG))
                ins_densenet_SG.append(mean_insertion_auc_SG)
                del_densenet_SG.append(mean_deletion_auc_SG)

            # Save the visualization as a file in the specified directory ("result/image_name/mode_SG_label_image_name.png")
            result_name=mode+'_SG_'+label+'_'+image_name
            output_path = os.path.join(output_dir,result_name)
            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)        
            plt.close()

#---------------------------Extremal Perturbation--------------------------------------------------
            # Compute EP
            masks, _ = extremal_perturbation(
                model, preprocessed_img,pred_index,
                reward_func=contrastive_reward,
                debug=False,
                areas=[0.12],
            )

            masks=masks.cpu().numpy()
            masks=masks[0,:]
            masks=masks[0,:]

            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            visualization = show_cam_on_image_EP(img,masks, use_rgb=True)
            
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
                ins_vit_EP.append(mean_insertion_auc_EP)
                del_vit_EP.append(mean_deletion_auc_EP)
            else:
                result_file.write('EP_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_EP, mean_deletion_auc_EP))
                ins_densenet_EP.append(mean_insertion_auc_EP)
                del_densenet_EP.append(mean_deletion_auc_EP)

            # Save the visualization as a file in the specified directory ("result/image_name/mode_EP_label_image_name.png")
            result_name=mode+'_EP_'+label+'_'+image_name
            output_path = os.path.join(output_dir,result_name)
        
            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

#---------------------------Grad CAM--------------------------------------------------
            # Compute Grad CAM
            targets = [ClassifierOutputTarget(pred_index)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)


            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
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
                ins_vit_CAM.append(mean_insertion_auc_grad)
                del_vit_CAM.append(mean_deletion_auc_grad)
            else:
                result_file.write('GradCAM_densenet insertion auc: {}, deletion auc: {}\n'.format(mean_insertion_auc_grad, mean_deletion_auc_grad))
                ins_densenet_CAM.append(mean_insertion_auc_grad)
                del_densenet_CAM.append(mean_deletion_auc_grad)

            # Save the visualization as a file in the specified directory ("result/image_name/mode_gradcam_label_image_name.png")
            result_name=mode+'_gradcam_'+label+'_'+image_name
            output_path = os.path.join(output_dir,result_name)
        
            plt.imshow(visualization)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

#---------------------------Bounding Box--------------------------------------------------
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
            # Draw bounding box to input image
            fig, ax = plt.subplots(1)
            image_rgb = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            rectangle = patches.Rectangle((x,y), w, h, linewidth=3, edgecolor='yellow', facecolor='none')
            ax.add_patch(rectangle)

    
            # Save the visualization as a file in the specified directory ("result/image_name/bbox_label_image_name.png")
            result_name='bbox_'+label+'_'+image_name
            output_path = os.path.join(output_dir,result_name)
        
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            
            result_file.write('------\n')
            plt.close()

            # for Vit
            is_vit=True
        
            # Close the result file 
            result_file.close()
        
        # Print mean value of evaluation
        ins_densenet_SG_score=sum(ins_densenet_SG)/len(ins_densenet_SG)
        del_densenet_SG_score=sum(del_densenet_SG)/len(del_densenet_SG)
        ins_densenet_EP_score=sum(ins_densenet_EP)/len(ins_densenet_EP)
        del_densenet_EP_score=sum(del_densenet_EP)/len(del_densenet_EP)
        ins_densenet_CAM_score=sum(ins_densenet_CAM)/len(ins_densenet_CAM)
        del_densenet_CAM_score=sum(del_densenet_CAM)/len(del_densenet_CAM)

        ins_vit_SG_score=sum(ins_vit_SG)/len(ins_vit_SG)
        del_vit_SG_score=sum(del_vit_SG)/len(del_vit_SG)
        ins_vit_EP_score=sum(ins_vit_EP)/len(ins_vit_EP)
        del_vit_EP_score=sum(del_vit_EP)/len(del_vit_EP)
        ins_vit_CAM_score=sum(ins_vit_CAM)/len(ins_vit_CAM)
        del_vit_CAM_score=sum(del_vit_CAM)/len(del_vit_CAM)

        print(f"Score for insertion densenet SmoothGrad : {ins_densenet_SG_score}")
        print(f"Score for deletion densenet SmoothGrad : {del_densenet_SG_score}")
        print(f"Score for insertion densenet EP : {ins_densenet_EP_score}")
        print(f"Score for deletion densenet EP : {del_densenet_EP_score}")
        print(f"Score for insertion densenet GradCAM : {ins_densenet_CAM_score}")
        print(f"Score for deletion densenet GradCAM : {del_densenet_CAM_score}")
        print()
        print(f"Score for insertion vit SmoothGrad : {ins_vit_SG_score}")
        print(f"Score for deletion vit SmoothGrad : {del_vit_SG_score}")
        print(f"Score for insertion vit EP : {ins_vit_EP_score}")
        print(f"Score for deletion vit EP : {del_vit_EP_score}")
        print(f"Score for insertion vit GradCAM : {ins_vit_CAM_score}")
        print(f"Score for deletion vit GradCAM : {del_vit_CAM_score}")

 
    is_vit=False

# Final evaluation results 
ins_densenet_SG_score=sum(ins_densenet_SG)/len(ins_densenet_SG)
del_densenet_SG_score=sum(del_densenet_SG)/len(del_densenet_SG)
ins_densenet_EP_score=sum(ins_densenet_EP)/len(ins_densenet_EP)
del_densenet_EP_score=sum(del_densenet_EP)/len(del_densenet_EP)
ins_densenet_CAM_score=sum(ins_densenet_CAM)/len(ins_densenet_CAM)
del_densenet_CAM_score=sum(del_densenet_CAM)/len(del_densenet_CAM)

ins_vit_SG_score=sum(ins_vit_SG)/len(ins_vit_SG)
del_vit_SG_score=sum(del_vit_SG)/len(del_vit_SG)
ins_vit_EP_score=sum(ins_vit_EP)/len(ins_vit_EP)
del_vit_EP_score=sum(del_vit_EP)/len(del_vit_EP)
ins_vit_CAM_score=sum(ins_vit_CAM)/len(ins_vit_CAM)
del_vit_CAM_score=sum(del_vit_CAM)/len(del_vit_CAM)

print(f"Score for insertion densenet SmoothGrad : {ins_densenet_SG_score}")
print(f"Score for deletion densenet SmoothGrad : {del_densenet_SG_score}")
print(f"Score for insertion densenet EP : {ins_densenet_EP_score}")
print(f"Score for deletion densenet EP : {del_densenet_EP_score}")
print(f"Score for insertion densenet GradCAM : {ins_densenet_CAM_score}")
print(f"Score for deletion densenet GradCAM : {del_densenet_CAM_score}")
print()
print(f"Score for insertion vit SmoothGrad : {ins_vit_SG_score}")
print(f"Score for deletion vit SmoothGrad : {del_vit_SG_score}")
print(f"Score for insertion vit EP : {ins_vit_EP_score}")
print(f"Score for deletion vit EP : {del_vit_EP_score}")
print(f"Score for insertion vit GradCAM : {ins_vit_CAM_score}")
print(f"Score for deletion vit GradCAM : {del_vit_CAM_score}")