"""
inference
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2, 4"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import torch
import utilz as ut
from arguments import args, num_classes, classes, input_size
from torchvision import transforms
from dataset import ImgDataset
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, precision_score, recall_score
import time
from cv_classification import labels_test_list_final, img_test_list_final # final_dir
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
start_time = time.time()
np.seterr(divide='ignore', invalid='ignore')

# norm_mean = [0.485, 0.456, 0.406]
# norm_std = [0.229, 0.224, 0.225]

###########################
model_name = 'resnet152'
# new_dir = os.path.join('results', args['data'], str(args['lr']), str(args['epx']))
final_dir = r'/home/florianh/Desktop/data_f/ewing2/results/14'
################### dataloader  ###################

transf = transforms.Compose([
                                    # transforms.Resize(input_size),
                                    # transforms.CenterCrop((input_size,input_size)),
                                    # transforms.ToTensor(),
                                    # transforms.Normalize(norm_mean, norm_std)
                                    ])


test_data = ImgDataset(img_test_list_final, labels_test_list_final, 'test', transf)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ut.initialize_model(model_name, num_classes, False, use_pretrained=True)
model = torch.nn.DataParallel(model)
final_acc_test = list()
final_sensitivity_test = list()
final_specificity_test = list()

# from torchvision.models import resnet50
# model = resnet50(pretrained=True)

model_path = r'/home/florianh/Desktop/data_f/ewing2/models/14'
# for x in os.listdir(os.path.join(os.getcwd(), 'models')):
for x in os.listdir(model_path):
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', x)))
    model.load_state_dict(torch.load(os.path.join(model_path, x)))
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    lab_list = list()
    pred_list = list()
    
    
# =============================================================================
# GradCam
# =============================================================================
    # target_layer = model.layer4[-1] ### resnet
    # # target_layer = model.features[-1] ### VGG and densenet161
    # gradcam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
    # # target_category = 281
    
    # for idx, d in enumerate(test_loader):
    #     images, label = d['img'].to(device), d['lab'].to(device)

    #     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #     gradcam_img = gradcam(input_tensor=images)
        
    #     plt.figure(figsize=(15,15))
    #     plt.axis('off')
    #     # plt.title(lab +'_' + str(classes[prediction[0]][2:]))
    #     plt.imshow(gradcam_img[0,:,:])
    #     plt.savefig(os.path.join(final_dir, 'gradcam_img_' + str(idx)))
        
        
    #     plt.figure(figsize=(15,15))
    #     plt.axis('off')
    #     # plt.title(lab +'_' + str(classes[prediction[0]][2:]))
    #     img = images.cpu().numpy()
    #     plt.imshow(img[0,0,:,:], cmap='bone')
    #     plt.savefig(os.path.join(final_dir, 'og_image' + str(idx)))
        
    #     # # In this example grayscale_cam has only one image in the batch:
    #     grayscale_cam = gradcam(input_tensor=images, target_category=None)
    #     plt.figure() 
    #     plt.imshow(img[0,0,:,:])
    #     plt.imshow(grayscale_cam[0,:,:], alpha=0.5, cmap='plasma') ## inferno, seismic
    #     plt.axis('off')




# =============================================================================
# test metics     
# =============================================================================
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            images, l = d['img'], d['lab'].to(device)
            outputs = model(images.to(device))
            _, p = torch.max(outputs, 1)
    
            total += 1
            correct += (l == p).sum().item()
            
            [t.cpu().numpy() for t in l]
            lab_list.append(l)
            [t.cpu().numpy() for t in p]
            pred_list.append(p)
            
            i = 0
            for i in range(images.size(0)):
                if l[i].item() == 0:
                    lab ='osteom'
                if l[i].item() == 1:
                    lab ='ewing'
                if l[i].item() == 2:
                    lab='norm'
        
                img = images[i,:,:,:].numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.figure(figsize=(15,15))
                plt.axis('off')
                plt.title(lab +'_' + str(classes[p[0]][2:]))
                plt.imshow(img, cmap='bone')
                plt.savefig(os.path.join(final_dir, 'prediction' + str(idx)))
                plt.close()
            
            
        sens, spec, acc = ut.calculate_sensitivity_specificity(lab_list, pred_list, num_classes)
        final_acc_test.append(acc)
        final_specificity_test.append(spec)
        final_sensitivity_test.append(sens)
                    

print('--------------')
print('Final Accuracy Test: ', round(np.mean(final_acc_test), 3))
print('Final Sensitivity Test: ', round(np.mean(final_sensitivity_test), 3))
print('Final Specificity Test: ', round(np.mean(final_specificity_test), 3))




end = time.time()
elapsed_time = end - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('time for prediction: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))