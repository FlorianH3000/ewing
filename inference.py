"""
inference
"""
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="2, 4"
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import torch
import utilz as ut
from arguments import args, num_classes
from dataset import ImgDataset
import time
from cv_classification import labels_test_list_final, img_test_list_final # final_dir
from pytorch_grad_cam import GradCAM
start_time = time.time()
np.seterr(divide='ignore', invalid='ignore')



###########################
model_name = 'resnet50'
final_dir = r'/home/florianh/Desktop/data_f/ewing2/results/{}'.format(args['id'])
model_path = r'/home/florianh/Desktop/data_f/ewing2/results/{}/models'.format(args['id'])
################### dataloader  ###################



test_data = ImgDataset(img_test_list_final, labels_test_list_final, 'test')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ut.initialize_model(model_name, num_classes, False, use_pretrained=True)
# model = torch.nn.DataParallel(model)
final_acc_test = list()
final_acc2_test = list()
final_sensitivity_test = list()
final_specificity_test = list()
gradcam_accum = list()
img_accum = list()

cv_counter = 0
for x in os.listdir(model_path):
    cv_counter += 1
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
    target_layer = model.layer4[-1] ### resnet
    # target_layer = model.features[-1] ### VGG and densenet161
    gradcam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

    for idx, d in enumerate(test_loader):
        images, label = d['img'].to(device), d['lab'].to(device)
        outputs = model(images.to(device))
        _, p = torch.max(outputs, 1)

        total = len(labels_test_list_final)
        if label == p:
            correct +=1
        
        [t.cpu().numpy() for t in label]
        lab_list.append(label)
        [t.cpu().numpy() for t in p]
        pred_list.append(p)
        
        # i = 0
        if idx <= total:
            lab = img_test_list_final[idx].split("train/", 1)[1][2:6]
            # In this example grayscale_cam has only one image in the batch:
            gradcam_img = gradcam(input_tensor=images, target_category=label.item(), aug_smooth=True)
            
            l = str(label.item())
            p = str(p.item())
            
            ### gradcam image
            # plt.figure()
            # plt.axis('off')
            # plt.title(lab +'_' + str(classes[p[0]][2:]))
            # plt.imshow(gradcam_img[0,:,:])
            # plt.savefig(os.path.join(final_dir, 'gradcam_img_' + str(idx)) + cv_counter)
            # plt.close()
            
            # normal image
            plt.figure()
            plt.axis('off')
            # plt.title(lab +'_' + str(classes[p[-1]][2:]))
            plt.title(l + '_' + p)
            img = images.cpu().numpy()
            plt.imshow(img[0,0,:,:], cmap='bone')
            # plt.savefig(os.path.join(final_dir, 'og_image_' + str(idx) + '_' + str(cv_counter)), dpi=300)
            plt.close()
            
            ### matched image
            plt.figure() 
            plt.axis('off')
            # plt.title(lab +'_' + str(classes[p[-1]][2:]))
            plt.title(l + '_' + p)
            plt.imshow(img[0,0,:,:], cmap='bone')
            plt.imshow(gradcam_img[0,:,:], alpha=0.5, cmap='rainbow')
            # plt.savefig(os.path.join(final_dir, 'gradcam_image_' + str(idx) + '_' + str(cv_counter)), dpi=300)
            plt.close()

            if cv_counter == 1:
                plt.figure()
                plt.axis('off')
                # plt.title(lab +'_' + str(classes[p[-1]][2:]))
                plt.title(l + '_' + p)
                img = images.cpu().numpy()
                plt.imshow(img[0,0,:,:], cmap='bone')
                # plt.savefig(os.path.join(final_dir, 'og_image_' + str(idx) + '_' + str(cv_counter)), dpi=300)
                plt.close()
                
                gradcam_accum.append(gradcam_img[0,:,:])
                img_accum.append(img[0,0,:,:])
            gradcam_accum[idx] += gradcam_img[0,:,:]
num = 0
for img_normal, img_gradcam_accum in zip(img_accum, gradcam_accum):
    plt.figure() 
    plt.axis('off')

    plt.title(l + '_' + p)
    plt.imshow(img_normal, cmap='bone')
    plt.imshow(img_gradcam_accum, alpha=0.5, cmap='rainbow')
    # plt.savefig(os.path.join(final_dir, 'gradcam_image_accum' + str(num)), dpi=500)
    plt.close()
    num += 1
        


# =============================================================================
# test metics     
# =============================================================================

    acc2 = correct/total
    sens, spec, acc = ut.calculate_sensitivity_specificity(lab_list, pred_list, num_classes)
    final_acc_test.append(acc)
    final_acc2_test.append(acc2)
    final_specificity_test.append(spec)
    final_sensitivity_test.append(sens)
        
    print('total: ', total)
    print('correct: ', correct)
print('--------------')
print('Final Accuracy Test: ', round(np.mean(final_acc_test), 3))
print('Final Accuracy2 Test: ', round(np.mean(final_acc2_test), 3))
print('Final Sensitivity Test: ', round(np.mean(final_sensitivity_test), 3))
print('Final Specificity Test: ', round(np.mean(final_specificity_test), 3))

print('acc', final_acc2_test)
print('final_sensitivity_test', final_sensitivity_test)
print('final_specificity_test', final_specificity_test)




end = time.time()
elapsed_time = end - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('time for prediction: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))