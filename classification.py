   
    
"""
Classification cv stratified

"""
import torch
from torchvision import transforms
import utilz as ut
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from dataset import ImgDataset
import torch.nn as nn
from arguments import args, num_classes, input_size, classes
import time
import os
import pandas as pd
start_time = time.time()
from sklearn.model_selection import train_test_split, StratifiedKFold




################### directory for saving results ###################
new_dir = os.path.join('results', args['data'], str(args['lr']), str(args['epx']))
if not os.path.exists(new_dir):
    os.makedirs(new_dir)   
    print('Directory ', new_dir, 'created!')
else:
    print('Directory ', new_dir, 'already exists!')

###ImageNet values for ResNet, VGG, ...
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
if args['augmentation'] == True:
################### dataloader  ###################
    transf_train = transforms.Compose([
        # transforms.Resize(input_size),
        #                                 transforms.CenterCrop((input_size, input_size)),
        #                                 transforms.ToTensor(),
        #                                 transforms.ToPILImage(),
        #                                 #transforms.RandomRotation(2),
        #                                 transforms.RandomHorizontalFlip(p=0.5),
        #                                 #transforms.RandomVerticalFlip(p=0.5),
        #                                 #transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=2, fill=0),
        #                                 #transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
        #                                # transforms.RandomCrop(input_size*0.95),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(norm_mean, norm_std),                               
                                        ])
else:
    transf_train = transforms.Compose([
        # transforms.Resize(input_size),
        #                         transforms.CenterCrop((input_size, input_size)),                               
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(norm_mean, norm_std),                               
                                ])

transf_val = transforms.Compose([transforms.Resize(input_size),
                                    transforms.CenterCrop((input_size, input_size)),                                    
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std),                                
                                    ])




file_paths = list()
labels_list0 = list()
img_val = list()
# nativ_list = list()

file_paths = list()
for x in os.listdir(args['data_dir_train']):
    y = os.path.join(args['data_dir_train'], x)
    for a in os.listdir(y):
        z = os.path.join(args['data_dir_train'], x , a)
        file_paths.append(z)
        if '1_ewing' in z:
            labels_list0.append(0)
        if '0_osteom' in z:
            labels_list0.append(1)


## split list into train and test
## split train into train and val cross validated
pat_train_val, pat_test, pat_label_train_val, _ = train_test_split(file_paths, labels_list0, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'], stratify=labels_list0)


# =============================================================================
# cross validation 
# =============================================================================
skf = StratifiedKFold(n_splits=args['cv_splits'])
skf.get_n_splits(pat_train_val, pat_label_train_val)
final_val_acc = list()
cv_counter = 1
pat_train_val, pat_train_val= np.array(pat_train_val), np.array(pat_train_val)

for train_index, val_index in skf.split(pat_train_val, pat_label_train_val):
    img_train, img_val = pat_train_val[train_index], pat_train_val[val_index]

    labels_list1 = list()
    labels_v_list1 = list()
    labels_test_list1 = list()
    img_list1 = list()
    img_v_list1 = list()
    img_test_list1 = list()
    for img_path in img_train:
            img_list1.append(img_path)
            if '1_ewing' in img_path:
                labels_list1.append(0)
            if '0_osteom' in img_path:
                labels_list1.append(1)
    for img_path in img_val:
            img_v_list1.append(img_path)
            if '1_ewing' in img_path:
                labels_v_list1.append(0)
            if '0_osteom' in img_path:
                labels_v_list1.append(1)
                
    for img_path in pat_test:
            img_test_list1.append(img_path)
            if '1_ewing' in img_path:
                labels_test_list1.append(0)
            if '0_osteom' in img_path:
                labels_test_list1.append(1)
    
    if __name__ == "__main__":
        
        if cv_counter == 1:
            print('number of files for training: ', len(img_list1))
            print('number of files for validation: ', len(img_v_list1))
            print('number of files for testing: ', len(img_test_list1))
        
        
        ### custom dataloader
        torch.set_num_threads(2)

        
        train_data = ImgDataset(img_list1, labels_list1, transf_train)
        val_data = ImgDataset(img_v_list1, labels_v_list1, transf_val)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=5)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args['batch_size'], shuffle=True, num_workers=5)

        
        
        # =============================================================================
        # architecture
        # =============================================================================
        
        # Initialize the model for this run
        model = ut.initialize_model(args['model_name'], num_classes, args['feature_extract'], use_pretrained=True)
        
        params_to_update = model.parameters()
        #print("Params to learn:")
        if args['feature_extract']:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    # print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
        
        
        # =============================================================================
        # median frequency balancing
        # =============================================================================
        # get the class labels of each image
        class_labels = labels_list1
        # empty array for counting instance of each class
        count_labels = np.zeros(len(classes))
        # empty array for weights of each class
        class_weights = np.zeros(len(classes))
        
        # populate the count array
        for l in class_labels:
            count_labels[l] += 1

        # get median count
        median_freq = np.median(count_labels)
        
        classes.sort()
        # calculate the weigths
        for i in range(len(classes)):
          class_weights[i] = median_freq/count_labels[i]
        
        # print the weights
        for i in range(len(classes)):
            print('weights: ', classes[i],":", class_weights[i])
        
        # =============================================================================
        #  optimizer, loss function
        # =============================================================================
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(class_weights)
        #criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params_to_update, lr=args['lr'])
        
        model.to(device)
        
        
        ################### training ###################
        valid_loss_min = 10000
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        train_dice_score = [] 
        val_dice_score, val_sens, val_spec, acc2_val = [],[],[],[]
        total_step_train = len(train_loader)
        total_step_val = len(val_loader)
        
        for epoch in range(1, args['epx']+1):
            running_loss = 0.0
            correct_t = 0
            total_t = 0
            dice_train, dice_val, sensitivity_val, specificity_val = 0,0,0,0
            print('----------------------------------')
            print(f'Epoch {epoch}')
            
            
            for batch_idx, d in enumerate(train_loader):
                data_t, target_t = d['img'].to(device), d['lab'].to(device)
                
                ### zero the parameter gradients
                optimizer.zero_grad()
                ### forward + backward + optimize
                outputs_t = model(data_t)
                
                loss_t = criterion(outputs_t, target_t.long())
                
                loss_t.backward()
                optimizer.step()
                ### print statistics
                running_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                # print('pred_t: ', pred_t)
                # print('target_t: ',target_t )
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
                    
                ### train dice score, acc
                dice_train += ut.dice_score(target_t.cpu().numpy(),pred_t.cpu().numpy())        
            
                # visualize training images
                # if batch_idx < 5:    
                #     img = data_t.cpu().numpy()
                #     plt.figure(dpi=600)
                #     plt.axis('off') 
                #     plt.imshow(img[0,1,:,:], cmap='gray')
            
            
            train_dice_score.append(dice_train / total_step_train)  
            train_acc.append(100 * correct_t / total_t)
            train_loss.append(running_loss / total_step_train)
            print(f'\ntrain loss: {(train_loss[-1]):.4f}, train acc: {(train_acc[-1]):.4f}, train dice: {(train_dice_score[-1]):.4f}')
            
            ################ validation ###################
            batch_loss = 0
            total_v=0
            correct_v=0
            with torch.no_grad():
 
                model.eval()
                for d in (val_loader):
                    data_v, target_v = d['img'].to(device), d['lab'].to(device)
                    outputs_v = model(data_v)
                    
                    
                    loss_v = criterion(outputs_v, target_v.long())
                    batch_loss += loss_v.item()
                    _,pred_v = torch.max(outputs_v, dim=1)
                    # print('target_v: ', target_v)
                    # print("pred_v:   ", pred_v)
                    correct_v += torch.sum(pred_v==target_v).item()
                    total_v += target_v.size(0) 
                   
                    # print('pred_v: ', pred_v)
                    # print('target_v: ',target_v)
                    
                    ### visualize validation images   
                    # img = data_v.cpu().numpy()
                    # plt.figure()
                    # plt.axis('off') 
                    # plt.imshow(img[0,0,:,:])
                    
                ### val dice score
                    #metric_val = lo.DiceLoss()
                    #dice_val += 1 - metric_val(outputs, target_v).item()
                    # dice_val += ut.dice_score(target_v.cpu().numpy(), pred_v.cpu().numpy())
                
                    # cm = confusion_matrix(target_v.cpu().numpy(), pred_v.cpu().numpy())
                    # #cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
                    # tn = cm[0,0]
                    # tp = cm[1,1]
                    # fp = cm[0,1]
                    # fn = cm[1,0]
        
                    ### recall_score = tp / (tp + fn)
                    # sensitivity_val += recall_score(target_v.cpu().numpy(), pred_v.cpu().numpy())
               
                    # ### specificity = tn/(tn + fp)
                    # acc, sens, spec = ut.calculate_sensitivity_specificity(target_v.cpu().numpy(), pred_v.cpu().numpy())
                    # specificity_val += spec
                    # acc2_val += acc
    
                
                # val_dice_score.append(dice_val / total_step_val)    
                # val_sens.append(sensitivity_val / total_step_val) 
                # val_spec.append(specificity_val / total_step_val) 
                val_acc.append(100 * correct_v / total_v)
                val_loss.append(batch_loss / total_step_val)
                
        
                
                network_learned = batch_loss < valid_loss_min        
                #print(f'validation loss: {(val_loss[-1]):.4f}, validation acc: {(val_acc[-1]):.4f}, validation dice: {(val_dice_score[-1]):.4f}\n')
                print(f'validation loss: {(val_loss[-1]):.4f}, validation acc: {(val_acc[-1]):.4f}')
                # Saving the best weight 
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), 'models/{}_model_classification_trained.pt'.format(cv_counter))
        
            model.train()
        torch.cuda.empty_cache()
                
        print('best validation loss: ', round(min(val_loss), 2))
        print('best validation acc: ', round(max(val_acc), 2))    
        
        
        ################### loss graphs ###################
        fig = plt.figure(figsize=(20,10))
        plt.title("Loss")
        plt.plot( train_loss, label='train', color='c')
        plt.plot( val_loss, label='validation', color='m')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(new_dir, 'loss'))
        
        ################### accuracy graphs ###################
        fig = plt.figure(figsize=(20,10))
        plt.title(" Accuracy")
        plt.plot(train_acc, label='train', linestyle='dotted', color='c')
        plt.plot(val_acc, label='validation', linestyle='dotted', color='m')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(new_dir, 'accuracy'))
        
        ################### dice score ###################
        # fig = plt.figure(figsize=(20,10))
        # plt.title("Dice score")
        # plt.plot(train_dice_score, label='train', linestyle='dashed',  color='c')
        # plt.plot(val_dice_score, label='validation', linestyle='dashed', color='m')
        # plt.xlabel('num_epochs', fontsize=12)
        # plt.ylabel('dice score', fontsize=12)
        # plt.legend(loc='best')
        # plt.savefig(os.path.join(new_dir, 'dice'))
        
        print('------------------------ cv split done ------------------------')
        
        
    # =============================================================================
    # saving data for final cross validated evaluation        
    # =============================================================================
        final_val_acc.append(max(val_acc))
        cv_counter += 1
        
    
        print('cross validated val accuracy: ', sum(final_val_acc)/args['cv_splits'])
        
        end = time.time()
        elapsed_time = end - start_time
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('time elapsed: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        torch.cuda.empty_cache()


df = pd.DataFrame(data=args, index=[0])
df = (df.T)
df.to_excel('results/dict1.xlsx')


