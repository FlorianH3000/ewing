"""
Classification

"""
import torch
from torchvision import transforms
import utilz as ut
import losses as lo
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from dataset import ImgDataset
import torch.nn as nn
import arguments
from arguments import args, num_classes, input_size, classes
import time
import os
start_time = time.time()
from PIL import ImageFile
from sklearn.model_selection import train_test_split
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# CUDA_LAUNCH_BLOCKING=1


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
                                        # transforms.CenterCrop((input_size, input_size)),
                                        # transforms.ToTensor(),
                                        # transforms.ToPILImage(),
                                        # transforms.RandomRotation(30),
                                        # transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.RandomVerticalFlip(p=0.5),
                                        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=2, fill=0),
                                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        # transforms.RandomCrop(input_size*0.95),
                                        # transforms.ToTensor(),
                                        # transforms.Normalize(norm_mean, norm_std),                               
                                        ])
else:
    transf_train = transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.Resize(input_size),
                                # transforms.CenterCrop((input_size, input_size)),                               
                                # transforms.ToTensor(),
                                # transforms.Normalize(norm_mean, norm_std),                               
                                ])

transf_val = transforms.Compose([
                                # transforms.ToPILImage(),
                                # transforms.Resize(input_size),
                                # transforms.CenterCrop((input_size, input_size)),                                    
                                # transforms.ToTensor(),
                                # transforms.Normalize(norm_mean, norm_std),                                
                                    ])




file_paths = list()
img_train = list()
img_val = list()
img_test = list()
osteom_list = list()
ewing_list = list()
osteosarcoma_list = list()
healthy_list = list()


file_paths = list()
for x in os.listdir(args['data_dir_train']):
    y = os.path.join(args['data_dir_train'], x)
    for a in os.listdir(y):
        file_paths.append(os.path.join(args['data_dir_train'], x , a))


### seperate classes to get balanced split
for x in file_paths:
    if '0_osteom' in x:
        osteom_list.append(x)
    if '1_ewing' in x:
        ewing_list.append(x)
    # if '2_osteosarcoma' in x:
    #     osteosarcoma_list.append(x)
    # if '3_healthy' in x:
    #     healthy_list.append(x)


### apply split to both classes
### cv_state 1-5 possible, 0 for shuffle
labels_list = list()


train_list_0, test_list_0 = train_test_split(osteom_list, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed']) # train+val and test
train_list_0, val_list_0 = train_test_split(train_list_0, test_size=args['val_split'], random_state=args['cv_state']) # train and val
train_list_1, test_list_1 = train_test_split(ewing_list, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'])
train_list_1, val_list_1 = train_test_split(train_list_1, test_size=args['val_split'], random_state=args['cv_state'])
# train_list_2, test_list_2 = train_test_split(osteosarcoma_list, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'])
# train_list_2, val_list_2 = train_test_split(train_list_2, test_size=args['val_split'], random_state=args['cv_state']) 
# train_list_3, test_list_3 = train_test_split(healthy_list, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'])
# train_list_3, val_list_3 = train_test_split(train_list_3, test_size=args['val_split'], random_state=args['cv_state'])




train_list = train_list_0 + train_list_1 #+ train_list_2 + train_list_3
val_list = val_list_0 + val_list_1 #+ val_list_2 + val_list_3
test_list = test_list_0 + test_list_1 #+ test_list_2 #+ test_list_3


for x in train_list:
    z = os.path.join(os.getcwd(), x)
    #img = plt.imread(z)
    img_train.append(z)
    ### for mfb
    if '0_osteom' in z:
        labels_list.append(0)
    if '1_ewing' in z:
        labels_list.append(1)
    # if '2_osteosarcoma' in z:
    #     labels_list.append(2)
    # if '3_healthy' in z:
    #     labels_list.append(3)
for x in val_list:
    z = os.path.join(os.getcwd(), x)
    img_val.append(z)

for x in test_list:
    z = os.path.join(os.getcwd(), x)
    img_test.append(z)
        
print('number of files for training: ', len(img_train))
print('number of files for validation: ', len(img_val))
print('number of files for testing: ', len(img_test))


### custom dataloader
train_data = ImgDataset(img_train, transf_train)
val_data = ImgDataset(img_val, transf_val)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args['batch_size'], shuffle=True, num_workers=0)



if __name__ == "__main__":

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
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    
    
    
    # =============================================================================
    # median frequency balancing
    # =============================================================================
    # get the class labels of each image
    class_labels = labels_list
    # empty array for counting instance of each class
    count_labels = np.zeros(len(classes))
    # empty array for weights of each class
    class_weights = np.zeros(len(classes))
    
    # populate the count array
    for l in class_labels:
      count_labels[l] += 1
    print('count_labels: ', count_labels)
    
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
    val_dice_score = []
    total_step_train = len(train_loader)
    total_step_val = len(val_loader)
    
    for epoch in range(1, args['epx']+1):
        running_loss = 0.0
        correct_t = 0
        total_t = 0
        dice_train, dice_val = 0,0
        print('----------------------------------')
        print(f'Epoch {epoch}')
        
        
        for batch_idx, d in enumerate(train_loader):
            data_t, target_t = d['img'].to(device), d['lab'].to(device)
            
            ### zero the parameter gradients
            optimizer.zero_grad()
            ### forward + backward + optimize
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            
            loss_t.backward()
            optimizer.step()
            ### print statistics
            running_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
            
        ### train dice score, acc
            dice_train += ut.dice_score(target_t.cpu().numpy(),pred_t.cpu().numpy())        
        
            # ## visualize training images
            # if batch_idx < 10:    
            #     img = data_t.cpu().numpy()
            #     plt.figure(dpi=200)
            #     plt.axis('off') 
            #     plt.imshow(img[0,0,:,:], cmap='bone')
            #     plt.imsave(r'\\nas.ads.mwn.de\ga87qis\Desktop\code\ewing vs om\results\test{}.png'.format(batch_idx), img[0,2,:,:], dpi=500, cmap='bone')
        
        
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
                
                
                loss_v = criterion(outputs_v, target_v)
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
                dice_val += ut.dice_score(target_v.cpu().numpy(), pred_v.cpu().numpy())
            
            val_dice_score.append(dice_val / total_step_val)        
            val_acc.append(100 * correct_v / total_v)
            val_loss.append(batch_loss / total_step_val)
            
    
            
            network_learned = batch_loss < valid_loss_min        
            print(f'validation loss: {(val_loss[-1]):.4f}, validation acc: {(val_acc[-1]):.4f}, validation dice: {(val_dice_score[-1]):.4f}\n')
            # Saving the best weight 
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), 'model_classification_trained.pt')
    
        model.train()
            
    print('best validation loss: ', min(val_loss))
    print('best validation acc: ', max(val_acc))    
         
    
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
    fig = plt.figure(figsize=(20,10))
    plt.title("Dice score")
    plt.plot(train_dice_score, label='train', linestyle='dashed',  color='c')
    plt.plot(val_dice_score, label='validation', linestyle='dashed', color='m')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('dice score', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(os.path.join(new_dir, 'dice'))
    
    
    end = time.time()
    elapsed_time = end - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print('time elapsed: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    
    del target_t, target_v, pred_t, pred_v, correct_t, correct_v, loss_t, loss_v, model, dice_train, dice_val
    del train_acc, train_data, train_dice_score, train_loader, train_loss, val_acc, val_data, val_dice_score, val_loader, val_loss, valid_loss_min
    del optimizer, fig, device, data_t, data_v, outputs_t, outputs_v, args
    del batch_idx, batch_loss, elapsed_time, end, epoch, network_learned, new_dir, running_loss, start_time, total_t, total_step_train, total_v
    torch.cuda.empty_cache()
