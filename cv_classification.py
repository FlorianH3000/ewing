
"""
Classification cv stratified

"""
import sys
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
from arguments import args, num_classes, classes
import torch.nn as nn
from dataset import ImgDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import utilz as ut
import torch
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
start_time = time.time()
torch.backends.cudnn.benchmark = True

################### directory for saving results ###################
new_dir = os.path.join(os.getcwd(), 'results/', args['id'])
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
    print('Directory ', new_dir, 'created!')
else:
    print('Directory ', new_dir, 'already exists!')

file_paths = list()
labels_list_initial = list()
img_val = list()


for x in os.listdir(args['data']):
    y = os.path.join(args['data'], x)
    for a in os.listdir(y):
        z = os.path.join(args['data'], x, a)
        file_paths.append(z)
        labels_list_initial.append(int(x[:1]))

# split list into train and test
# split train into train and val cross validated
pat_train_val, pat_test, pat_label_train_val, pat_label_test = train_test_split(
    file_paths, labels_list_initial, test_size=args['test_split'], train_size=args['train_split'], random_state=args['seed'], stratify=labels_list_initial)


# =============================================================================
# cross validation
# =============================================================================
skf = StratifiedKFold(n_splits=args['cv_splits'])
skf.get_n_splits(pat_train_val, pat_label_train_val)
final_val_acc = list()
final_val_spec = list()
final_val_sens = list()


cv_counter = 1
pat_train_val, pat_label_train_val = np.array(
    pat_train_val), np.array(pat_label_train_val)
pat_test = np.array(pat_test)

for train_index, val_index in skf.split(pat_train_val, pat_label_train_val):
    img_train, img_val = pat_train_val[train_index], pat_train_val[val_index]

    labels_train_list_final = list()
    labels_val_list_final = list()
    labels_test_list_final = list()
    img_train_list_final = list()
    img_val_list_final = list()
    img_test_list_final = list()

# train
    for img_path in img_train:
        for x in os.listdir(img_path):
            if 'png' in x:
                img_train_list_final.append(os.path.join(img_path, x))
                labels_train_list_final.append(
                    int(img_path.split('\\')[1][0:3]))

# val
    for img_path in img_val:
        for x in os.listdir(img_path):
            if 'png' in x:
                img_val_list_final.append(os.path.join(img_path, x))
                labels_val_list_final.append(
                    int(img_path.split('\\')[1][0:3]))

# test
    for img_path in pat_test:
        for x in os.listdir(img_path):
            if 'png' in x:
                img_test_list_final.append(os.path.join(img_path, x))
                labels_test_list_final.append(
                    int(img_path.split('\\')[1][0:3]))



    if __name__ == "__main__":

        if cv_counter == 1:
            print('number of files for training: ', len(img_train_list_final))
            print('number of files for validation: ', len(img_val_list_final))
            print('number of files for testing: ', len(img_test_list_final))

        # custom dataloader
        train_data = ImgDataset(img_train_list_final,
                                labels_train_list_final, 'train')
        val_data = ImgDataset(img_val_list_final,
                              labels_val_list_final, 'val')

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args['batch_size'], shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args['batch_size'], shuffle=True, num_workers=0)

        # =============================================================================
        # architecture
        # =============================================================================

        # Initialize the model
        model = ut.initialize_model(
            args['model_name'], num_classes, args['feature_extract'], use_pretrained=True)

        params_to_update = model.parameters()
        #print("Params to learn:")
        if args['feature_extract']:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    # print("\t",name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # =============================================================================
        # median frequency balancing
        # =============================================================================
        # get the class labels of each image
        class_labels = labels_train_list_final
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
            print(i)
            print(class_weights[i])
            class_weights[i] = median_freq/count_labels[i]
            # print the weights
            print('weights: ', classes[i], ":", class_weights[i])

        # =============================================================================
        #  optimizer, loss function
        # =============================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(class_weights)
        optimizer = optim.Adam(params_to_update, lr=args['lr'])
        # model = torch.nn.DataParallel(model)
        model.to(device)

        ################### training ###################
        valid_loss_min = 10000
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []

        val_sens, val_spec = [], []
        total_step_train = len(train_loader)
        total_step_val = len(val_loader)
        lab_list, pred_list = list(), list()


        for epoch in range(1, args['epx']+1):
            running_loss = 0.0
            correct_t = 0
            total_t = 0
            sensitivity_train, specificity_val_train, sensitivity_val, specificity_val = 0, 0, 0, 0
            print('----------------------------------')
            print(f'Epoch {epoch}')

            for batch_idx, d in enumerate(train_loader):
                data_t, target_t = d['img'].to(device), d['lab'].to(device)
                ### zero the parameter gradients
                optimizer.zero_grad()
                ### forward + backward + optimize
                outputs_t = model(data_t.float())

                loss_t = criterion(outputs_t, target_t.long())

                loss_t.backward()
                optimizer.step()
                ### print statistics
                running_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)

                ### visualize training images
                # if batch_idx < 20:
                #     img = data_t.cpu().numpy()
                #     plt.figure(dpi=300)
                #     plt.axis('off')
                #     plt.imshow(img[0,0,:,:], cmap='bone')


            train_acc.append(100 * (correct_t / total_t))
            train_loss.append(running_loss / total_step_train)
            print(
                f'\ntrain loss: {(train_loss[-1]):.4f}, train acc: {(train_acc[-1]):.4f}')

            ################ validation ###################
            batch_loss = 0
            total_v = 0
            correct_v = 0
            with torch.no_grad():
                model.eval()
                for d in (val_loader):
                    data_v, target_v = d['img'].to(device), d['lab'].to(device)
                    outputs_v = model(data_v.float())

                    [t.cpu().numpy() for t in target_v]
                    lab_list.append(target_v)

                    loss_v = criterion(outputs_v, target_v.long())
                    batch_loss += loss_v.item()
                    pred, pred_v = torch.max(outputs_v, dim=1)
                    pred = pred.cpu()  # .numpy()

                    [p.cpu().numpy() for p in pred_v]
                    pred_list.append(pred_v)

                    correct_v += torch.sum(pred_v == target_v).item()
                    total_v += target_v.size(0)

                    ### visualize validation images
                    # img = data_v.cpu().numpy()
                    # plt.figure()
                    # plt.axis('off')
                    # plt.imshow(img[0,0,:,:])

                ### specificity = tn/(tn + fp)
                sens, spec, _ = ut.calculate_sensitivity_specificity(
                    lab_list, pred_list, num_classes)




                val_acc.append(100*correct_v/total_v)
                val_spec.append(spec)
                val_sens.append(sens)
                val_loss.append(batch_loss / total_step_val)

                network_learned = batch_loss < valid_loss_min

                print('validation loss: ', round(val_loss[-1], 3))
                print('validation acc: ', round(val_acc[-1], 3))
                print('validation spec: ', round(val_spec[-1], 3))
                print('validation sens: ', round(val_sens[-1], 3))
                # Saving the best weight
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), os.path.join(new_dir,
                                                                '{}_model_classification_trained.pt'.format(cv_counter)))

            model.train()

        torch.cuda.empty_cache()

        print('best validation loss: ', round(min(val_loss), 3))
        print('best validation acc: ', round(max(val_acc), 3))
        print('best validation spec: ', round(max(val_spec), 3))
        print('best validation sens: ', round(max(val_sens), 3))

        final_dir = os.path.join(new_dir, str(cv_counter))
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            print('Directory for result: ', final_dir, 'created!')
        else:
            print('Directory for result: ', final_dir, 'already exists!')
 
        ################### loss graphs ###################
        fig = plt.figure(figsize=(20, 10))
        plt.title("Loss")
        plt.plot(train_loss, label='train', color='c')
        plt.plot(val_loss, label='validation', color='m')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(final_dir, 'loss'))
        plt.close()

        ################### accuracy graphs ###################
        fig = plt.figure(figsize=(20, 10))
        plt.title(" Accuracy")
        plt.plot(train_acc, label='train', linestyle='dotted', color='c')
        plt.plot(val_acc, label='validation', linestyle='dotted', color='m')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(os.path.join(final_dir, 'accuracy'))
        plt.close()


        
        print('------------------------ cv split done ------------------------')

    # =============================================================================
    # saving data for final cross validated evaluation
    # =============================================================================
        final_val_acc.append(max(val_acc))
        final_val_sens.append(max(val_sens))
        final_val_spec.append(max(val_spec))

        print('cv_counter: ', str(cv_counter) + '/' + str(args['cv_splits']))
        cv_counter += 1

        print()
        print()
        print(final_dir)
        print('------------------------ FINAL RESULTS ------------------------')
        print('cross validated val accuracy: ', round(np.mean(final_val_acc), 3))
        print('cross validated val sensitivity: ', round(np.mean(final_val_sens), 3))
        print('cross validated val specificity: ', round(np.mean(final_val_spec), 3))
        print('---------------------------------------------------------------')

        end = time.time()
        elapsed_time = end - start_time
        print('elapsed mins: ', elapsed_time/60)
        print('elapsed hours: ', elapsed_time/3600)
        torch.cuda.empty_cache()
        
    
            
        

    if cv_counter > args['cv_splits']:
        with open(os.path.join(new_dir, 'args.txt'), 'w') as file:
            file.write(json.dumps(args))




