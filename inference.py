"""
inference
"""

import matplotlib.pyplot as plt
import torch
import utilz as ut
from arguments import args, num_classes, classes, input_size
from torchvision import transforms
from dataset import ImgDataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, precision_score, recall_score
import os
import time
from cv_classification import labels_test_list1, img_test_list1, norm_mean, norm_std
start_time = time.time()


new_dir = os.path.join('results', args['data'], str(args['lr']), str(args['epx']))

################### dataloader  ###################

transf = transforms.Compose([transforms.Resize(input_size),
                                    transforms.CenterCrop((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)
                                    ])


test_data = ImgDataset(img_test_list1, labels_test_list1, transf)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ut.initialize_model(args['model_name'], num_classes, args['feature_extract'], use_pretrained=True)
final_acc_test = list()
final_f1_test = list()
final_precision_test = list()
final_recall_test = list()
final_specificity_test = list()


for x in os.listdir(os.path.join(os.getcwd(), 'models')):
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', x)))
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    l = list()
    p = list()
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            images, label = d['img'], d['lab'].to(device)
            outputs = model(images.to(device))
            # print('outputs: ', outputs)
            _, predicted = torch.max(outputs, 1)
            # print('l: ', label, ' p: ', predicted)
    
            total += 1
            correct += (label == predicted).sum().item()
            l.append(label.cpu().numpy().item())
            p.append(predicted.cpu().numpy().item())
            
            if label.item() == 0:
                lab='neurogen'
            if label.item() == 1:
                lab='strukturell'
    
            # img = ut.img_display(images[0,:,:,:])
            # plt.figure(figsize=(15,15))
            # plt.axis('off')
            # plt.title(lab +'_' + str(classes[predicted[0]][2:]))
            # plt.imshow(img, cmap='bone')
            # plt.savefig(os.path.join(new_dir, 'prediction_' + str(idx)))
        
            
        label = label.cpu()
        predicted = predicted.cpu()
        cm = confusion_matrix(l, p)
        cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title('Confusion Matrix')
        tn = cm[0,0]
        tp = cm[1,1]
        fp = cm[0,1]
        fn = cm[1,0]
        
        ## accuracy
        acc = correct/total
        # print('accuracy of testing: ', acc)
        final_acc_test.append(acc)
        
        ## f1_score = 2 * (pre * rec) / (pre + rec)
        f1 = f1_score(l, p)
        # print('f1/ Dice : ', round(f1, 3))
        final_f1_test.append(f1)
        
        # precision_score = tp / (tp + fp)
        precision = precision_score(l, p)
        # print('precision: ', round(precision, 3))
        final_precision_test.append(precision)
        
        ### recall_score = tp / (tp + fn)
        recall = recall_score(l, p)
        # print('recall/ sensitivity: ', round(recall, 3))
        final_recall_test.append(recall)
        
        ### specificity = tn/(tn + fp)
        specificity = tn/(tn + fp)
        # print('specificity : ', round(specificity, 3))
        final_specificity_test.append(specificity)


print('--------------')
print('Final Accuracy Test: ', round(sum(final_acc_test)/args['cv_splits'], 2))
print('Final f1/ Dice Test: ', round(sum(final_f1_test)/args['cv_splits'], 2))
print('Final Precision Test: ', round(sum(final_precision_test)/args['cv_splits'], 2))
print('Final Recall Test: ', round(sum(final_recall_test)/args['cv_splits'], 2))
print('Final Specificity Test: ', round(sum(final_specificity_test)/args['cv_splits'], 2))



end = time.time()
elapsed_time = end - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('time for prediction: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

