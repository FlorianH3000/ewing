"""
inference
"""


import matplotlib.pyplot as plt
import torch
import utilz as ut
from arguments import args, num_classes, classes, input_size
from torchvision import transforms
from dataset import ImgDataset
import os
import time
from classification import test_list, norm_mean, norm_std
start_time = time.time()


new_dir = os.path.join('results', args['data'], str(args['lr']), str(args['epx']))

################### dataloader  ###################

transf = transforms.Compose([transforms.Resize(input_size),
                                    transforms.CenterCrop((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)
                                    ])


files = list()
for x in test_list:
    for y in os.listdir(x):
        img = os.path.join(x, y)
        files.append(img)

test_data = ImgDataset(files, transf)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ut.initialize_model(args['model_name'], num_classes, args['feature_extract'], use_pretrained=True)
model.load_state_dict(torch.load('model_classification_trained.pt'))
model.eval()
model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for idx, d in enumerate(test_loader):
        images, label = d['img'], d['lab'].to(device)
        outputs = model(images.to(device))
        # print('outputs: ', outputs)
        _, predicted = torch.max(outputs, 1)
        # print('l: ', label, ' p: ', predicted)

        total += 1
        correct += (label == predicted).sum().item()

        
        img = ut.img_display(images[0,:,:,:])
        plt.figure()
        plt.axis('off')
        
        if label.item() == 0:
            lab='osteom'
        if label.item() == 1:
            lab='ewing'
        # if label.item() == 2:
        #     lab='osteosarcoma'
        # if label.item() == 3:
        #     lab='healthy'

        
        plt.title(lab +'_' + str(classes[predicted[0]][2:]))
        plt.imshow(img)
        plt.savefig(os.path.join(new_dir, 'prediction_' + str(idx)))
        
    print('accuracy of testing: ', correct/total)
end = time.time()
elapsed_time = end - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('time for prediction: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
