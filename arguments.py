"""
args
"""

import argparse
import os
from torchvision import datasets



################### hyperparameters ###################

parser = argparse.ArgumentParser()

parser.add_argument('--lr',
        type=float, default=0.0001)

parser.add_argument('--epx',
        type=int, default=400)

parser.add_argument('--batch_size',
        type=int, default=256)

""" seed specifies the way the list is shuffled before split for cross validation, defines test data """
parser.add_argument('--seed',
        type=int, default=3)

""" specifies the cross validation state, None for random """
parser.add_argument('--cv_splits',
        type=int, default=6)

""" train split """
parser.add_argument('--train_split',
        type=int, default=0.9)
# """ val split """
# parser.add_argument('--val_split',
#         type=int, default=0.2)
""" test split """
parser.add_argument('--test_split',
        type=int, default=0.1)


parser.add_argument('--augmentation',
        type=bool, default=False)


# parser.add_argument('--data',
#         type=str, default='data') ###################
parser.add_argument('--data',
        type=str, default='/home/florianh/Desktop/data_f/ewing2/data_all_png_monochcleaned_selected_1/train') ###################


# Gather the parameters to be optimized/updated in this run. If we are#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
parser.add_argument('--feature_extract',
        type=bool, default=True) 


parser.add_argument('--model_name',
        type=str, default='resnet152') 




classes = list()
x = (os.path.join(os.getcwd(), parser.get_default('data')))
for a in os.listdir(x):
    classes.append(a)
num_classes = len(classes)


input_size = 500

args = parser.parse_args()
args = vars(args)