import argparse
import shutil
import time

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import os
import numpy as np
import dataset.dataset as dataset
import densenet as dn
import dataset.joint_transforms as joint_transforms
from evaluation import evaluate

parser = argparse.ArgumentParser(description='Pytorch RemoteNet Test')
parser.add_argument('--data-root', default='/home/jinqizhao/dataset/image/Remote_sensing/potsdam/2_Ortho_RGB_seg/', type=str,
                    help='path to data')
parser.add_argument('--label-root', default='/home/jinqizhao/dataset/image/Remote_sensing/potsdam/Label_gray/', type=str,
                    help='path to label')
parser.add_argument('--label-list', default='./dataset/list/top_potsdam.txt', type=str,
                    help='label list')
parser.add_argument('--resume', default='', type=str,
                    help='path to latset checkpoint(default: None')
parser.set_defaults(augment=True)
