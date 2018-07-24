import torch
from torch.utils import data
import torchvision.transforms as transforms

import numpy as np
import os.path as osp
from PIL import Image
import joint_transforms


class RSData(data.Dataset):
    def __init__(self, mode, data_root, label_root, label_list, transforms = None):
        self.mode = mode
        self.data_root = data_root
        self.label_root = label_root
        self.label_list = label_list
        self.transforms = transforms
        self.im2te = ImToTensor()
        self.lb2te = MaskToTensor()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        img_name = osp.join(self.data_root, self.label_list[item].replace('label_noBoundary', 'RGB') + '.tif')
        label_name = osp.join(self.label_root, self.label_list[item] + '.tif')
        img, label = Image.open(img_name).convert('RGB'), Image.open(label_name)
        if self.mode == 'train' and self.transforms is not None:
            img, label = self.transforms(img, label)
        return self.im2te(img), self.lb2te(label)


class ImToTensor(object):
    def __call__(self, im):
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[x/255.0 for x in [85.86, 91.79, 85.00]],
            #                      std=[x/255.0 for x in [35.79, 35.13, 36.51]]),
        ])
        return trans(im)


class MaskToTensor(object):
    def __call__(self, label):
        return torch.from_numpy(np.array(label, dtype=np.int32)).long()


if __name__ == '__main__':
    label = [i_id.strip() for i_id in open('./list/top_potsdam.txt')]
    transform_train = joint_transforms.Compose([
            joint_transforms.RandomCrop(384),
            joint_transforms.Scale(400),
            joint_transforms.RandomRotate(10),
            joint_transforms.RandomHorizontallyFlip(),
    ])
    dataset = RSData('train', '/home/jinqizhao/dataset/image/Remote_sensing/potsdam/2_Ortho_RGB_seg/',
                     '/home/jinqizhao/dataset/image/Remote_sensing/potsdam/Label_gray/', label, transforms=transform_train)
    img, label = dataset[0]
    print(1)
