import sys
import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)

        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):

        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):

        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, label):

        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):

        return F.to_tensor(pic), F.to_tensor(label)


class Setdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = DataProcessing(os.path.join(self.config.data.data_dir, 'train',"low"),
                                          patch_size=self.config.data.patch_size,
                                          filelist='{}_train.txt'.format(self.config.data.data_dir))
        val_dataset = DataProcessing(os.path.join(self.config.data.data_dir, 'val','low'),
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.data_dir), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class DataProcessing(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        print("File Exist：", self.file_list)
        print(os.path.exists(self.file_list))


        if not os.path.exists(self.file_list):
            paths = self.dir
            output_file = self.file_list
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            def sort_filenames(filename):
                digits = ''.join(filter(str.isdigit, filename))
                if digits:
                  return int(digits)
                else:
                  return float('inf')  

            filenames = sorted(os.listdir(paths), key=sort_filenames)
            with open(output_file, 'w') as f:
                for filename in filenames:
                    if os.path.splitext(filename)[1] == '.JPG' or os.path.splitext(filename)[1] == '.png':
                       f.write(filename + '\n')
            f.close()


        with open(self.file_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('low', 'normal') for i in input_names]


        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)






