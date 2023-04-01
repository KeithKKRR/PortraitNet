import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets.data_augmentation import DataAugmentation


class EG1800(Dataset):
    def __init__(self, data_root_path, train_or_test):
        self.data_root_path = data_root_path
        self.train_or_test = train_or_test
        if self.train_or_test:  # training data
            self.data_root_path = os.path.join(data_root_path, "training")
        else:  # test data
            self.data_root_path = os.path.join(data_root_path, "testing")

        self.raw_img_name = os.listdir(self.data_root_path)
        self.img_filenames = self.raw_img_name[::2]
        self.mask_filenames = self.raw_img_name[1::2]

    def __getitem__(self, index):
        img_item = Image.open(os.path.join(self.data_root_path, self.img_filenames[index])).convert("RGB")
        mask_item = Image.open(os.path.join(self.data_root_path, self.mask_filenames[index])).convert("1")
        original_img_item, img_item, mask_item, edge = DataAugmentation.process_image(img_item, mask_item)
        # show image test
        # img_item.show()
        # mask_item.show()
        # return img_item, mask_item
        return original_img_item.permute(2, 0, 1), (img_item - DataAugmentation.image_avg).permute(2, 0, 1) * \
                                                   DataAugmentation.image_val, torch.Tensor(mask_item).long(), edge

    def __len__(self):
        return len(self.img_filenames)

# test
# dataset = EG1800("datasets/EG1800", train_or_test=False)
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
# for index, (img, mask, edge) in enumerate(dataloader):
#     if index == 0:
#         print(img.shape)
#         print(mask.shape)
#         print(edge.shape)
#         break
