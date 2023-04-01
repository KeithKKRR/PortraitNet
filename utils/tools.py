import shutil

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.sum = None
        self.count = None
        self.avg = None
        self.val = None
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, root + filename)
    if is_best:
        shutil.copyfile(root + filename, root + 'model_best.pth.tar')


def Anti_Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j, :, :, i] = img[j, :, :, i] / val[i] + mean[i]
        return np.array(img * scale, np.uint8)
    else:
        for i in range(len(mean)):
            img[:, :, i] = img[:, :, i] / val[i] + mean[i]
        return np.array(img * scale, np.uint8)
