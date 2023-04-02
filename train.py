import os
import time

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets.eg1800loader import EG1800
from models.model_mobilenetv2_seg_small import MobileNetV2
from utils.logger import Logger
from utils.loss import FocalLoss, loss_KL
from utils.metrices import calcIOU
from utils.tools import AverageMeter, Anti_Normalize_Img, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model and data
model_name = "PortraitNet"
dataset = "EG1800"
dataset_root_path = "datasets/EG1800"
model_root_path = "models"
log_root_path = "logs"
img_scale = 1
img_mean = [103.94, 116.78, 123.68]  # BGR order, image mean
img_val = [0.017, 0.017, 0.017]  # BGR order, image val

# train hyperparameters
learning_rate = 0.001
weight_decay = 5e-4
max_epoch = 2000

# model hyperparameters
# whether to add boundary auxiliary loss
addEdge = True
# the weight of boundary auxiliary loss
edgeRatio = 0.1
# whether to add consistency constraint loss
stability = False
# whether to use KL loss in consistency constraint loss
use_kl = True
# temperature in consistency constraint loss
temperature = 1
# the weight of consistency constraint loss
alpha = 2
# whether to use pretrain model to init portrait-net
init = True
# whether to continue training
resume = False
# if useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
useUpsample = False
# if useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
useDeconvGroup = False

# log parameter
log_epoch = 10


def adjust_learning_rate(optimizer, epoch, multiple):
    """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
    lr = learning_rate * (0.95 ** (epoch // 20))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    pass


# Copied from official code
def get_parameters(model, useDeconvGroup=True):
    lr_0 = []
    lr_1 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'deconv' in key and useDeconvGroup == True:
            print("useDeconvGroup=True, lr=0, key: ", key)
            lr_0.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_0, 'lr': learning_rate * 0},
              {'params': lr_1, 'lr': learning_rate * 1}]
    return params, [0., 1.]


'''
- Train and test
'''


def evaluate(dataLoader, model, optimizer, epoch, logger):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('losses')
    losses_mask_ori = AverageMeter('losses_mask_ori')
    losses_mask = AverageMeter('losses_mask')
    losses_edge_ori = AverageMeter('losses_edge_ori')
    losses_edge = AverageMeter('losses_edge')
    losses_stability_mask = AverageMeter('losses_stability_mask')
    losses_stability_edge = AverageMeter('losses_stability_edge')

    # switch to eval mode
    model.eval()

    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)  # mask loss
    loss_Focalloss = FocalLoss(gamma=2)  # edge loss
    loss_l2 = nn.MSELoss()  # edge loss

    end = time.time()
    softmax = nn.Softmax(dim=1)
    iou = 0

    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):
        data_time.update(time.time() - end)
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())

        if addEdge:
            output_mask, output_edge = model(input_var)
            loss_mask = loss_Softmax(output_mask, torch.Tensor(mask_var).long())
            losses_mask.update(loss_mask.item(), input.size(0))
            # loss_edge = loss_l2(output_edge, edge_var) * edgeRatio
            loss_edge = loss_Focalloss(output_edge, edge_var) * edgeRatio
            losses_edge.update(loss_edge.item(), input.size(0))
            loss = loss_mask + loss_edge

            if stability:
                output_mask_ori, output_edge_ori = model(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, torch.Tensor(mask_var).long())
                losses_mask_ori.update(loss_mask_ori.item(), input.size(0))
                # loss_edge_ori = loss_l2(output_edge_ori, edge_var) * edgeRatio
                loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * edgeRatio
                losses_edge_ori.update(loss_edge_ori.item(), input.size(0))

                if not use_kl:
                    # consistency constraint loss: L2 distance
                    loss_stability_mask = loss_l2(output_mask,
                                                  Variable(output_mask_ori.data, requires_grad=False)) * alpha
                    loss_stability_edge = loss_l2(output_edge, Variable(output_edge_ori.data,
                                                                        requires_grad=False)) * alpha * edgeRatio
                else:
                    # consistency constraint loss: KL distance
                    loss_stability_mask = loss_KL(output_mask, Variable(output_mask_ori.data, requires_grad=False),
                                                  temperature) * alpha
                    loss_stability_edge = loss_KL(output_edge, Variable(output_edge_ori.data, requires_grad=False),
                                                  temperature) * alpha * edgeRatio

                losses_stability_mask.update(loss_stability_mask.item(), input.size(0))
                losses_stability_edge.update(loss_stability_edge.item(), input.size(0))

                # total loss
                # loss = loss_mask + loss_mask_ori + loss_edge + loss_edge_ori + loss_stability_mask + loss_stability_edge
                loss = loss_mask + loss_mask_ori + loss_stability_mask + loss_edge
        else:
            output_mask = model(input_var)
            loss_mask = loss_Softmax(output_mask, torch.Tensor(mask_var).long())
            losses_mask.update(loss_mask.item(), input.size(0))
            loss = loss_mask

            if stability:
                # loss part2
                output_mask_ori = model(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, torch.Tensor(mask_var).long())
                losses_mask_ori.update(loss_mask_ori.item(), input.size(0))
                if not use_kl:
                    # consistency constraint loss: L2 distance
                    loss_stability_mask = loss_l2(output_mask,
                                                  Variable(output_mask_ori.data, requires_grad=False)) * alpha
                else:
                    # consistency constraint loss: KL distance
                    loss_stability_mask = loss_KL(output_mask, Variable(output_mask_ori.data, requires_grad=False),
                                                  temperature) * alpha
                losses_stability_mask.update(loss_stability_mask.item(), input.size(0))
                # total loss
                loss = loss_mask + loss_mask_ori + loss_stability_mask

        # total loss
        loss = loss_mask
        losses.update(loss.item(), input.size(0))

        prob = softmax(output_mask)[0, 1, :, :]
        pred = prob.data.cpu().numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        iou += calcIOU(pred, mask_var[0].data.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_epoch == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr-deconv: [{3}]\t'
                  'Lr-other: [{4}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(dataLoader),
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
                loss=losses))

    # return losses.avg
    return 1 - iou / len(dataLoader)


def train(dataLoader, model, optimizer, epoch, logger):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('losses')
    losses_mask = AverageMeter('losses_mask')

    if addEdge:
        losses_edge_ori = AverageMeter('losses_edge_ori')
        losses_edge = AverageMeter('losses_edge')

    if stability:
        losses_mask_ori = AverageMeter('losses_mask_ori')
        losses_stability_mask = AverageMeter('losses_stability_mask')
        losses_stability_edge = AverageMeter('losses_stability_edge')

    model.train()  # switch to train mode

    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)  # mask loss
    # in our experiments, focalloss is better than l2 loss
    loss_Focalloss = FocalLoss(gamma=2)  # boundary loss
    loss_l2 = nn.MSELoss()  # boundary loss

    end = time.time()
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):
        data_time.update(time.time() - end)
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())

        if addEdge:
            output_mask, output_edge = model(input_var)
            loss_mask = loss_Softmax(output_mask, torch.Tensor(mask_var).long())
            losses_mask.update(loss_mask.item(), input.size(0))

            # loss_edge = loss_l2(output_edge, edge_var) * edgeRatio
            loss_edge = loss_Focalloss(output_edge, edge_var) * edgeRatio
            losses_edge.update(loss_edge.item(), input.size(0))

            # total loss
            loss = loss_mask + loss_edge

            if stability:
                output_mask_ori, output_edge_ori = model(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, torch.Tensor(mask_var).long())
                losses_mask_ori.update(loss_mask_ori.item(), input.size(0))

                # loss_edge_ori = loss_l2(output_edge_ori, edge_var) * edgeRatio
                loss_edge_ori = loss_Focalloss(output_edge_ori, edge_var) * edgeRatio
                losses_edge_ori.update(loss_edge_ori.item(), input.size(0))

                # in our experiments, kl loss is better than l2 loss
                if not use_kl:
                    # consistency constraint loss: L2 distance
                    loss_stability_mask = loss_l2(output_mask,
                                                  Variable(output_mask_ori.data, requires_grad=False)) * alpha
                    loss_stability_edge = loss_l2(output_edge, Variable(output_edge_ori.data,
                                                                        requires_grad=False)) * edgeRatio
                else:
                    # consistency constraint loss: KL distance (better than L2 distance)
                    loss_stability_mask = loss_KL(output_mask, Variable(output_mask_ori.data, requires_grad=False),
                                                  temperature) * alpha
                    loss_stability_edge = loss_KL(output_edge, Variable(output_edge_ori.data, requires_grad=False),
                                                  temperature) * alpha * edgeRatio

                losses_stability_mask.update(loss_stability_mask.item(), input.size(0))
                losses_stability_edge.update(loss_stability_edge.item(), input.size(0))

                # total loss
                # loss = loss_mask + loss_mask_ori + loss_edge + loss_edge_ori + loss_stability_mask + loss_stability_edge
                loss = loss_mask + loss_mask_ori + loss_stability_mask + loss_edge
        else:
            output_mask = model(input_var)
            loss_mask = loss_Softmax(output_mask, torch.Tensor(mask_var).long())
            losses_mask.update(loss_mask.item(), input.size(0))
            # total loss: only include mask loss
            loss = loss_mask

            if stability:
                output_mask_ori = model(input_ori_var)
                loss_mask_ori = loss_Softmax(output_mask_ori, torch.Tensor(mask_var).long())
                losses_mask_ori.update(loss_mask_ori.item(), input.size(0))
                if not use_kl:
                    # consistency constraint loss: L2 distance
                    loss_stability_mask = loss_l2(output_mask,
                                                  Variable(output_mask_ori.data, requires_grad=False)) * alpha
                else:
                    # consistency constraint loss: KL distance (better than L2 distance)
                    loss_stability_mask = loss_KL(output_mask, Variable(output_mask_ori.data, requires_grad=False),
                                                  temperature) * alpha
                losses_stability_mask.update(loss_stability_mask.item(), input.size(0))

                # total loss
                loss = loss_mask + loss_mask_ori + loss_stability_mask

        losses.update(loss.item(), input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_epoch == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr-deconv: [{3}]\t'
                  'Lr-other: [{4}]\t'
            # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(dataLoader),
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
                loss=losses))


'''
- Datasets
'''
train_dataset = EG1800(dataset_root_path, train_or_test=True)
test_dataset = EG1800(dataset_root_path, train_or_test=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

'''
- Model
'''
portrait_net = MobileNetV2(n_class=2, useUpsample=useUpsample, useDeconvGroup=useDeconvGroup, addEdge=addEdge,
                           channelRatio=1.0, minChannel=16, weightInit=True).cuda()

'''
- Optimizer
'''
params, multiple = get_parameters(portrait_net, useDeconvGroup)
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

if init:
    pretrained_state_dict = torch.load("pretrained/pretrained_mobilenetv2_base.pth")
    pretrained_state_dict_keys = list(pretrained_state_dict.keys())
    portrait_net_state_dict = portrait_net.state_dict()
    portrait_net_state_dict_keys = list(portrait_net_state_dict.keys())
    print("pretrain keys: ", len(pretrained_state_dict_keys))
    print("netmodel keys: ", len(portrait_net_state_dict_keys))
    weights_load = {}
    for k in range(len(pretrained_state_dict_keys)):
        if pretrained_state_dict[pretrained_state_dict_keys[k]].shape == portrait_net_state_dict[ portrait_net_state_dict_keys[k]].shape:
            weights_load[portrait_net_state_dict_keys[k]] = pretrained_state_dict[pretrained_state_dict_keys[k]]
            print('init model', portrait_net_state_dict_keys[k], 'from pretrained', pretrained_state_dict_keys[k])
        else:
            break
    print("init len is:", len(weights_load))
    portrait_net_state_dict.update(weights_load)
    portrait_net.load_state_dict(portrait_net_state_dict)
    print("load model init finish...")
portrait_net.to(device)

if resume:
    bestModelFile = os.path.join(model_root_path, 'model_best.pth.tar')
    if os.path.isfile(bestModelFile):
        checkpoint = torch.load(bestModelFile)
        portrait_net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        gap = checkpoint['epoch']
        minLoss = checkpoint['minLoss']
        print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(bestModelFile))
else:
    minLoss = 10000
    gap = 0

logger_train = Logger(log_root_path + 'train')
logger_test = Logger(log_root_path + 'test')

for epoch in range(gap, 2000):
    adjust_learning_rate(optimizer, epoch, multiple)
    print('===========>   training    <===========')
    train(train_dataloader, portrait_net, optimizer, epoch, logger_train)
    print('===========>   testing    <===========')
    loss = evaluate(test_dataloader, portrait_net, optimizer, epoch, logger_test)
    print("loss: ", loss, minLoss)
    is_best = False
    if loss < minLoss:
        minLoss = loss
        is_best = True

    save_checkpoint({'epoch': epoch + 1, 'minLoss': minLoss, 'state_dict': portrait_net.state_dict(),
                     'optimizer': optimizer.state_dict()}, is_best, model_root_path)
