import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torch.utils.tensorboard


from easydict import EasyDict as edict


# ============================ main engine for training and evaluation ============================


def L1_loss(pred,label=0):
    loss = (pred.contiguous()-label).abs()
    return loss.mean()

def MSE_loss(pred,label=0):
    loss = (pred.contiguous()-label)**2
    return loss.mean()
