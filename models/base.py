import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torch.utils.tensorboard

from math import log10, sqrt
import cv2
import numpy as np
import lpips
from utils import camera


from easydict import EasyDict as edict


# ============================ main engine for training and evaluation ============================


def L1_loss(pred,label=0):
    loss = (pred.contiguous()-label).abs()
    return loss.mean()

def MSE_loss(pred,label=0):
    loss = (pred.contiguous()-label)**2
    return loss.mean()

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = np.array(img1, dtype=np.float)  #img1.astype(np.float64)
    img2 = np.array(img2, dtype=np.float) # img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    print("img2 shape ", img2.shape)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def evaluate_camera_alignment(opt,pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1) #TODO:shape
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error



def prealign_cameras(self,opt,pose,pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3,device=opt.device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    return pose_aligned,sim3