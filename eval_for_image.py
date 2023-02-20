"""

psnr : https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/

https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
"""
import sys
import os
import argparse
from pathlib import Path
import lpips

import torch
import numpy as np
from tqdm import tqdm
import imageio
import PIL

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.pose_utils import create_spiral_poses
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.lie_group_helper import convert3x4_4x4
from models.nerf_models import OfficialNerf
from tasks.any_folder.train import model_render_image
from models.intrinsics import LearnFocal
from models.poses import LearnPose

from models.base import *
import torchvision
import torchvision.transforms.functional as torchvision_F
from external.pohsun_ssim import pytorch_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    # parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')
    parser.add_argument('--test_img_num', type=int, default=-1, help='num of images to train')


    return parser.parse_args()



def main(args):
    img_path = Path(os.path.join(args.base_dir, 'output'))
    gt_path =  Path(os.path.join(args.base_dir, 'gt'))
    ours_path = Path(os.path.join(args.base_dir, 'ours_output'))

    n = args.test_img_num
    test_index = np.array([1,55,110,165,220,274,329,348,441,495,550,607,661,716,770,825,879,935,989,1044])

    """"
        python eval_for_image.py --test_img_num=20 --base_dir=for_eval/h_5
    """

    test_img = []
    gt_img = []
    '''load redner image'''
    # imgs = (imgs.numpy() * 255).astype(np.uint8)
    for i in range(n):
        img_name = '{}/{}'.format(img_path, str(int(i)).zfill(3) + '.png')
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)))
        test_img.append(imageio.imread(img_name))
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)).float())

        gt_img_name = '{}/{}'.format(gt_path, '1'+str(test_index[i]).zfill(5) + '.png')
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)))
        gt_img.append(imageio.imread(gt_img_name))
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)).float())

    # DSNeRF TEST GT evaluate
    res = []
    import lpips
    lpips_loss = lpips.LPIPS(net="alex")
    # print('img size ', torch.from_numpy(imgs[0]).float().size())
    h,w,_ = torch.from_numpy(test_img[0]).float().size() #h,w,3

    for i in range(n):
        img = np.array(test_img[i]) / 255.0
        img = torch.from_numpy(img).float().view(-1, h, w, 3).permute(0, 3, 1, 2)# [B,3,H,W]
        gt_img = np.array(gt_img[i]) / 255.0
        gt_img = torch.from_numpy(gt_img).float().view(-1, h, w, 3).permute(0, 3, 1, 2) # [B,3,H,W]

        psnr = -10 * MSE_loss(img, gt_img).log10().item()  # mabye rendering,true image
        ssim = pytorch_ssim.ssim(img, gt_img).item()
        lpips = lpips_loss(img* 2 - 1, gt_img * 2 - 1).item()
        res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
        # res.append(edict(psnr=psnr, ssim=ssim))
        break

    print("---------DSNeRF------------")
    print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
    print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
    print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
    print("--------------------------")

    gt_img = []
    ours_img = []
    '''load redner image'''
    # imgs = (imgs.numpy() * 255).astype(np.uint8)
    for i in range(n):
        gt_img_name = '{}/{}'.format(gt_path, '1' + str(test_index[i]).zfill(5) + '.png')
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)))
        gt_img.append(imageio.imread(gt_img_name))
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)).float())

        gt_img_name = '{}/{}'.format(ours_path, 'rgb_' + str(i) + '.png')
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)))
        ours_img.append(imageio.imread(gt_img_name))
        # test_gt_img.append(torch.from_numpy(imageio.imread(image_name)).float())

    # Ours TEST GT evaluate
    res = []
    import lpips
    lpips_loss = lpips.LPIPS(net="alex")
    # print('img size ', torch.from_numpy(imgs[0]).float().size())
    h,w,_ = torch.from_numpy(ours_img[0]).float().size() #h,w,3
    # print('test img size ', test_gt_img[0].float().size())

    for i in range(n):
        img = np.array(ours_img[i]) / 255.0
        img = torch.from_numpy(img).float().view(-1, h, w, 3).permute(0, 3, 1, 2)# [B,3,H,W]
        gt_img = np.array(gt_img[i]) / 255.0
        gt_img = torch.from_numpy(gt_img).float().view(-1, h, w, 3).permute(0, 3, 1, 2) # [B,3,H,W]

        psnr = -10 * MSE_loss(img, gt_img).log10().item()  # mabye rendering,true image
        ssim = pytorch_ssim.ssim(img, gt_img).item()
        lpips = lpips_loss(img* 2 - 1, gt_img * 2 - 1).item()
        res.append(edict(psnr=psnr, ssim=ssim, lpips=lpips))
        # res.append(edict(psnr=psnr, ssim=ssim))
        break

    print("--------Ours------------")
    print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
    print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
    print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
    print("--------------------------")






    return


if __name__ == '__main__':
    args = parse_args()
    with torch.no_grad():
        main(args)


