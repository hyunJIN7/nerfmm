import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import imageio

from utils.comp_ray_dir import * #comp_ray_dir_cam_fxfy
from utils.pose_utils import center_poses
from utils.lie_group_helper import convert3x4_4x4


def resize_imgs(imgs, new_h, new_w):
    """
    :param imgs:    (N, H, W, 3)            torch.float32 RGB
    :param new_h:   int/torch int
    :param new_w:   int/torch int
    :return:        (N, new_H, new_W, 3)    torch.float32 RGB
    """
    imgs = imgs.permute(0, 3, 1, 2)  # (N, 3, H, W)
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')  # (N, 3, new_H, new_W)
    imgs = imgs.permute(0, 2, 3, 1)  # (N, new_H, new_W, 3)

    return imgs  # (N, new_H, new_W, 3) torch.float32 RGB


def load_imgs(image_dir, img_ids, new_h, new_w):
    print("## arkit image load : ", image_dir)  #TODO: image folder name check
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_names = img_names[img_ids]  # image name for this split

    img_paths = [os.path.join(image_dir, n) for n in img_names]

    img_list = []
    for p in tqdm(img_paths):
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    img_list = resize_imgs(img_list, new_h, new_w)
    return img_list, img_names


def load_split(scene_dir, img_dir, data_type, num_img_to_load, skip, c2ws, H, W, load_img):
    # load pre-splitted train/val ids
    img_ids = np.loadtxt(os.path.join(scene_dir, data_type + '_ids.txt'), dtype=np.int32, ndmin=1)
    if num_img_to_load == -1:
        img_ids = img_ids[::skip]
        print('Loading all available {0:6d} images'.format(len(img_ids)))
    elif num_img_to_load > len(img_ids):
        print('Required {0:4d} images but only {1:4d} images available. '
              'Exit'.format(num_img_to_load, len(img_ids)))
        exit()
    else:
        img_ids = img_ids[:num_img_to_load:skip]

    N_imgs = img_ids.shape[0]

    # use img_ids to select camera poses
    c2ws = c2ws[img_ids]  # (N, 3, 4)

    # load images
    if load_img:
        imgs, img_names = load_imgs(img_dir, img_ids, H, W)  # (N, H, W, 3) torch.float32
    else:
        imgs, img_names = None, None

    result = {
        'c2ws': c2ws,  # (N, 3, 4) np.float32
        'imgs': imgs,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'img_ids': img_ids,  # (N, ) np.int
    }
    return result


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """

    # H,W,Focal
    intrin_file = os.path.join(in_dir, 'Frames.txt')
    assert os.path.isfile(intrin_file), "camera info:{} not found".format(intrin_file)
    with open(intrin_file, "r") as f:  # frame.txt
        cam_intrinsic_lines = f.readlines()
    cam_intrinsics = []
    line_data_list = cam_intrinsic_lines[0].split(',')  # intrinsic 첫번째 값으로 고정
    cam_intrinsics.append([float(i) for i in line_data_list])
    intr = torch.tensor([
        [cam_intrinsics[0][2], 0, cam_intrinsics[0][4]],
        [0, cam_intrinsics[0][3], cam_intrinsics[0][5]],
        [0, 0, 1]
    ]).float()
    # origin video's origin_size(1920,1440) -> extract frame (640,480)
    ori_size = (1920, 1440)
    size = (640, 480)
    W, H = [640,480]
    intr[0, :] /= (ori_size[0] / size[0])
    intr[1, :] /= (ori_size[1] / size[1])  # resize 전 크기가 orgin_size 이기 때문에
    # focal = torch.tensor([intr[0,0],intr[1,1]]).float()
    focal =  intr[0,0]
    focal_x = intr[0,0]
    focal_y = intr[1, 1]

    #pose
    pose_file =  os.path.join(in_dir, 'transforms_train.txt')
    assert os.path.isfile(pose_file), "pose info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:  # frame.txt 읽어서
        cam_frame_lines = f.readlines()
    c2ws = []  # r1x y z tx r2x y z ty r3x y z tz
    for line in cam_frame_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        pose_raw = np.reshape(line_data_list[2:], (3, 4))
        c2ws.append(pose_raw)
    c2ws = np.array(c2ws, dtype=float)  # (N_images, 3, 4)
    # pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws
    # _ , pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws #TODO: test

    # bounds ???
    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # near_original = bounds.min()
        near_original = 1
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        # bounds /= scale_factor
        c2ws[..., 3] /= scale_factor

    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        # 'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal ,  #[focal_x,focal_y],     # scalar
        'focal_x' : focal_x,
        'focal_y': focal_y,
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


class DataLoaderARKit:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, data_type, res_ratio, num_img_to_load, skip, use_ndc, load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param data_type:   'train' or 'val'.
        :param res_ratio:   int [1, 2, 4] etc to resize images to a lower resolution.
        :param num_img_to_load/skip: control frame loading in temporal domain.
        :param use_ndc      True/False, just centre the poses and scale them.
        :param load_img:    True/False. If set to false: only count number of images, get H and W,
                            but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.data_type = data_type
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.skip = skip
        self.use_ndc = use_ndc
        self.load_img = load_img

        self.imgs_dir = os.path.join(self.base_dir, self.scene_name, 'train') #iphone_train_val_images

        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        self.img_dir = os.path.join(self.scene_dir, 'images')

        # all meta info
        meta = read_meta(self.scene_dir, self.use_ndc)
        self.c2ws = meta['c2ws']  # (N, 4, 4) all camera pose
        self.H = meta['H']
        self.W = meta['W']
        self.focal = float(meta['focal']) #torch.tensor(meta['focal']).float() #
        self.focal_x = float(meta['focal_x'])
        self.focal_y = float(meta['focal_y'])
        self.total_N_imgs = self.c2ws.shape[0]

        if self.res_ratio > 1:
            self.H = self.H // self.res_ratio
            self.W = self.W // self.res_ratio
            self.focal /= self.res_ratio
            self.focal_x /= self.res_ratio
            self.focal_y /= self.res_ratio

        self.near = 0.0
        self.far = 1.0

        '''Load train/val split'''
        split_results = load_split(self.scene_dir, self.img_dir, self.data_type, self.num_img_to_load,
                                   self.skip, self.c2ws, self.H, self.W, self.load_img)
        self.c2ws = split_results['c2ws']  # (N, 4, 4) np.float32
        self.imgs = split_results['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = split_results['img_names']  # (N, )
        self.N_imgs = split_results['N_imgs']
        self.img_ids = split_results['img_ids']  # (N, ) np.int

        # generate cam ray dir.
        self.ray_dir_cam = comp_ray_dir_cam(self.H, self.W, self.focal)  # (H, W, 3) torch.float32
        # self.ray_dir_cam = comp_ray_dir_cam_fxfy(self.H, self.W, self.focal[0], self.focal[1]) # (H, W, 3) torch.float32

        # convert np to torch.
        self.c2ws = torch.from_numpy(self.c2ws).float()  # (N, 4, 4) torch.float32
        self.ray_dir_cam = self.ray_dir_cam.float()  # (H, W, 3) torch.float32


if __name__ == '__main__':
    scene_name = 'LLFF/fern'
    use_ndc = True
    scene = DataLoaderARKit(base_dir='/your/data/path',
                                 scene_name=scene_name,
                                 data_type='train',
                                 res_ratio=8,
                                 num_img_to_load=-1,
                                 skip=1,
                                 use_ndc=use_ndc)
