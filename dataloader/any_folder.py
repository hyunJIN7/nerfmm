import os

import torch
import numpy as np
from tqdm import tqdm
import imageio

from dataloader.with_colmap import resize_imgs
from utils.lie_group_helper import convert3x4_4x4


def load_imgs(image_dir, num_img_to_load, start, end, skip, load_sorted, load_img):
    print("## ", image_dir)
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names

    # down sample frames in temporal domain
    if end == -1:
        img_names = img_names[start::skip]
    else:
        img_names = img_names[start:end:skip]

    if not load_sorted:
        np.random.shuffle(img_names)

    # load images after down sampled
    if num_img_to_load > len(img_names):
        print('Asked for {0:6d} images but only {1:6d} available. Exit.'.format(num_img_to_load, len(img_names)))
        exit()
    elif num_img_to_load == -1:
        print('Loading all available {0:6d} images'.format(len(img_names)))
    else:
        print('Loading {0:6d} images out of {1:6d} images.'.format(num_img_to_load, len(img_names)))
        img_names = img_names[:num_img_to_load]

    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    if load_img:
        for p in tqdm(img_paths):
            img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
            img_list.append(img)
        img_list = np.stack(img_list)  # (N, H, W, 3)
        img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
        H, W = img_list.shape[1], img_list.shape[2]
    else:
        tmp_img = imageio.imread(img_paths[0])  # load one image to get H, W
        H, W = tmp_img.shape[0], tmp_img.shape[1]

    results = {
        'imgs': img_list,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    return results

def read_file(pose_file): # timestamp filename r11 r12 r13 tx r21 r22 r23 y r31 r32 r33 z
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
    return c2ws

def read_meta(in_dir):
    #test pose
    test_file = os.path.join(in_dir, 'transforms_test.txt')
    test_poses = read_file(test_file)
    test_poses = np.array(test_poses, dtype=float)  # (N_images, 3, 4)
    test_poses = convert3x4_4x4(test_poses) #(N,4,4)

    #train pose
    train_file = os.path.join(in_dir, 'transforms_test.txt')
    train_poses = read_file(train_file)
    train_poses = np.array(train_poses, dtype=float)  # (N_images, 3, 4)
    train_poses = convert3x4_4x4(train_poses)  # (N,4,4)

    results = {
        'test_poses': test_poses,
        'train_poses': train_poses,
    }

    return results

class DataLoaderAnyFolder:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, res_ratio, num_img_to_load, start, end, skip, load_sorted, load_img=True):
        """
        :param base_dir:
        :param scene_name:
        :param res_ratio:       int [1, 2, 4] etc to resize images to a lower resolution.
        :param start/end/skip:  control frame loading in temporal domain.
        :param load_sorted:     True/False.
        :param load_img:        True/False. If set to false: only count number of images, get H and W,
                                but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.res_ratio = res_ratio
        self.num_img_to_load = num_img_to_load
        self.start = start
        self.end = end
        self.skip = skip
        self.load_sorted = load_sorted
        self.load_img = load_img

        self.imgs_dir = os.path.join(self.base_dir, self.scene_name, 'train') #TODO : image fname

        image_data = load_imgs(self.imgs_dir, self.num_img_to_load, self.start, self.end, self.skip,
                                self.load_sorted, self.load_img)
        self.imgs = image_data['imgs']  # (N, H, W, 3) torch.float32
        self.img_names = image_data['img_names']  # (N, )
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']

        # always use ndc
        self.near = 0.0
        self.far = 1.0

        if self.res_ratio > 1:
            self.H = self.ori_H // self.res_ratio
            self.W = self.ori_W // self.res_ratio
        else:
            self.H = self.ori_H
            self.W = self.ori_W

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W)  # (N, H, W, 3) torch.float32

        # test pose, train pose load
        """Fort test_view & novel_view --> load train_pose for novel_view and test_pose"""
        self.scene_dir = os.path.join(self.base_dir, self.scene_name)
        meta = read_meta(self.scene_dir)
        self.test_poses = meta['test_poses']  #(N,4,4)
        self.train_poses = meta['train_poses']

        # convert np to torch.
        self.test_poses = torch.from_numpy(self.test_poses).float()  # (N, 4, 4) torch.float32
        self.train_poses = torch.from_numpy(self.train_poses).float()  # (N, 4, 4) torch.float32


if __name__ == '__main__':
    base_dir = '/your/data/path'
    scene_name = 'LLFF/fern/images'
    resize_ratio = 8
    num_img_to_load = -1
    start = 0
    end = -1
    skip = 1
    load_sorted = True
    load_img = True

    scene = DataLoaderAnyFolder(base_dir=base_dir,
                                scene_name=scene_name,
                                res_ratio=resize_ratio,
                                num_img_to_load=num_img_to_load,
                                start=start,
                                end=end,
                                skip=skip,
                                load_sorted=load_sorted,
                                load_img=load_img)
