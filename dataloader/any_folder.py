import os

import torch
import numpy as np
from tqdm import tqdm
import imageio
import PIL
from PIL import Image

from dataloader.with_colmap import resize_imgs
from utils.lie_group_helper import convert3x4_4x4
from scipy.spatial.transform import Rotation

# any_folder for strayscanner, load identity pose
def load_imgs(image_dir, num_img_to_load, start, end, skip, load_sorted, load_img):
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
            # img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
            img = PIL.Image.fromarray(imageio.imread(p))
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

def compose_pair(pose_a,pose_b): #pose_new,pose
    # pose_new(x) = pose_b o pose_a(x)
    R_a,t_a = pose_a[...,:3],pose_a[...,3:]
    R_b,t_b = pose_b[...,:3],pose_b[...,3:]
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)#[...,0]
    pose_new = torch.empty(3, 4)
    pose_new[...,:3] = R_new
    pose_new[...,3:] = t_new
    return pose_new
def compose(pose_list):
    # compose a sequence of poses together
    # pose_new(x) = poseN o ... o pose2 o pose1(x)
    pose_new = pose_list[0]   # in novel view, [pose_shift,pose_rot,pose_shift2]
    for pose in pose_list[1:]:
        pose_new = compose_pair(pose_new,pose)
    return pose_new

def read_pose_csv(pose_path):
    odometry = np.loadtxt(pose_path, delimiter=',')  # , skiprows=1
    poses = []
    for line in odometry:  # timestamp, frame(float ex 1.0), x, y, z, qx, qy, qz, qw
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        T_WC = torch.tensor(T_WC).float()

        pose_flip_R = torch.diag(torch.tensor([1,-1,-1]))  #[right,down,forward] -> [right,up,backward]
        # pose_flip
        pose_flip_t = torch.zeros(pose_flip_R.shape[:-1])
        pose_flip_R = pose_flip_R.float()
        pose_flip_t = pose_flip_t.float()
        pose_flip = torch.cat([pose_flip_R, pose_flip_t[..., None]], dim=-1)  # [...,3,4]

        pose = compose([pose_flip, T_WC[:3]])
        poses.append(pose)
    poses = torch.stack([p for p in poses], dim=0)
    return poses

def read_meta(in_dir):
    #test pose
    test_file = os.path.join(in_dir, 'odometry_test.csv')
    test_poses = read_pose_csv(test_file)
    test_poses = np.array(test_poses, dtype=float)  # (N_images, 3, 4)
    test_poses = convert3x4_4x4(test_poses) #(N,4,4)

    #train pose
    train_file = os.path.join(in_dir, 'odometry_train.csv')
    train_poses = read_pose_csv(train_file)
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
    def __init__(self, base_dir, scene_name,res_ratio, num_img_to_load, start, end, skip, load_sorted, load_img=True):
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
        self.imgs_dir = os.path.join(self.base_dir, self.scene_name, 'rgb_train')

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
    base_dir = './data/strayscanner'
    scene_name = 'meeting_room_5'
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
