import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import lpips

"""
nerf.py의 novel_pose plot 위한 코드 
"""
# python visualization_novel_view.py --expname cube
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default='./data/any_folder_demo/llff_main_computers',
                        help='data directory')
    parser.add_argument("--logsdir", type=str, default='./logs/any_folder/llff_main_computers/llff_main_computers_01',
                        help='logs name')
    return parser

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def cam2world(X,pose): #x 가 center ?..
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]   # in novel view, [pose_shift,pose_rot,pose_shift2]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new



def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = cam2world(vertices[None],pose)
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

# for novel_view test
def plot_save_novel_poses(fig,pose,pose_ref=None,path=None,ep=None): # pose = novel_view, pose_ref= rectangle_pose
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=0.5)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=0.5)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title(" {}".format(ep),pad=0)
    setup_3D_plot(ax,elev=10,azim=50,lim=edict(x=(-3.5,1),y=(-3.5,1),z=(-3,1))) #lim=edict(x=(-1,1),y=(-1,1),z=(-0.5,0.3)) lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))

    for i in range(len(cam_ref)):
        ax.plot(cam_ref[i, :, 0], cam_ref[i, :, 1], cam_ref[i, :, 2], color=ref_color, linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)

    png_fname = "{}/{}_GT.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
    for i in range(len(cam_ref)):
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    for i in range(N):
        ax.plot(cam[i,5,0],
                cam[i,5,1],
                cam[i,5,2],color=(1,0,0),linewidth=3)
    for i in range(len(cam_ref)):
        ax.plot(cam_ref[i,5,0],
                cam_ref[i,5,1],
                cam_ref[i,5,2],color=(1,0,0),linewidth=3)
    png_fname = "{}/{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def load_pose(pose_file, isGT = False):
    """
     if isGT is ture :
        ARKit의 GT pose는 timestamp와 image number가 함께 있으므로 건너뛰기 위해

    """
    pose = []
    with open(pose_file, "r") as f:  # frame.txt 읽어서
        pose_lines = f.readlines()
    for line in pose_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        if isGT :
            pose_raw = np.reshape(line_data_list[2:], (3, 4))
        else :
            pose_raw = np.reshape(line_data_list, (3, 4))
        pose.append(pose_raw)
    pose = np.array(pose, dtype=float)
    return pose

def generate_videos_pose(args):# novel pose, raw pose

    # pose, pose_ref
    #(3,4)
    gt_pose_file = os.path.join(args.datadir, "transforms_train.txt")
    refine_pose_file = os.path.join(args.logsdir, "transforms_train.txt")
    gt_pose = load_pose(gt_pose_file,True)
    refine_pose = load_pose(refine_pose_file,False)
    #근데  포즈 프레임을 바꿔줘야해


    fig = plt.figure(figsize=(10,10))
    cam_path = "novel_poses"
    os.makedirs(cam_path,exist_ok=True)
    plot_save_novel_poses(fig,refine_pose,pose_ref=gt_pose,path=cam_path,ep=args.expname)
    plt.close()

# python visualization_novel_view.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    generate_videos_pose(args)
