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
import PIL
import imageio
import camera

"""
nerf.py의 novel_pose plot 위한 코드 
"""
# python pose_visualization.py --datadir ./data/arkit/llff_main_computers --logsdir ./logs/arkit/llff_main_computers/llff_main_computers_02

# python pose_visualization.py --datadir ./data/any_folder_demo/llff_main_computers --logsdir ./logs/any_folder/llff_main_computers/llff_main_computers_02
# python pose_visualization.py --logsdir ./logs/nerfmm/trex/lr_0.001_gpu0_seed_17_resize_4_Nsam_128_Ntr_img_-1_freq_10__220416_1604 --ref_pose=False
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default='./data/arkit/llff_main_computers02',
                        help='data directory')
    parser.add_argument("--logsdir", type=str, default='./logs/arkit/llff_main_computers02/llff_main_computers02_03',
                        help='logs name')
    parser.add_argument('--ref_pose', type=bool, default=True, help='false : only 2d pose plot')
    return parser


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
    vertices = camera.cam2world(vertices[None],pose)
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


def plot_save_2d_poses(fig,pose,pose_ref=None,path=None,ep=None):
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=0.5)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=0.5)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    setup_3D_plot(ax1,elev=-90,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))  #lim=edict(x=(-1,1),y=(-1,1),z=(-1,1))
    setup_3D_plot(ax2,elev=0,azim=-90,lim=edict(x=(-1,1),y=(-1,1),z=(-1,1)))  #lim=edict(x=(-1,1),y=(-1,1),z=(-1,1))
    ax1.set_title("forward-facing view",pad=0)
    ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=40)
    png_fname = "{}/2d_{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()



# for novel_view test
def plot_save_3d_poses(fig,pose,pose_ref=None,path=None,ep=None): # pose = novel_view, pose_ref= rectangle_pose
    # get the camera meshes
    _,_,cam = get_camera_mesh(pose,depth=0.5)
    cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=0.5)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title(" {}".format(ep),pad=0)
    setup_3D_plot(ax,elev=10,azim=50,lim=edict(x=(-0.5,0.5),y=(-0.5,0.5),z=(-0.5,0.5))) #lim=edict(x=(-1,1),y=(-1,1),z=(-0.5,0.3)) lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4))
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

    if ep == 0 :
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
    png_fname = "{}/3d_{}.png".format(path,ep)
    plt.savefig(png_fname,dpi=75)
    # clean up
    plt.clf()

def load_pose(pose_file):
    pose = []
    with open(pose_file, "r") as f:  # frame.txt 읽어서
        pose_lines = f.readlines()
    for line in pose_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        pose_raw = np.reshape(line_data_list[2:], (3, 4))  #timestamp와 image number 함께 있으므로 건너뛰기 위해
        pose.append(pose_raw)
    pose = np.array(pose, dtype=float)
    pose = torch.from_numpy(pose).float()
    return pose



def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])  # [right, forward, up]
    pose = camera.pose.invert(pose)  #아마 c2w->w2c?
    return pose

def generate_videos_pose(args):
    """
    [right,up,back] --> [right, forward, up]
    """
    #일단 초반 N개만
    N = 20

    #GT pose load
    args.ref_pose = True
    print('##### ',args.ref_pose)
    gt_pose = None
    if args.ref_pose:
        gt_pose_file = os.path.join(args.datadir, "transforms_train.txt")
        gt_pose = load_pose(gt_pose_file)[:N]
        gt_pose = torch.stack([parse_raw_camera(p) for p in gt_pose], dim=0) #[right,up,back] --> [right, forward, up]

    img_path = os.path.join(args.logsdir, 'pose_history_image_'+str(N))
    os.makedirs(img_path, exist_ok=True)
    pose_3d_imgs = []
    pose_2d_imgs = []

    pose_history_milestone = list(range(0, 100, 5)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
    for epoch_i in pose_history_milestone:
        #refine pose load
        refine_pose = np.load(os.path.join(args.logsdir,'pose_history', str(epoch_i).zfill(6) + '.npy'))[:N,:3,:]
        refine_pose = torch.from_numpy(refine_pose).float()
        refine_pose = torch.stack([parse_raw_camera(p) for p in refine_pose], dim=0)  # [right,up,back] --> [right, forward, up]

        if args.ref_pose:
            fig = plt.figure(figsize=(10,10))
            plot_save_3d_poses(fig,refine_pose,pose_ref=gt_pose,path=img_path,ep=epoch_i)
            plt.close()
            image_fname_3d = "{}/3d_{}.png".format(img_path, epoch_i)
            image_3d = PIL.Image.fromarray(imageio.imread(image_fname_3d))
            pose_3d_imgs.append(image_3d)

        fig = plt.figure(figsize=(16,18))
        plot_save_2d_poses(fig,refine_pose,pose_ref=gt_pose,path=img_path,ep=epoch_i)
        plt.close()
        image_fname_2d = "{}/2d_{}.png".format(img_path, epoch_i)
        image_2d = PIL.Image.fromarray(imageio.imread(image_fname_2d))
        pose_2d_imgs.append(image_2d)

    if args.ref_pose:
        imageio.mimwrite(os.path.join(img_path, 'pose_3d.gif'), pose_3d_imgs, fps=60)
    imageio.mimwrite(os.path.join(img_path, 'pose_2d.gif'), pose_2d_imgs, fps=60)


# python pose_visualization.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    generate_videos_pose(args)

