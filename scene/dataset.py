from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov,getProjectionMatrix,getWorld2View2
from PIL import Image
from torchvision import transforms as T
import os
from typing import NamedTuple
from glob import glob
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
                
    def get_cameras_from_timeindex(self,timeindex):
        cam_infos = self.dataset.get_frames_from_timeindex(timeindex)
        cameras = []
        for cam_info in cam_infos:
            orig_w, orig_h = cam_info.image.size
            if self.args.resolution > 0.0:
                resolution = round(orig_w/(self.args.resolution)), round(orig_h/(self.args.resolution))
            else:
                resolution = cam_info.image.size
            resized_image_rgb = PILtoTorch(cam_info.image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            loaded_mask = None
            if resized_image_rgb.shape[1] == 4:
                loaded_mask = resized_image_rgb[3:4, ...]    
            cameras.append(
                Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image=gt_image, gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name, uid=cam_info.uid, data_device=torch.device("cuda"), 
                timestamp=cam_info.timestamp
                )
            )
        return cameras
    
    def __getitem__(self, index):
        assert False, "Not impled"
        # cam_info = self.dataset[index]
        # orig_w, orig_h = cam_info.image.size
        # resolution = round(orig_w/(self.args.resolution)), round(orig_h/(self.args.resolution))

        # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        # gt_image = resized_image_rgb[:3, ...]

        # loaded_mask = None
        # if resized_image_rgb.shape[1] == 4:
        #     loaded_mask = resized_image_rgb[3:4, ...]    
            
        # return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
        #         FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
        #         image=gt_image, gt_alpha_mask=loaded_mask,
        #         image_name=cam_info.image_name, uid=cam_info.id, data_device=torch.device("cuda"),
        #         timestamp=cam_info.timestamp)
     
    def __len__(self):
        return len(self.dataset)

class N3V_Dataset(Dataset):#follow scene.neural_3D-dataset_NDC.Neural3D_NDC_Dataset
    def __init__(
            self,
            path, #colmap_0/images
            split="train"
    ):

        images_paths = []
        self.camera_params = []
        
        image_dir = os.path.join(path,'images')
        image_per_cam_paths = sorted(glob(os.path.join(image_dir,'cam*')))

        self.duration = None
        originnumpy = os.path.join(os.path.dirname(image_dir), "poses_bounds.npy")
        with open(originnumpy, 'rb') as numpy_file:
            poses_bounds = np.load(numpy_file)
            poses = poses_bounds[:,:15].reshape(-1,3,5)
            hwfs = poses[:,:,4]

            poses = np.concatenate([poses[:,:,1:2],poses[:,:,0:1],-poses[:,:,2:3],poses[:,:,3:4]],2)
            c2w_mats = np.concatenate([poses,np.tile(np.array([[[0,0,0,1.]]]),[poses.shape[0],1,1])],1)
            w2c_mats = np.linalg.inv(c2w_mats)
            Rotations = w2c_mats[:,:3,:3].transpose([0,2,1])
            Translations = w2c_mats[:,:3,3]
        
        assert len(image_per_cam_paths) == len(poses), "poses and the number of images/cam* does not match"
        
        for uid,(hwf,R,T,image_per_cam_path) in enumerate(zip(hwfs,Rotations,Translations,image_per_cam_paths)):
            height,width,focal_length_x = hwf
            height = int(height)
            width = int(width)

            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            
            self.camera_params.append([uid,R,T,FovY,FovX,width,height])

            images_path_per_cam = glob(os.path.join(image_per_cam_path,"*"))
            if self.duration is None:
                self.duration = len(images_path_per_cam)
            else:
                assert len(images_path_per_cam) == self.duration, f"number of images does not match:{image_per_cam_path}"
            images_paths.append(images_path_per_cam)
        
        if split == "train":
            images_paths = images_paths[1:]
            self.camera_params = self.camera_params[1:]
        elif split == "test":
            images_paths = images_paths[:1]
            self.camera_params = self.camera_params[:1]
        else:
            assert False, "split only train or test"
        self.images_paths = [list(x) for x in zip(*images_paths)]
        assert len(self.images_paths) == self.duration , "duration does not match"
        assert len(self.images_paths[0]) == len(self.camera_params)

    def __len__(self):
        return len(self.images_paths)
    def get_frames_from_timeindex(self,timeindex):
        
        cam_infos = []
        for image_path, camera_params in zip(self.images_paths[timeindex],self.camera_params):
            uid,Rotation,Translation,FovY,FovX,width,height = camera_params
            image = Image.open(image_path)
            cam_infos.append(
                CameraInfo(uid=uid, R=Rotation, T=Translation, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                image_name=os.path.basename(image_path).split(".")[0], width=width, height=height,
                                timestamp=timeindex)
                                )

        
        return cam_infos
    def __getitem__(self, index):
        assert False , "Not impled"
        return cam_info
    

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float
import json
from pathlib import Path
class Blender_Dataset(Dataset):#follow scene.neural_3D-dataset_NDC.Neural3D_NDC_Dataset
    def __init__(
            self,
            path, #colmap_0/images
            split="train",
            white_background=False,
            extension='.png'
    ):
        if split=='train':
            transformsfile = 'transforms_train.json'
        elif split=='test':
            transformsfile = 'transforms_test.json'
        else:
            assert False , "split only train and test"
        images_paths = []
        self.camera_params = []
        self.white_background = white_background
        
        image_dir = os.path.join(path,'images')
        image_per_cam_paths = sorted(glob(os.path.join(image_dir,'cam*')))

        self.duration = None

        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]
            height = contents["height"]
            width = contents["width"]

            frames = contents["frames"]
            for idx, frame in enumerate(frames): 
                cam_name = frame["file_path"].split('/')[-2]
                images_path_per_cam = sorted(glob(os.path.join(image_dir,cam_name,"*")))
                if self.duration is None:
                    self.duration = len(images_path_per_cam)
                else:
                    assert len(images_path_per_cam) == self.duration, f"number of images does not match:{images_path_per_cam}"
                images_paths.append(images_path_per_cam)

                matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
                R = -np.transpose(matrix[:3,:3])
                R[:,0] = -R[:,0]
                T = -matrix[:3, 3]

                fovy = focal2fov(fov2focal(fovx, height), width)
                FovY = fovy 
                FovX = fovx

                self.camera_params.append([idx,R,T,FovY,FovX,width,height])
                
        self.images_paths = [list(x) for x in zip(*images_paths)]
        assert len(self.images_paths) == self.duration , "duration does not match"
        assert len(self.images_paths[0]) == len(self.camera_params)

    def __len__(self):
        return len(self.images_paths)
    def get_frames_from_timeindex(self,timeindex):
        cam_infos = []
        for image_path, camera_params in zip(self.images_paths[timeindex],self.camera_params):
            uid,Rotation,Translation,FovY,FovX,width,height = camera_params
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            cam_infos.append(
                CameraInfo(uid=uid, R=Rotation, T=Translation, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                image_name=os.path.basename(image_path).split(".")[0], width=width, height=height,
                                timestamp=timeindex)
                                )

        
        return cam_infos
    def __getitem__(self, index):
        assert False , "Not impled"
        return cam_info



class COLMAP_Dataset(Dataset):#follow scene.neural_3D-dataset_NDC.Neural3D_NDC_Dataset
    def __init__(
            self,
            path, #colmap_0/images
            split="train"
    ):

        images_paths = []
        self.camera_params = []
        
        frame_paths = sorted(glob(os.path.join(path,'frame*')))

        
        frame_path = frame_paths[0]
        try:
            cameras_extrinsic_file = os.path.join(frame_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(frame_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(frame_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(frame_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        sorted_extrinsics = sorted(cam_extrinsics.items(), key=lambda x: x[1].name)
        if split == "train":
            sorted_extrinsics = sorted_extrinsics[1:]
        elif split == "test":
            sorted_extrinsics = sorted_extrinsics[:1]
        else:
            assert False, "split only train or test"

        for idx, (key,extr) in enumerate(sorted_extrinsics):
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = idx # intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            self.camera_params.append([uid,R,T,FovY,FovX,width,height])

        self.images_paths = []
        for frame_path in frame_paths:
            image_path = sorted(glob(os.path.join(frame_path,'images_2','*')))    
            if split == "train":
                self.images_paths.append(image_path[1:])    
            elif split == "test":
                self.images_paths.append(image_path[:1])
            else:
                assert False, "split only train or test"
        

        assert len(self.images_paths[0]) == len(self.camera_params)

    def __len__(self):
        return len(self.images_paths)
    def get_frames_from_timeindex(self,timeindex):
        cam_infos = []
        for image_path, camera_params in zip(self.images_paths[timeindex],self.camera_params):
            uid,Rotation,Translation,FovY,FovX,width,height = camera_params
            image = Image.open(image_path)
            cam_infos.append(
                CameraInfo(uid=uid, R=Rotation, T=Translation, FovY=FovY, FovX=FovX, image=image, image_path=image_path, 
                                image_name=os.path.basename(image_path).split(".")[0], width=width, height=height,
                                timestamp=timeindex)
                                )
        return cam_infos
    def __getitem__(self, index):
        assert False , "Not impled"
        return cam_info