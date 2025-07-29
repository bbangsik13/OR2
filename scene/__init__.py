#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0],online=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        try:
            self.output_path = args.output_path
        except:
            self.output_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if online:
            if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                scene_info = sceneLoadTypeCallbacks["BlenderOnline"](args.source_path, args.white_background)
            else:
                scene_info = sceneLoadTypeCallbacks["ColmapOnline"](args.source_path)
            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            self.train_dataset = FourDGSdataset(scene_info.train_cameras, args)
            self.test_dataset = FourDGSdataset(scene_info.test_cameras, args)
            self.timeindex = 0
            os.makedirs(os.path.join(self.output_path,'ntc'),exist_ok=True)
            os.makedirs(os.path.join(self.output_path,'add'),exist_ok=True)
            os.makedirs(os.path.join(self.output_path,'test/rendering1'),exist_ok=True)
            os.makedirs(os.path.join(self.output_path,'test/rendering2'),exist_ok=True)
        else:
            self.train_cameras = []
            self.test_cameras = []
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"

            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.output_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.output_path, "cameras.json"), 'w') as file:
                    json.dump(json_cams, file)

            # if shuffle:
            #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            #     random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            if args.extent == 0:
                self.cameras_extent = scene_info.nerf_normalization["radius"]
            else:
                self.cameras_extent = args.extent
            print("Loading Training Cameras")
            self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
            print("Loading Test Cameras")
            self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),#'added',
                                                        "point_cloud.ply"), self.cameras_extent)
            self.gaussians.load_offset(os.path.join(self.model_path,
                                                        "offset",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "offset.npy"))

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.set_offset(self.getTrainCameras())

    def save(self, iteration, save_type='all'):
        point_cloud_path = os.path.join(self.output_path, "point_cloud/iteration_{}".format(iteration))
        if save_type=='added' or save_type=='origin':
            self.gaussians.save_ply(os.path.join(point_cloud_path, save_type, "point_cloud.ply"), save_type)
        elif save_type=='all':
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), save_type)
        else:
            raise NotImplementedError('Not Implemented!')

    def save_offset(self,iteration):
        offset_path = os.path.join(self.output_path, "offset/iteration_{}".format(iteration))
        self.gaussians.save_offset(offset_path)

    def save_add(self, iteration,frame, save_type='all'):
        point_cloud_path = os.path.join(self.output_path,'add',str(frame).zfill(5)+".ply")
        self.gaussians.save_ply(point_cloud_path, save_type)


    def dump_NTC(self,frame):
        NTC_path = os.path.join(self.output_path,'ntc',f"{frame:05}.pth")
        self.gaussians.ntc.dump(NTC_path)

    def getTrainCameras(self):
        return self.train_cameras
    def updateCameras(self):
        self.updateTrainCameras()
        self.updateTestCameras()
    def updateTrainCameras(self):
        self.train_cameras = self.train_dataset.get_cameras_from_timeindex(self.timeindex)
    def getTestCameras(self):
        return self.test_cameras
    def updateTestCameras(self):
        self.test_cameras = self.test_dataset.get_cameras_from_timeindex(self.timeindex)
    def updateTimeindex(self,index=None):
        if index:
            self.timeindex = index
        else:
            self.timeindex += 1
