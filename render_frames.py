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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import commentjson as ctjs
import tinycudann as tcnn
from plyfile import PlyData, PlyElement
from ntc import NeuralTransformationCache
import numpy as np

def load_ply(path,max_sh_degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        _xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        _features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        _opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        return _xyz,_features_dc,_features_rest,_opacity,_scaling,_rotation
        
def add_gaussians(gaussians,add_gaussians_path):
    _added_xyz,_added_features_dc,_added_features_rest,_added_opacity,_added_scaling,_added_rotation = load_ply(add_gaussians_path,gaussians.max_sh_degree)

    gaussians._xyz = torch.nn.Parameter(torch.cat((gaussians._xyz, _added_xyz.detach().clone().to(gaussians._xyz.device)), dim=0).requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(torch.cat((gaussians._features_dc, _added_features_dc.detach().clone().to(gaussians._features_dc.device)), dim=0).requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(torch.cat((gaussians._features_rest, _added_features_rest.detach().clone().to(gaussians._features_rest.device)), dim=0).requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(torch.cat((gaussians._opacity, _added_opacity.detach().clone().to(gaussians._opacity.device)), dim=0).requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(torch.cat((gaussians._scaling, _added_scaling.detach().clone().to(gaussians._scaling.device)), dim=0).requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(torch.cat((gaussians._rotation, _added_rotation.detach().clone().to(gaussians._rotation.device)), dim=0).requires_grad_(True))

def deform_gaussians(gaussians,path):
    gaussians.ntc.load_state_dict(torch.load(path))
    gaussians._xyz_bound_min = gaussians.ntc.xyz_bound_min
    gaussians._xyz_bound_max = gaussians.ntc.xyz_bound_max
    gaussians.query_ntc()
    gaussians.update_by_ntc()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, ntc_conf_path:str, output_path:str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,online=True)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        with open(ntc_conf_path) as ntc_conf_file:
            ntc_conf = ctjs.load(ntc_conf_file)
        model=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
        gaussians.ntc=NeuralTransformationCache(model,gaussians.get_xyz_bound()[0],gaussians.get_xyz_bound()[1])

        for frame in range(len(scene.train_dataset)):
            scene.updateTimeindex(frame)
            scene.updateCameras()
            if frame > 0:
                deform_gaussians(gaussians,os.path.join(output_path,'ntc',f'{frame:05}.pth'))
                add_gaussians(gaussians,os.path.join(output_path,'add',f'{frame:05}.ply'))
            if not skip_train:
                render_set(output_path, "train", frame, scene.getTrainCameras(), gaussians, pipeline, background)
            if not skip_test:
                render_set(output_path, "test", frame, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ntc_conf_path", type=str, default="configs/cache/cache_F_4.json")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.ntc_conf_path,args.output_path)