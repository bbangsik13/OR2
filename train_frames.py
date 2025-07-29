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
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, quaternion_loss, d_xyz_gt, d_rot_gt
from gaussian_renderer import render, network_gui
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.debug_utils import save_tensor_img
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import re
import numpy as np
from matplotlib import pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

def training_one_frame(dataset, opt, pipe, load_iteration, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,opt.rotate_sh)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False,online=True)
    scene.updateTimeindex(1)
    for frame in range(1,len(scene.train_dataset)):
        start_time=time.time()
        last_s1_res = []
        last_s2_res = []
        first_iter = 0
        scene.updateCameras()
        
        gaussians.training_one_frame_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"Training progress frame{frame:06}")
        first_iter += 1
        s1_start_time=time.time()
        # Train the NTC
        for iteration in range(first_iter, opt.iterations + 1):        
            iter_start.record()

            gaussians.update_learning_rate_ntc(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
                        
            # Query the NTC
            
            loss = torch.tensor(0.).cuda()
            
            
            # A simple 
            for batch_iteraion in range(opt.batch_size):
            
                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                
                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True
                gaussians.query_ntc()
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                image = torch.clamp(image, min=0.,max=1.)
                offset = gaussians.get_offset(viewpoint_cam.uid)
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image = torch.clamp(gt_image+offset, min=0.,max=1.)

                Ll1 = l1_loss(image, gt_image)
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                
            loss = loss + opt.lambda_offset * torch.abs(offset).mean()
            loss/=opt.batch_size
            loss.backward()
            iter_end.record()
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",'Num Gaussians': f"{gaussians._xyz.shape[0]}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                s1_res = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if s1_res is not None:
                    last_s1_res.append(s1_res)

                # Tracking Densification Stats
                if iteration > opt.densify_from_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.ntc_optimizer.step()
                    gaussians.ntc_optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")

        s1_end_time=time.time()
        # Densify
        if(opt.iterations_s2>0):
        # Dump the NTC
            scene.dump_NTC(frame)
        # Update Gaussians by NTC
            gaussians.update_by_ntc()
        # Prune, Clone and setting up  
            gaussians.training_one_frame_s2_setup(opt)
            progress_bar = tqdm(range(opt.iterations, opt.iterations + opt.iterations_s2), desc="Training progress of Stage 2")    
        
        # Train the new Gaussians
        for iteration in range(opt.iterations + 1, opt.iterations + opt.iterations_s2 + 1):  
            if gaussians._added_xyz.shape[0] == 0:
                with torch.no_grad():
                    s2_res = training_report(tb_writer, testing_iterations[-1], Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                    if s2_res is not None:
                        last_s2_res.append(s2_res)
                    progress_bar.close()
                break      
            iter_start.record()
                        
            # Update Learning Rate
            gaussians.update_learning_rate_new(iteration - opt.iterations)
            
            loss = torch.tensor(0.).cuda()
            
            for batch_iteraion in range(opt.batch_size):
            
                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                
                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True
                render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                # Loss
                image = torch.clamp(image, min=0.,max=1.)
                offset = gaussians.get_offset(viewpoint_cam.uid)
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image = torch.clamp(gt_image+offset, min=0.,max=1.)

                Ll1 = l1_loss(image, gt_image)
                loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                
            loss = loss + opt.lambda_offset * torch.abs(offset).mean()
            loss/=opt.batch_size
            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if (iteration - opt.iterations) % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",'New Gaussians': f"{gaussians._added_xyz.shape[0]}"})
                    progress_bar.update(10)
                if iteration == opt.iterations + opt.iterations_s2:
                    progress_bar.close()

                # Log and save
                s2_res = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if s2_res is not None:
                    last_s2_res.append(s2_res)
                        
                # Densification
                if (iteration - opt.iterations) % opt.densification_interval == 0:
                    gaussians.adding_and_prune(opt,scene.cameras_extent)
                                
                # Optimizer step
                if iteration < opt.iterations + opt.iterations_s2:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
            
        s2_end_time=time.time()
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save_add(iteration,frame, save_type='added')
        
        pre_time = s1_start_time - start_time
        s1_time = s1_end_time - s1_start_time
        s2_time = s2_end_time - s1_end_time

        gaussians._xyz = torch.nn.Parameter(torch.cat((gaussians._xyz, gaussians._added_xyz.detach().clone().to(gaussians._xyz.device)), dim=0).requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(torch.cat((gaussians._features_dc, gaussians._added_features_dc.detach().clone().to(gaussians._features_dc.device)), dim=0).requires_grad_(True))
        gaussians._features_rest = torch.nn.Parameter(torch.cat((gaussians._features_rest, gaussians._added_features_rest.detach().clone().to(gaussians._features_rest.device)), dim=0).requires_grad_(True))
        gaussians._opacity = torch.nn.Parameter(torch.cat((gaussians._opacity, gaussians._added_opacity.detach().clone().to(gaussians._opacity.device)), dim=0).requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(torch.cat((gaussians._scaling, gaussians._added_scaling.detach().clone().to(gaussians._scaling.device)), dim=0).requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(torch.cat((gaussians._rotation, gaussians._added_rotation.detach().clone().to(gaussians._rotation.device)), dim=0).requires_grad_(True))
        gaussians._added_xyz = None
        gaussians._added_features_dc = None
        gaussians._added_features_rest = None
        gaussians._added_opacity = None
        gaussians._added_scaling = None
        gaussians._added_rotation = None
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        print(f"Preparation: {pre_time}")
        if pre_time > 2:
            print(f"If preparation is time-consuming, consider down-scaling the images BEFORE running 3DGStream.")
        print(f"Stage 1: {s1_time}")
        print(f"Stage 2: {s2_time}")
        scene.updateTimeindex()
        save_tensor_img(s1_res['last_test_image'],os.path.join(args.output_path,'test/rendering1',f'{frame:05}'))
        save_tensor_img(s2_res['last_test_image'],os.path.join(args.output_path,'test/rendering2',f'{frame:05}'))

def prepare_output_and_logger(args):    
    if not args.output_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    last_test_psnr=0.0
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'test':
                    last_test_psnr = psnr_test
                    last_test_image = image
                    last_gt = gt_image

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        return {'last_test_psnr':last_test_psnr.cpu().numpy()
                , 'last_test_image':last_test_image.cpu()
                , 'last_points_num':scene.gaussians.get_xyz.shape[0]
                }

def train_frames(lp, op, pp, args):
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.source_path = args.video_path
    training_one_frame(lp.extract(args), op.extract(args), pp.extract(args), args.first_load_iteration, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--load_iteration', type=int, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.output_path == "":
        args.output_path=args.model_path
    if args.read_config and args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if key not in ["output_path", "source_path", "model_path", "video_path", "debug_from","first_load_iteration","ntc_path","ntc_lr","sh_degree","offset_lr_init","offset_lr_final","use_offset","images","white_background"]:
                setattr(args, key, value)
    serializable_namespace = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    json_namespace = json.dumps(serializable_namespace)
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args.json"), 'w') as f:
        f.write(json_namespace)
    train_frames(lp,op,pp,args)