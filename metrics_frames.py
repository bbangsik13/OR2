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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from glob import glob

def evaluate(args):
    init_path = args.init_path
    model_path = args.model_path
    source_path = args.source_path
    
    pred_paths = [os.path.join(init_path,'test','ours_15000','renders','00000.png')]
    pred_paths.extend([frame_path for frame_path in sorted(glob(os.path.join(model_path,'test','rendering2','*')))])
    gt_paths = [os.path.join(frame_path,'images_2','cam00.png') for frame_path in sorted(glob(os.path.join(source_path,'frame*')))]

    full_dict = {}
    per_view_dict = {}
    print("")


    psnrs = []
    ssims = []
    image_names = []
    render_stacks = []
    for i, (pred_path, gt_path) in tqdm(enumerate(zip(pred_paths,gt_paths)), desc="Metric evaluation progress"):
        render = Image.open(pred_path)
        gt = Image.open(gt_path)
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

        ssims.append(ssim(render, gt))
        psnrs.append(psnr(render, gt))
        image_names.append(f'frame{i:06}')
        render_stacks.append(render.squeeze().cpu())
    
    render_stacks = torch.stack(render_stacks).cpu()
    difference = render_stacks[1:, ...] - render_stacks[:-1, ...] 
    mask = tf.to_tensor(Image.open(args.mask_path).convert("L")).squeeze() > 0.5
    tv = torch.abs(difference[...,mask]).mean(-1).mean(-1)

    print("Scene: ", model_path,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("Scene: ", model_path,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("Scene: ", model_path,  "TV: {:>12.7f}".format(tv.mean(), ".5"))
    print("")
    print("Scene: ", model_path,  "SSIM std: {:>12.7f}".format(torch.tensor(ssims).std(), ".5"))
    print("Scene: ", model_path,  "PSNR std: {:>12.7f}".format(torch.tensor(psnrs).std(), ".5"))
    print("Scene: ", model_path,  "TV std: {:>12.7f}".format(tv.std(), ".5"))
    print("")

    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "TV": tv.mean().item(),
                                            })
    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                "TV": {name: tv for tv, name in zip(tv.tolist(), image_names[1:])}})
                                                
    with open(model_path + "/results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(model_path + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--init_path', '-i', required=True)
    parser.add_argument('--model_path', '-m', required=True)
    parser.add_argument('--source_path', '-s', required=True)
    parser.add_argument('--mask_path', '-c', required=True)
    args = parser.parse_args()
    evaluate(args)


