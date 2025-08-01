import sys
import os
project_directory = '.'
sys.path.append(os.path.abspath(project_directory))

import tinycudann as tcnn
import commentjson as ctjs
import torch
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch.nn.functional as F
from ntc import NeuralTransformationCache
from glob import glob

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def fetchXYZ(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    return torch.tensor(xyz, dtype=torch.float, device="cuda")

def get_xyz_bound(xyz, percentile=80):
    ## Hard-code the coordinate of the corners here!!
    return torch.tensor([-20, -15,   5]).cuda(), torch.tensor([15, 10, 23]).cuda()
    # return torch.tensor([-1.3, -1.3,  -1.3]).cuda(), torch.tensor([1.3, 1.3, 1.3]).cuda() # synthetic

def get_contracted_xyz(xyz):
    xyz_bound_min, xyz_bound_max = get_xyz_bound(xyz, 80)
    normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
    return normalzied_xyz

def quaternion_multiply(a, b):
    a_norm=nn.functional.normalize(a)
    b_norm=nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_loss(q1, q2):
    cos_theta = F.cosine_similarity(q1, q2, dim=1)
    cos_theta = torch.clamp(cos_theta, -1+1e-7, 1-1e-7)
    return 1-torch.pow(cos_theta, 2).mean()

def l1loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

from argparse import ArgumentParser
parser = ArgumentParser("NTC warmup")
parser.add_argument("--init_path", default='3dgs_init_best', type=str)
args = parser.parse_args()

folder_path=args.init_path
input_paths = glob(os.path.join(folder_path,'*'))
for input_path in input_paths:
    SCENE = input_path.split('/')[-1]

    postfixs=['F_4']
    ntc_conf_paths=['configs/cache/cache_'+postfix+'.json' for postfix in postfixs]
    pcd_path=f'{input_path}/point_cloud/iteration_15000/point_cloud.ply'
    save_paths=[f'ntc/{SCENE}_ntc_params_'+postfix+'.pth' for postfix in postfixs]


    ntcs=[]
    for ntc_conf_path in ntc_conf_paths:    
        with open(ntc_conf_path) as ntc_conf_file:
            ntc_conf = ctjs.load(ntc_conf_file)
        ntc=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
        ntc_optimizer = torch.optim.Adam(ntc.parameters(), lr=1e-4)
        xyz=fetchXYZ(pcd_path)
        normalzied_xyz=get_contracted_xyz(xyz)
        mask = (normalzied_xyz >= 0) & (normalzied_xyz <= 1)
        mask = mask.all(dim=1)
        ntc_inputs=torch.cat([normalzied_xyz[mask]],dim=-1)
        noisy_inputs = ntc_inputs + 0.01 * torch.rand_like(ntc_inputs)
        d_xyz_gt=torch.tensor([0.,0.,0.]).cuda()
        d_rot_gt=torch.tensor([1.,0.,0.,0.]).cuda()
        dummy_gt=torch.tensor([1.]).cuda()
        def cacheloss(resi):
            masked_d_xyz=resi[:,:3]
            masked_d_rot=resi[:,3:7]
            masked_dummy=resi[:,7:8]
            loss_xyz=l1loss(masked_d_xyz,d_xyz_gt)
            loss_rot=quaternion_loss(masked_d_rot,d_rot_gt)
            loss_dummy=l1loss(masked_dummy,dummy_gt)
            loss=loss_xyz+loss_rot+loss_dummy
            return loss
        for iteration in range(0,3000):      
            ntc_inputs_w_noisy = torch.cat([noisy_inputs, ntc_inputs, torch.rand_like(ntc_inputs)],dim=0)  
            ntc_output=ntc(ntc_inputs_w_noisy)
            loss=cacheloss(ntc_output)
            if iteration % 100 ==0:
                print(loss)
            loss.backward()
            ntc_optimizer.step()
            ntc_optimizer.zero_grad(set_to_none = True)
        ntcs.append(ntc)



    for idx, save_path in enumerate(save_paths):
        ntc=NeuralTransformationCache(ntcs[idx],get_xyz_bound(xyz)[0],get_xyz_bound(xyz)[1])
        torch.save(ntc.state_dict(),save_path)