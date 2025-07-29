import json
import os
import numpy as np
from glob import glob
NUM_RUN=10

from argparse import ArgumentParser
parser = ArgumentParser("find best 3dgs init")
parser.add_argument("--init_path", default='3dgs_init', type=str)
args = parser.parse_args()

input_dir = args.init_path

paths = sorted(glob(os.path.join(input_dir, '*')))
paths = [path for path in paths]
N = len(paths) // NUM_RUN
reshaped_paths = np.reshape(paths, (N, NUM_RUN))

total_metric = {
    'psnr_mean': [],
    'psnr_std': [],
    'ssim_mean': [],
    'ssim_std': [],
    'psnr_max_mean': [],
    'ssim_at_max_psnr_mean': [],
    'best': []
}

for scene_path in reshaped_paths:
    scene_metric = {
        'psnr': [],
        'ssim': [],
    }
    max_psnr = -np.inf
    max_psnr_metrics = {'name': '', 'psnr': 0, 'ssim': 0}

    for input_path in scene_path:
        with open(os.path.join(input_path, 'results.json'), 'r') as f:
            ff = json.load(f)['ours_15000']
        name = input_path.split('/')[-1]
        psnr_value = ff['PSNR']
        ssim_value = ff['SSIM']

        # Append each metric to scene_metric
        scene_metric['psnr'].append(psnr_value)
        scene_metric['ssim'].append(ssim_value)

        # Check if this is the highest PSNR so far in this scene
        if psnr_value > max_psnr:
            max_psnr = psnr_value
            max_psnr_metrics = {
                'name': name,
                'psnr': psnr_value,
                'ssim': ssim_value,
            }
    total_metric['best'].append(max_psnr_metrics['name'])
    # Calculate and store mean and std for each metric in total_metric
    total_metric['psnr_mean'].append(np.mean(scene_metric['psnr']))
    total_metric['psnr_std'].append(np.std(scene_metric['psnr']))
    total_metric['ssim_mean'].append(np.mean(scene_metric['ssim']))
    total_metric['ssim_std'].append(np.std(scene_metric['ssim']))

    # Store the max PSNR values and corresponding metrics
    total_metric['psnr_max_mean'].append(max_psnr_metrics['psnr'])
    total_metric['ssim_at_max_psnr_mean'].append(max_psnr_metrics['ssim'])


print(f"method: {input_dir.split('/')[-1]} "
    f"psnr:{np.mean(total_metric['psnr_mean']):.2f} ({np.mean(total_metric['psnr_std']):.3f}), "
    f"ssim:{np.mean(total_metric['ssim_mean']):.3f} ({np.mean(total_metric['ssim_std']):.3f}), "
    f"psnr_max:{np.mean(total_metric['psnr_max_mean']):.2f}, "
    f"ssim_at_max_psnr:{np.mean(total_metric['ssim_at_max_psnr_mean']):.3f}, "
    )
print(total_metric['best'])
os.makedirs(os.path.join(input_dir+'_best'),exist_ok=True)
for item in total_metric['best']:
    source_path = os.path.join(input_dir,item)
    target_path = os.path.join(input_dir+'_best',"_".join(item.split("_")[:-1]))
    
    if os.path.exists(target_path):
        if os.path.islink(target_path):  
            os.unlink(target_path)
        elif os.path.isdir(target_path): 
            os.system(f'rm -rf {target_path}')
        else: 
            os.remove(target_path)
    
    cmd = f"ln -r -s {source_path} {target_path}"
    result = os.system(cmd)
    
    if result != 0:
        print(f"Failed to create symlink: {target_path} -> {source_path}")

