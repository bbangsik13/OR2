meta = {# bbox_x,bbox_y,bbox_width,bbox_height,image_width,image_height
    'cut_roasted_beef':[
        [0,1348,596,502,2704,2028],
        [473,199,1510,1264,2704,2028],
        [1025,848,913,1138,2704,2028],
        [2608,99,96,1512,2704,2028],
        [564,1379,1701,649,2704,2028]
        ],
    'sear_steak':[
        [423,199,1581,1290,2704,2028],
        [1054,832,809,1120,2704,2028],
        [54,1379,568,471,2704,2028],
        [2579,73,125,1528,2704,2028],
        [1983,1714,118,209,2704,2028]
    ],
    'cook_spinach':[
        [452,207,1528,1253,2704,2028],
        [0,1400,706,497,2704,2028],
        [991,866,848,1089,2704,2028],
        [1996,1730,97,147,2704,2028],
        [2587,76,117,1518,2704,2028],
        [2370,1902,199,126,2704,2028]
    ],
    'coffee_martini_wo_cam13':[
        [645,714,1123,1314,2704,2028],
        [434,1355,173,270,2704,2028],
        [1718,1403,510,500,2704,2028]
    ],
    'flame_steak':[
        [1004,814,958,1172,2704,2028],
        [41,1340,581,497,2704,2028],
        [2600,63,104,1510,2704,2028],
        [457,209,1515,1259,2704,2028],
        [2556,1732,131,272,2704,2028],
        [2011,1745,76,170,2704,2028]
    ],
    'flame_salmon_frag1':[
        [345,720,1437,1261,2704,2028]
    ]
}
import numpy as np
import os
import cv2
from argparse import ArgumentParser

parser = ArgumentParser("Mask generator")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--mask_path", "-m", required=True, type=str)
args = parser.parse_args()

os.makedirs(os.path.join(args.mask_path,'n3v'),exist_ok=True)
for scene, bboxs in meta.items():
    os.makedirs(os.path.join(args.mask_path,'n3v',scene),exist_ok=True)
    W,H = bboxs[0][-2:]
    mask = 255 * np.ones((H,W)).astype(np.uint8)
    for bbox in bboxs:
        bbox_x,bbox_y,bbox_width,bbox_height = bbox[:-2]
        mask[bbox_y:bbox_y+bbox_height,bbox_x:bbox_x+bbox_width] = 0

    cv2.imwrite(os.path.join(args.mask_path,'n3v',scene,'cam00.png'),mask)

    img_undist_cmd = ("colmap" + " image_undistorter \
        --image_path " + os.path.join(args.mask_path,'n3v',scene)  + " \
        --input_path " + os.path.join(args.source_path,'n3v',scene) + "/distorted/sparse/0 \
        --output_path " + os.path.join(args.mask_path,'n3v',scene) + " \
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        print(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    os.makedirs(os.path.join(args.mask_path,'n3v',scene,'images_2'),exist_ok=True)
    exit_code = os.system(f"convert -resize 50% {os.path.join(args.mask_path,'n3v',scene,'images','cam00.png')} {os.path.join(args.mask_path,'n3v',scene,'images_2','cam00.png')}")
    exit_code = os.system(f"convert -resize 50% {os.path.join(args.mask_path,'n3v',scene,'cam00.png')} {os.path.join(args.mask_path,'n3v',scene,'cam00_2.png')}")
