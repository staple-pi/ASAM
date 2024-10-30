from typing import List, Optional, Tuple
import sys
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, sam_model_registry_o
from statistics import mean
import torch.nn.init as init
from torch.nn.functional import threshold, normalize
import os
import tempfile
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
#from skimage import transform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean
join = os.path.join
import torch.distributed as dist
from val_utils import  ASAM, SA1BDataset, KINSDataset_v,KINSDataset_0,COCOADataset_0, COCOADataset_v,val
from pycocotools.coco import COCO
import json
def main(args):
    device = 'cuda'
    model_type = "vit_l"
    asam_model = sam_model_registry[model_type](checkpoint=args.asam_checkpoint)
    asam = ASAM(model=asam_model).to(device=device)
    annotations_path = args.annotation_path
    kins = COCO(annotations_path)
    with open(annotations_path,'r') as f:
        data = json.load(f)
    annotations_list = data['annotations']  
    annotations_list = annotations_list[:args.data_num]
    val_dataset = KINSDataset_v(annotations_list, args.img_dir, kins)
    print("Number of training samples: ", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        )
    
    mean_iou, mean_iou_post,  mean_oiou_list,  mean_oiou_post_list = val(asam_model=asam,dataloader=val_dataloader,device=device)
    print(f'mean_iou {mean_iou}, mean_iou_post {mean_iou_post}, mean_oiou_list {mean_oiou_list}, mean_oiou_post_list {mean_oiou_post_list}')
    print('DONE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asam_checkpoint', type=str, default= 'E:/code/asam-kins-1.pth')
    parser.add_argument('--img_dir',type=str,default="E:/Code/KINS/testing/image_2")
    parser.add_argument('--annotation_path',type=str,default='E:/Code/KINS/annotations/update_test_2020.json')
    parser.add_argument('--data_num',type=int,default = 10000)
    parser.add_argument('--batch_size',type=int,default = 1)
    opt = parser.parse_args()

    main(opt)