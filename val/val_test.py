import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import sys
import numpy as np
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
from pycocotools import mask as mask_utils
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import random
import numpy as np
from statistics import mean
import torch.nn.init as init
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F 
import argparse

def read_json(file):
    # 打开 JSON 文件
    with open(file, "r") as f:
        data = json.load(f)
    return data['annotations'][0]

def get_mask_preprocess_shape(oldh, oldw, long_side_length):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def mask_resieze_pad(mask,target_size):
    x,y = mask.shape
    resized_mask = mask.reshape(1,1,x,y)
    input_torch = torch.as_tensor(resized_mask, dtype=torch.float)
    output_size = target_size
    downsampled_tensor = torch.nn.functional.interpolate(input_torch, size=output_size, mode='bilinear', align_corners=False)
    h, w = downsampled_tensor.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    downsampled_tensor = F.pad(downsampled_tensor, (0, padw, 0, padh))
    aarray = downsampled_tensor.numpy()
    return aarray

def mask_preprocess(mask):
    target_size = get_mask_preprocess_shape(mask.shape[0],mask.shape[1],1024)
    output = mask_resieze_pad(mask,target_size)
    return output[0,:,:,:]


def calculate_iou(pred, gt):
    # 计算预测值和ground truth之间的交并比（IoU）
    intersection = torch.logical_and(pred, gt).sum().float()
    
    union = torch.logical_or(pred, gt).sum().float()
    iou = intersection / union
    return iou

def calculate_AP(pred, gt):
    true_positives = torch.logical_and(pred, gt).sum().float()
    true_in_pred_false_in_ge = pred & ~gt
    false_positives = torch.sum(true_in_pred_false_in_ge).float()
    total_positives = gt.sum().float()
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_positives
    if precision > 0 or recall > 0:
        ap += (recall - 0) * precision

    return ap

def calculate_average_precision(preds, gts, iou_threshold=0.5):
    num_instances = len(preds)
    ap = 0.0

    for i in range(num_instances):
        pred = preds[i]
        gt = gts[i]
        iou = calculate_iou(pred, gt)

        # 如果IoU大于阈值，则该预测被视为正确，否则为错误
        is_correct = iou >= iou_threshold

        # 计算精度和召回率
        true_positives = is_correct.sum().float()
        false_positives = (~is_correct).sum().float()
        total_positives = gt.sum().float()

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_positives

        # 使用梯形法则计算AP
        if precision > 0 or recall > 0:
            ap += (recall - 0) * precision

    # 将AP除以实例数量得到平均精度
    ap /= num_instances
    return ap


#sam_checkpoint = "E:/code/fine_mask_unet.pth"

def main(args):
    asam_checkpoint = args.asam_checkpoint
    occlude_dir = args.img_dir
    #sam_checkpoint = None
    model_type = "vit_l"
    device = "cuda"

    ckpt = torch.load(asam_checkpoint,map_location='cuda:0')
    sam = sam_model_registry[model_type](checkpoint= None)
    sam.load_state_dict(ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    img_list = os.listdir(occlude_dir)
    img_list = [img for img in img_list if img.endswith(".jpg")]
    img_list.sort(key=lambda x: int(x[3:-4]))

    lenth_of_imglist = len(img_list)
    num_range = lenth_of_imglist
    print(num_range)
    batch_no = 0
    total_num = 0
    total_ious = 0.
    total_occlu_ious = 0.
    while batch_no < 5000:
        bbox_coords = {}
        occlusion_mask = {}
        ground_truth_masks = {}
        visibel_mask={}
        ious = 0.
        occl_ious = 0.
        instance_num = 0
        print(batch_no)
        for i in range(batch_no, batch_no+50):
            image_name = img_list[i]
            image_path = os.path.join(occlude_dir, image_name)
            json_name = image_name[:-3] + 'json'

            annotation = read_json(os.path.join(occlude_dir,json_name))
            x, y, w, h = annotation['bbox']
            bbox_coords[image_name] = np.array([x, y, x + w, y + h])
            mask = mask_utils.decode(annotation['segmentation'])
            ground_truth_masks[image_name] = mask
            if 'occluder_mask'  in annotation.keys():
                omask =  mask_utils.decode(annotation['occluder_mask'])
                occlusion_mask[image_name] = omask
            else:
                occlusion_mask[image_name] = np.zeros_like(mask)
            vmask = mask & (~omask)
            visibel_mask[image_name] = vmask
        transformed_data = defaultdict(dict)

        # 将图像转换为SAM的格式
        for k in bbox_coords.keys():
            image = cv2.imread(os.path.join(occlude_dir,k))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            transformed_data[k]['image'] = image
            
        keys = list(bbox_coords.keys())

        for k in keys:
            input_image = transformed_data[k]['image']
            predictor.set_image(input_image)
            prompt_box = bbox_coords[k]
            gt_binary_mask = ground_truth_masks[k]
            mask_input = visibel_mask[k]
            #mask_input = occlusion_mask[k]
            mask_input = mask_preprocess(mask_input)  

            masks_pred, _, _ = predictor.predict(
                box=prompt_box,
                mask_input=mask_input,
                multimask_output=False,
            )
            pred = torch.tensor(masks_pred[0], dtype=torch.bool)
            gt = torch.tensor(gt_binary_mask, dtype=torch.bool)
            iou = calculate_iou(pred, gt)
            ious = ious +iou
            instance_num = instance_num + 1
            total_num = total_num + 1
            occlusion_pred = torch.tensor(np.bitwise_xor(masks_pred[0], visibel_mask[k]),dtype=torch.bool)
            occlusion_gt = torch.tensor(np.bitwise_xor(gt_binary_mask, visibel_mask[k]),dtype=torch.bool)
            #occlusion_gt = torch.tensor(np.bitwise_and(gt_binary_mask, occlusion_mask[k]),dtype=torch.bool)
            occlusion_iou = calculate_iou(occlusion_pred, occlusion_gt)
            occl_ious = occl_ious + occlusion_iou

        mIoU = (ious / instance_num).float()
        occlu_mIoU = (occl_ious / instance_num).float()
        print('miou'+ str(mIoU))
        print('occlu_miou'+ str(occlu_mIoU))

        total_ious = total_ious + ious
        total_occlu_ious = total_occlu_ious + occl_ious
        batch_no = batch_no + 50

    mIoU = (total_ious / total_num).float()
    occlu_mIoU = (total_occlu_ious / total_num).float()
    print('miou'+ str(mIoU))
    print('occlu_miou'+ str(occlu_mIoU))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asam_checkpoint', type=str, default= 'E:/code/asam-kins-1.pth')   # asam weight 的地址
    parser.add_argument('--img_dir',type=str,default="E:/Code/KINS/testing/image_2")  # SA1B-test的地址
    opt = parser.parse_args()
