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

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    
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

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def box_to_mask(box,mask_shape):
    # 计算box的宽度和高度
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    # 计算扩展的大小
    expand_width = int(box_width * 0.4)
    expand_height = int(box_height * 0.4)
    # 更新box的坐标，并确保不超过mask的边界
    new_x1 = int(max(0, box[0] - expand_width))
    new_y1 = int(max(0, box[1] - expand_height))
    new_x2 = int(min(mask_shape[1], box[2] + expand_width))
    new_y2 = int(min(mask_shape[0], box[3] + expand_height))
    # 创建与mask相同形状的数组
    output_array = np.zeros(mask_shape, dtype=int)
    # 填充扩展后的box内部区域的值为1
    output_array[new_y1:new_y2, new_x1:new_x2] = 1
    return output_array

#sam_checkpoint = "E:/code/fine_mask_unet.pth"

def main(args):
    asam_checkpoint = args.asam_checkpoint
    annotations_path = args.annotations_path
    image_dir = args.img_dir
    #sam_checkpoint = None
    model_type = "vit_l"
    device = "cuda"

    ckpt = torch.load(asam_checkpoint,map_location='cuda:0')
    sam = sam_model_registry[model_type](checkpoint= None)
    sam.load_state_dict(ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    kins = COCO(annotations_path)
    with open(annotations_path,'r') as f:
        data = json.load(f)
    annotations_list =  data['annotations']
    '''
    if args.moiou:
        i=0
        while i < len(annotations_list):
            a_polys = annotations_list[i]['a_segm']
            i_polys = annotations_list[i]['i_segm']
            if a_polys == i_polys:
                del annotations_list[i]
            else:
                i += 1
    '''            
    lenth_of_imglist = len(annotations_list)
    num_range = args.num_range
    if num_range > lenth_of_imglist:
        num_range = lenth_of_imglist
    print(num_range)
    batch_no = 0
    total_num = 0
    totalocc_num=0 
    total_ious = 0.
    total_occlu_ious = 0.
    while batch_no < num_range:
        bbox_coords = {}
        points = {}
        labels = {}
        occlusion_mask = {}
        ground_truth_masks = {}
        visibel_mask={}
        maskinput = {}
        ious = 0.
        occl_ious = 0.
        instance_num = 0
        num_occ = 0 
        img_size={}
        image_path = {}
        no_occ={}
        print(batch_no)
        next_num = batch_no+50
        if next_num  > lenth_of_imglist:
            next_num = lenth_of_imglist
        for i in range(batch_no, next_num):
            annotation = annotations_list[i]
            image_id = annotations_list[i]['image_id']
            image_info = kins.loadImgs([image_id])[0]
            image_name = image_info['file_name']
            height, width = image_info['height'], image_info['width']
            image_path[i] = os.path.join(image_dir,image_name)
            origin_size = (height,width)
            x, y, w, h = annotation['a_bbox']
            box = np.array([x, y, x + w, y + h])
            bbox_coords[i] = np.array([x, y, x + w, y + h])
            maskin = box_to_mask(box,origin_size)
            a_polys = annotation['a_segm']
            amask = polys_to_mask(a_polys,height,width)
            i_polys = annotation['i_segm']
            imask = polys_to_mask(i_polys,height,width)
            if args.minus_v:
                maskin = maskin - imask     ##################################################################################
            if a_polys==i_polys:
                maskin = np.zeros_like(amask)
                no_occ[i] == True
            else:
                no_occ[i] == False
            points_1 = np.argwhere(imask == 1)  
            if len(points_1) == 0:
                vmask = amask
                points_1 = np.argwhere(vmask == 1)    
                print('error raise')
                print(image_name)
            #if len(points_1) >= 2:
            #    point_coords = points_1[np.random.choice(points_1.shape[0], 2, replace=False)]
            #    selected_points = point_coords[:, ::-1]
            #    point_labels = np.array([1, 1])
            #if len(points_1) == 1:
            point_coords = points_1[np.random.choice(points_1.shape[0], 1, replace=False)]
            selected_points = point_coords[:, ::-1]
            point_labels = np.array([1])

            points[i] = selected_points
            labels[i] = point_labels    
            omask = np.bitwise_xor(amask,imask)
            visibel_mask[i]  =imask     
            ground_truth_masks[i] = amask
            occlusion_mask[i] = omask
            maskinput[i] = maskin
            img_size[i] = [height,width]    
        transformed_data = defaultdict(dict)

        # 将图像转换为SAM的格式
        for k in bbox_coords.keys():
            image = cv2.imread(image_path[k])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            transformed_data[k]['image'] = image
            
        keys = list(bbox_coords.keys())

        for k in keys:
            input_image = transformed_data[k]['image']
            predictor.set_image(input_image)
            prompt_box = bbox_coords[k]
            point_in = points[k]
            label_in = labels[k]
            gt_binary_mask = ground_truth_masks[k]
            #mask_input = np.ones_like(gt_binary_mask)
            mask_input = maskinput[k]
            #mask_input = occlusion_mask[i]
            mask_input = mask_preprocess(mask_input)

            masks_pred, _, _ = predictor.predict(
                point_coords=point_in, ####
                point_labels=label_in, ###
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
            if no_occ[k] == False:   #表示没有遮挡
                num_occ+=1
                totalocc_num+=1
                occlusion_pred = torch.tensor(np.bitwise_xor(masks_pred[0], visibel_mask[k]),dtype=torch.bool)
                occlusion_gt = torch.tensor(np.bitwise_and(gt_binary_mask, occlusion_mask[k]),dtype=torch.bool)
                occlusion_iou = calculate_iou(occlusion_pred, occlusion_gt)
                occl_ious = occl_ious + occlusion_iou

        mIoU = (ious / instance_num).float()
        occlu_mIoU = (occl_ious / num_occ).float()
        print('miou'+ str(mIoU))
        print('occlu_miou'+ str(occlu_mIoU))

        total_ious = total_ious + ious
        total_occlu_ious = total_occlu_ious + occl_ious
        batch_no = batch_no + 50

    mIoU = (total_ious / total_num).float()
    occlu_mIoU = (total_occlu_ious / totalocc_num).float()
    print('miou'+ str(mIoU))
    print('occlu_miou'+ str(occlu_mIoU))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asam_checkpoint', type=str, default= "E:/code/asam-2w-0.pth")   # train_o weight 的地址
    parser.add_argument('--img_dir',type=str,default='E:/code/KINS/testing/image_2')      # KINS-test的地址
    parser.add_argument('--annotations_path',type=str,default='E:/code/KINS/annotations/update_test_2020.json')
    parser.add_argument('--num_range',type=int,default=5000)
    parser.add_argument('--minus_v',type=str2bool,default=True)
    opt = parser.parse_args()
    print("minus_v:", opt.minus_v)
    main(opt)