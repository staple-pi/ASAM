import torch
import sys
import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
#from skimage import transform
import numpy as np
import os
from pycocotools import mask as mask_utils
import json
from statistics import mean
join = os.path.join
import torch.nn.init as init
import torch.distributed as dist

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def cleanup():
    dist.destroy_process_group()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def get_mask_preprocess_shape(oldh, oldw, long_side_length):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def mask_resieze_pad(mask, target_size, target_long, return_torch=False):
    x, y, = mask.shape
    resized_mask = mask.reshape(1, 1, x, y)
    input_torch = torch.as_tensor(resized_mask, dtype=torch.float)
    output_size = target_size
    downsampled_tensor = torch.nn.functional.interpolate(
        input_torch, size=output_size, mode='bilinear', align_corners=False)
    h, w = downsampled_tensor.shape[-2:]
    padh = target_long - h
    padw = target_long - w
    downsampled_tensor = F.pad(downsampled_tensor, (0, padw, 0, padh))
    if return_torch:
        return downsampled_tensor
    else:
        aarray = downsampled_tensor.numpy()
        return aarray


def mask_preprocess(mask, target_long=1024, return_torch=False):
    target_size = get_mask_preprocess_shape(
        mask.shape[0], mask.shape[1], target_long)
    output = mask_resieze_pad(mask, target_size, target_long, return_torch)
    return output[0, :, :, :]

def mask_to_bbox(mask):
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        return None
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    return [x_min, y_min, x_max, y_max]

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)  # 使用 Xavier 初始化方法，也可以根据需要选择其他初始化方法
        if m.bias is not None:
            init.constant_(m.bias, 0)  # 如果模型有偏置项，可以将其初始化为零或其他值

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

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



class ASAM(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam_model = model

    def postprocess_mask(self, low_res_masks):
        masks = F.interpolate(
            low_res_masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        return masks


    def forward(self, image, bbox, maskin):
        '''
        image: cv2.imread() > cv2.cvtColor() > transform.apply_image() >as tensro>permute>preprocess
        bbox:[x,y,x,y] > transform.apply_boxex > as_tensor > [None,:]
        maskin: mask_preprocess() > as_tensor > [None,:,:,:]
        original_image_size = original_image.shape[:2]  #(高，宽)
        input_size = image before preprocess .shape[-2:]
        '''
        image_embedding, mul_outputs = self.sam_model.image_encoder(image, maskin)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None,
        )
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = self.postprocess_mask(low_res_masks)
        return upscaled_masks, image_embedding, mul_outputs

def val(asam_model, dataloader, device):
    dataloader = tqdm(dataloader, file=sys.stdout)
    oiou_list = []
    oiou_post_list = []
    mean_iou = torch.zeros(1).to(device)
    mean_iou_post = torch.zeros(1).to(device)
    asam_model.eval()
    for step, (input_image, bbox_torch, gt_mask, maskin, occ_mask, gmask ,vmask, omask, input_size, origin_size) in enumerate(dataloader):
        image, bbox,gt_mask,maskin,gmask,occ_mask,omask = input_image.to(device), bbox_torch.to(device),gt_mask.to(device), maskin.to(device),gmask.to(device),occ_mask.to(device),omask.to(device)
        asam_pred, _ , _ = asam_model(image, bbox, maskin)#[0]
        asam_pred_post = asam_pred[..., : input_size[0], : input_size[1]]
        asam_pred_post = F.interpolate(asam_pred_post, origin_size, mode="bilinear", align_corners=False)

        if omask.sum() != 0:
            o_pred = asam_pred - maskin
            o_pred_post = o_pred[..., : input_size[0], : input_size[1]]
            o_pred_post = F.interpolate(o_pred_post, origin_size, mode="bilinear", align_corners=False)
            o_pred = o_pred > 0.0
            oiou = calculate_iou(o_pred, occ_mask)   
            oiou_list.append(oiou.item())
            o_pred_post = o_pred_post > 0.0
            oiou_post = calculate_iou(o_pred_post, omask) 
            oiou_post_list.append(oiou_post.item())        

        asam_pred = asam_pred > 0.0
        iou = calculate_iou(asam_pred, gt_mask) 
        asam_pred_post = asam_pred_post > 0.0
        iou_post = calculate_iou(asam_pred_post,gmask)

        #ap =calculate_AP(asam_pred, gt_mask.float())
        mean_iou = (mean_iou * step + iou.detach()) / (step + 1)  # update mean losses
        mean_iou_post = (mean_iou_post * step + iou_post.detach()) / (step + 1)  # update mean losses
        #torch.nn.utils.clip_grad_norm_(asam_model.parameters(), max_norm=1.0)
        if oiou_list:
            dataloader.desc = "m_iou: {}, m_iou_post: {}, m_oiou: {}, m_oiou_post: {}".format(round(mean_iou.item(), 6),round(mean_iou_post.item(), 6),mean(oiou_list),mean(oiou_post_list))
        else:
            dataloader.desc = "m_iou: {}, m_iou_post: {}".format(round(mean_iou.item(), 6),round(mean_iou_post.item(), 6))
    mean_oiou_list, mean_oiou_post_list = mean(oiou_list), mean(oiou_post_list)
    return mean_iou,mean_iou_post,  mean_oiou_list,  mean_oiou_post_list



class SA1BDataset(Dataset):
    def __init__(
        self,
        image_list, image_path, image_path_o,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.image_list = image_list
        self.image_path = image_path
        self.image_path_o = image_path_o
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.image_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

    def __getitem__(self, idx):
        image_filepath = join(self.image_path, self.image_list[idx])
        # image
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.preprocess(input_image_torch)

        image_filepath_o = join(self.image_path_o, self.image_list[idx])
        image_o = cv2.imread(image_filepath_o)
        image_o = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
        input_image_o = self._transform.apply_image(image_o)
        input_image_o_torch = torch.as_tensor(input_image_o)
        input_image_o_torch = input_image_o_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_o = self.preprocess(input_image_o_torch)

        annotation_path = image_filepath[:-3] + 'json'
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        # origin_size
        w, h = annotation_data['image']['width'], annotation_data['image']['height']
        origin_size = (h, w)
        # bbox
        annotation = annotation_data['annotations'][0]
        x, y, w, h = annotation['bbox']
        bbox = np.array([x, y, x + w, y + h])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        segmentation = mask_utils.decode(annotation['segmentation'])
        gt_mask = mask_preprocess(mask=segmentation, return_torch=True)
       
        #gt_mask = gt_mask[None, :, :, :] 
        # maskin
        occ_mask = mask_utils.decode(annotation['occluder_mask'])
        #maskin = maskin[None, :, :, :]
        v_mask = (segmentation & ~occ_mask)
        maskin = mask_preprocess(mask=v_mask, return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, occ_mask


class KINSDataset_v(Dataset):
    def __init__(
        self,
        annotations_list, image_path, kins,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.annotations_list = annotations_list
        self.image_path = image_path
        self.kins = kins
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.annotations_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

    def __getitem__(self, idx):
        annotation = self.annotations_list[idx]
        image_id = annotation['image_id']
        image_info = self.kins.loadImgs([image_id])[0]
        image_name = image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image_filepath = join(self.image_path,image_name)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.preprocess(input_image_torch)
        # input_size
        input_size = input_image_torch.shape[-2:]
        origin_size = (height, width)
        x, y, w, h = annotation['a_bbox']
        bbox = np.array([x, y, x + w, y + h])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['a_segm']
        gmask = polys_to_mask(a_polys,height,width)
        i_polys = annotation['i_segm']
        vmask = polys_to_mask(i_polys,height,width)
        omask = gmask - vmask
        gt_mask = mask_preprocess(mask=gmask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)
        occ_mask = mask_preprocess(mask=omask, return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, occ_mask, gmask ,vmask, omask, input_size, origin_size  
    
class KINSDataset_0(Dataset):
    def __init__(
        self,
        annotations_list, image_path, kins,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.annotations_list = annotations_list
        self.image_path = image_path
        self.kins = kins
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.annotations_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

    def __getitem__(self, idx):
        annotation = self.annotations_list[idx]
        image_id = annotation['image_id']
        image_info = self.kins.loadImgs([image_id])[0]
        image_name = image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image_filepath = join(self.image_path,image_name)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.preprocess(input_image_torch)
        # input_size
        input_size = input_image_torch.shape[-2:]
        origin_size = (height, width)
        x, y, w, h = annotation['a_bbox']
        bbox = np.array([x, y, x + w, y + h])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['a_segm']
        gmask = polys_to_mask(a_polys,height,width)
        i_polys = annotation['i_segm']
        omask = gmask - vmask    #先计算出遮挡区域
        if (a_polys - i_polys).sum()==0:
            i_polys = np.zeros_like(a_polys)
        vmask = polys_to_mask(i_polys,height,width)
        gt_mask = mask_preprocess(mask=gmask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)

        occ_mask = mask_preprocess(mask=omask, return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, occ_mask, gmask ,vmask, omask, input_size, origin_size  



class COCOADataset_0(Dataset):
    def __init__(
        self,
        annotations_list, image_path, kins,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.annotations_list = annotations_list
        self.image_path = image_path
        self.kins = kins
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.annotations_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def __getitem__(self, idx):
        annotation = self.annotations_list[idx]
        image_id = annotation['image_id']
        image_info = self.kins.loadImgs([image_id])[0]
        image_name = image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image_filepath = join(self.image_path,image_name)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_size = input_image_torch.shape[-2:]
        input_image = self.preprocess(input_image_torch)

        origin_size = (height, width)
        

        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['segmentation']
        a_polys = [a_polys]
        gmask = polys_to_mask(a_polys,height,width)
        x1,y1,x2,y2 = mask_to_bbox(gmask)
        keys = annotation.keys()
        if 'visible_mask' in keys:
            visible_mask = annotation['visible_mask']
            vmask = mask_utils.decode(visible_mask)
        else:
            vmask = np.zeros_like(gmask)
        if 'invisible_mask' in keys:
            omask = annotation['invisible_mask']
        else:
            omask = np.zeros_like(gmask)
        gt_mask = mask_preprocess(mask=gmask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)
        
        bbox = np.array([x1, y1, x2, y2])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        occ_mask = mask_preprocess(mask=omask, return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, occ_mask, gmask ,vmask, omask, input_size, origin_size  

class COCOADataset_v(Dataset):
    def __init__(
        self,
        annotations_list, image_path, kins,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.annotations_list = annotations_list
        self.image_path = image_path
        self.kins = kins
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.annotations_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


    def __getitem__(self, idx):
        annotation = self.annotations_list[idx]
        image_id = annotation['image_id']
        image_info = self.kins.loadImgs([image_id])[0]
        image_name = image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image_filepath = join(self.image_path,image_name)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_size = input_image_torch.shape[-2:]
        input_image = self.preprocess(input_image_torch)

        origin_size = (height, width)
        

        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['segmentation']
        a_polys = [a_polys]
        gmask = polys_to_mask(a_polys,height,width)
        x1,y1,x2,y2 = mask_to_bbox(gmask)
        keys = annotation.keys()
        if 'visible_mask' in keys:
            visible_mask = annotation['visible_mask']
            vmask = mask_utils.decode(visible_mask)
        else:
            vmask = gmask.copy()
        if 'invisible_mask' in keys:
            omask = annotation['invisible_mask']
        else:
            omask = np.zeros_like(gmask)
        gt_mask = mask_preprocess(mask=gmask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)
        
        bbox = np.array([x1, y1, x2, y2])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        occ_mask = mask_preprocess(mask=omask, return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, occ_mask, gmask ,vmask, omask, input_size, origin_size  