import torch
import sys
import cv2
import torch.nn.functional as F
import monai
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn
import torch
from typing import Any, Dict, List, Tuple
#from skimage import transform
from tqdm import tqdm
import numpy as np
import os
from pycocotools import mask as mask_utils
import json
from statistics import mean
join = os.path.join
import torch.nn.init as init
from PIL import Image
from torchvision.transforms import Resize
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

def mask_to_bbox(mask):
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        return None
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    return [x_min, y_min, x_max, y_max]

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
        occlude_image_embedding,featurs = self.sam_model.image_encoder(image, maskin)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None,
        )
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=occlude_image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = self.postprocess_mask(low_res_masks)
        return upscaled_masks, occlude_image_embedding, featurs

class SAM_o(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam_model = model
    
    def forward(self, image):
        origin_image_embedding,features = self.sam_model.image_encoder(image)
        return origin_image_embedding,features
    
class MaskDiscriminator(nn.Module):
    def __init__(self):
        super(MaskDiscriminator, self).__init__()
        # 输入通道数：3（图像） + 1（掩码） = 4
        input_channels = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # [4, 1024, 1024] -> [64, 512, 512]
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # [64, 512, 512] -> [128, 256, 256]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # [128, 256, 256] -> [256, 128, 128]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),           # [256, 128, 128] -> [512, 64, 64]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),          # [512, 64, 64] -> [1024, 32, 32]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),         # [1024, 32, 32] -> [2048, 16, 16]
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))                  # [2048, 16, 16] -> [2048, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, mask):
        # 拼接图像和掩码在通道维度
        x = torch.cat([image, mask], dim=1)  # [batch_size, 4, 1024, 1024]
        x = self.conv1(x)  # [batch_size, 64, 512, 512]
        x = self.conv2(x)  # [batch_size, 128, 256, 256]
        x = self.conv3(x)  # [batch_size, 256, 128, 128]
        x = self.conv4(x)  # [batch_size, 512, 64, 64]
        x = self.conv5(x)  # [batch_size, 1024, 32, 32]
        x = self.conv6(x)  # [batch_size, 2048, 16, 16]
        x = self.global_avg_pool(x)  # [batch_size, 2048, 1, 1]
        x = x.view(x.size(0), -1)    # [batch_size, 2048]
        x = self.fc(x)               # [batch_size, 1]
        
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
    
        self.net1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1).cuda(),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1).cuda(),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1).cuda(),
                    nn.ReLU(),  #输出（64,32,32）
                )
        self.net2 = nn.Sequential(
                    nn.Linear(64*32*32, 128),
                    nn.ReLU(),
                    nn.Linear(128,1),
                    nn.Sigmoid(),
                )

    def forward(self, mask):
        #mask = mask_preprocess(mask)
        x = self.net1(mask)
        x = torch.flatten(x)
        prob = self.net2(x)
        return prob


def train_one_epoch_o(asam_model,sam_o, d_model, train_dataloader,epoch,optimizer, optimizer_d, device,batch_size,writer):  ###################

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) 
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    mse_loss = torch.nn.MSELoss()
    mean_loss0 = torch.zeros(1).to(device)
    mean_loss1 = torch.zeros(1).to(device)
    mean_loss2 = torch.zeros(1).to(device)
    mean_loss3 = torch.zeros(1).to(device)
    loss_d = torch.nn.BCELoss()
    asam_model.train()
    if is_main_process():
        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
    for step, (input_image, input_image_o, bbox_torch, gt_mask, vmask,omask) in enumerate(train_dataloader):
        optimizer.zero_grad()
        image, image_o, bbox,gt_mask = input_image.to(device),input_image_o.to(device), bbox_torch.to(device),gt_mask.to(device)
        omask, vmask = omask.to(device), vmask.to(device)
        asam_pred, image_feature , asam_features = asam_model(image, bbox, omask)#[0]
        image_feature_o, sam_features= sam_o(image_o)
        asam_feature0, asam_feature1, asam_feature2 = asam_features[0], asam_features[1], asam_features[2]
        sam_feature0, sam_feature1, sam_feature2 = sam_features[0], sam_features[1], sam_features[2]
    
        loss1 = 0.5 * seg_loss(asam_pred, gt_mask) 
        asam_pred_s = torch.sigmoid(asam_pred)
        gt_binary_mask1 = torch.as_tensor(gt_mask > 0,dtype=torch.float32)
        vi_binary_mask = torch.as_tensor(vmask > 0,dtype=torch.float32)
        o_pred = asam_pred_s - vi_binary_mask
        o_gt = gt_mask - vmask
        o_binary_mask = torch.as_tensor(o_gt > 0,dtype=torch.float32)
        loss2 = 20 * mse_loss(o_pred,o_binary_mask)+ 10 *mse_loss(asam_pred_s,gt_binary_mask1)
        loss3 = 0.25*(mse_loss(image_feature,image_feature_o) + mse_loss(asam_feature0,sam_feature0) + mse_loss(asam_feature1,sam_feature1) + mse_loss(asam_feature2,sam_feature2))
        g_loss = loss_d(d_model(image_o,asam_pred), torch.zeros(size=(batch_size,1),device=device,requires_grad=True))
        loss = loss1 + loss2 + loss3 + g_loss 
        
        loss.backward()
        loss = reduce_value(loss, average=True)
        loss1 = reduce_value(loss1, average=True)
        loss2 = reduce_value(loss2, average=True)
        loss3 = reduce_value(loss3, average=True)
        g_loss = reduce_value(g_loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        mean_loss0 = (mean_loss0 * step + loss1.detach()) / (step + 1)  # update mean losses
        mean_loss1 = (mean_loss1 * step + loss2.detach()) / (step + 1)  # update mean losses
        mean_loss2 = (mean_loss2 * step + loss3.detach()) / (step + 1)  # update mean losses
        mean_loss3 = (mean_loss3 * step + g_loss.detach()) / (step + 1)  # update mean losses
        if is_main_process():
            writer.add_scalar('Loss/sample', loss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('dloss/sample', loss1.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('bloss/sample', loss2.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('mloss/sample', loss3.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('g_loss/sample', g_loss.item(), epoch * len(train_dataloader) + step)
        #torch.nn.utils.clip_grad_norm_(asam_model.parameters(), max_norm=1.0)
        if is_main_process():
            train_dataloader.desc = "[epoch {}] mean loss {},dloss {}, bloss {},mloss {},gloss {}".format(epoch, round(mean_loss.item(), 6),round(mean_loss0.item(), 6),round(mean_loss1.item(), 6),round(mean_loss2.item(), 6),round(mean_loss3.item(), 6))
        optimizer.step()

        optimizer_d.zero_grad()
        d_loss = 0.5*loss_d(d_model(image_o,gt_mask), torch.ones(size=(batch_size,1),device=device,requires_grad=True)) + loss_d(d_model(image_o,asam_pred.detach()), torch.zeros(size=(batch_size,1),device=device,requires_grad=True))
        d_loss.backward()
        optimizer_d.step()
    return mean_loss.item(),mean_loss0.item(),mean_loss1.item(),mean_loss2.item(),mean_loss3.item()

@torch.no_grad()
def evaluate(asam_model, val_dataloader, device,epoch):
    mean_iou = torch.zeros(1).to(device)
    mean_oiou = torch.zeros(1).to(device)
    if is_main_process():
        val_dataloader = tqdm(val_dataloader, file=sys.stdout)
    #input_image[0], bbox_torch, gt_mask, v_mask, omask, maskin
    for step, (input_image, bbox_torch, gt_mask, v_mask,o_mask,maskin) in enumerate(val_dataloader):
        image, bbox = input_image.to(device), bbox_torch.to(device)
        gt_mask, v_mask, o_mask, maskin = gt_mask.to(device), v_mask.to(device), o_mask.to(device), maskin.to(device)
        asam_pred, _ , _ = asam_model(image, bbox, maskin)#[0]
        o_pred = asam_pred - v_mask
        o_gt = gt_mask - v_mask
        iou = calculate_iou(asam_pred,gt_mask)
        oiou = calculate_iou(o_pred,o_gt)
        
        iou = reduce_value(iou, average=True)
        oiou = reduce_value(oiou, average=True)

        mean_iou = (mean_iou * step + iou.detach()) / (step + 1)  # update mean losses
        mean_oiou = (mean_oiou * step + oiou.detach()) / (step + 1)  # update mean losses
        if is_main_process():
            val_dataloader.desc = "[epoch {}] miou {} moiou".format(epoch, round(mean_iou.item(),5), round(mean_oiou.item(),5))
    return mean_iou.item(), mean_oiou.item()


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
        occluder_mask = mask_utils.decode(annotation['occluder_mask'])
        v_mask = (segmentation & ~occluder_mask)
        v_mask = mask_preprocess(mask=v_mask, return_torch=True)
        #v_64 = mask_preprocess(mask=v_mask,target_long=64, return_torch=True)
        occluder_mask = mask_preprocess(mask=occluder_mask, return_torch=True)
        
        #maskin = maskin[None, :, :, :]

        return input_image[0], input_image_o[0], bbox_torch, gt_mask, v_mask, occluder_mask


class SA1BDataset_val(Dataset):
    def __init__(
        self,
        image_list, image_path,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.image_list = image_list
        self.image_path = image_path
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
        
        v_mask = mask_preprocess(mask=v_mask, return_torch=True)
        #v_64 = mask_preprocess(mask=v_mask, target_long=64,return_torch=True)
        omask = mask_preprocess(mask=occ_mask, target_long=1024, return_torch=True)
        maskin = mask_preprocess(mask=occ_mask,return_torch=True)
        return input_image[0], bbox_torch, gt_mask, v_mask, omask, maskin


class KINSDataset(Dataset):
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

        origin_size = (height, width)
        x, y, w, h = annotation['a_bbox']
        bbox = np.array([x, y, x + w, y + h])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['a_segm']
        gt_mask = polys_to_mask(a_polys,height,width)
        i_polys = annotation['i_segm']
        vmask = polys_to_mask(i_polys,height,width)
        omask = gt_mask - vmask
        gt_mask = mask_preprocess(mask=gt_mask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)
        omask = mask_preprocess(omask,return_torch=True)
        return input_image[0], bbox_torch, gt_mask, maskin, omask

    
class COCOADataset(Dataset):
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

        origin_size = (height, width)
        

        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        a_polys = annotation['segmentation']
        a_polys = [a_polys]
        gt_mask = polys_to_mask(a_polys,height,width)
        x1,y1,x2,y2 = mask_to_bbox(gt_mask)
        keys = annotation.keys()
        if 'visible_mask' in keys:
            visible_mask = annotation['visible_mask']
            vmask = mask_utils.decode(visible_mask)
        else:
            vmask = gt_mask  #np.zeros_like(gt_mask)
        if 'invisible_mask' in keys:
            occ_mask = annotation['invisible_mask']
            omask = mask_utils.decode(occ_mask)
        else:
            omask =   gt_mask - vmask
        gt_mask = mask_preprocess(mask=gt_mask, return_torch=True)
        maskin = mask_preprocess(mask=vmask, return_torch=True)
        omask = mask_preprocess(mask=omask,return_torch=True)
        
        bbox = np.array([x1, y1, x2, y2])
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)

        return input_image[0], bbox_torch, gt_mask, maskin, omask