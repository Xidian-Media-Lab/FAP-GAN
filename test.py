import cv2
import torch
import glob
import os
import time  # 保持完整导入，避免命名冲突
import thop
from thop import profile
import torchvision
from args import args
import numpy as np
from PIL import Image, ImageDraw
import torch.amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import itertools
import torchvision.transforms as transforms
import functools
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
from comput_loss import MultiStagePerceptualLoss
from torch.utils.data import Dataset, DataLoader
from dataset import dataloader
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
import math
from statistics import mean
from tensorboardX import SummaryWriter
from network1 import RefGuidedRestorationNet, LightweightDiscriminator
import os


# parameters

def loadG(dir_chck, GA_B, optimG=[], epoch=[],
         mode='train'):
    if not epoch:
        ckpt = os.listdir(dir_chck)
        ckpt.sort()
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

    dict_net = torch.load('%s/modelGA1_epoch%04d.pth' % (dir_chck, epoch))

    print('Loaded %dth network' % epoch)

    if mode == 'train':
        GA_B.load_state_dict(dict_net['GA_B'])
        optimG.load_state_dict(dict_net['optimG'])

        return GA_B, optimG, epoch

    elif mode == 'test':

        GA_B.load_state_dict(dict_net['GA_B'])

        return  GA_B, epoch


def loadD(dir_chck, Dis=[], optimD=[], epoch=[], mode='train'):
    if not epoch:
        ckpt = os.listdir(dir_chck)
        ckpt.sort()
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

    dict_net = torch.load('%s/modelD1_epoch%04d.pth' % (dir_chck, epoch))

    print('Loaded %dth network' % epoch)

    if mode == 'train':

        Dis.load_state_dict(dict_net['Dis'])
        optimD.load_state_dict(dict_net['optimD'])

        return Dis, optimD, epoch

G1A_B = RefGuidedRestorationNet(base_ch=64).cuda(args.gpu)

# 创建一个测试输入（假设输入图像大小为 128x128）



is_cuda = torch.cuda.is_available()

G1A_B.eval()


G1A_B, st_epoch = loadG(args.dir_chck, G1A_B, epoch=94, mode="test")
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f"Model G1A_B Parameters: {count_model_parameters(G1A_B):,}")

if is_cuda:
    G1A_B.cuda(args.gpu)
    # GB_A.cuda(args.gpu)

qps=["22","32","42","52"]

for qp in qps:
    # for i in range(1,16):
    # for i in range(1,135):
    # for i in range(141,159):
    #for i in range(161,176):
    for i in range(301,334):
# dataloader

        # gt_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/gfvc_v1-AhG16_GFVC_Software_v4/source/experiment/Rec_frames/VOXCELEB_"+f"{i:03d}"+"_"+f"{qp}"+"/"]
        # LR_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/gfvc_v1-AhG16_GFVC_Software_v4/source/experiment/Rec_frames/VOXCELEB_"+f"{i:03d}"+"_52"+"/"]

        gt_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/"+ f"{qp}"+"/video_"+f"{i}"+"/"]
        LR_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/traindataset/HQ/HQ_compress/LRdecoded/52"+"/video_"+f"{i}"+"/"]



        save_path_A  = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/multiresolutionGFVC/test/C256/CFVQA_"+f"{i:03d}"+"_"+f"{qp}"+"/"]
        if not os.path.exists(save_path_A[0]):
            os.makedirs(save_path_A[0],exist_ok=True)
        os.makedirs(save_path_A[0]+"input/", exist_ok=True)
        os.makedirs(save_path_A[0]+"gt/", exist_ok=True)
        os.makedirs(save_path_A[0]+"pred_gt/", exist_ok=True)
        for i in range(0,len(LR_list)):
            LR_inputdir = LR_list[i]
            LR_input = os.listdir(LR_inputdir)
            LR_input.sort()

            gtdir = gt_list[i]
            gt = os.listdir(gtdir)
            gt.sort()
            total_inference_time = 0
            num_samples = len(LR_input)


            for j in range(0, len(LR_input)):
                j = j
                H = 256
                W = 256
                ref_img_3_path = gtdir + gt[1]
                gt_img_3_path = gtdir + gt[j]
                QP52_img_3_path = LR_inputdir + LR_input[j]

                if QP52_img_3_path.endswith(".png") and gt_img_3_path.endswith(".png"):

                    ref_img_3 = Image.open(ref_img_3_path).convert('RGB')
                    QP52_img_3 = Image.open(QP52_img_3_path).convert('RGB')
                    gt_img_3 = Image.open(gt_img_3_path).convert('RGB')
                    gt_img_3 = gt_img_3.resize((H, W), Image.BILINEAR)
                    ref_img_3 = ref_img_3.resize((H, W), Image.BILINEAR)
                    QP52_img_3 = QP52_img_3.resize((H, W), Image.BILINEAR)

                    QP52_img_3.save(f"{save_path_A[0]}input/{str(j).zfill(4)}_QP22_img_3.png")
                    ref_img_3.save(f"{save_path_A[0]}input/{str(j).zfill(4)}_ref_img_3.png")
                    gt_img_3.save(f"{save_path_A[0]}gt/{str(j).zfill(4)}_gt_img_3.png")




                    lr_16_3 = np.array(QP52_img_3)
                    ref_img_3 = np.array(ref_img_3)
                    gt_img_3 = np.array(gt_img_3)

                    ref_img_3 = torch.from_numpy(ref_img_3.transpose((2, 0, 1)))
                    ref_img_3 = ref_img_3.float().div(255)

                    gt_img_3 = torch.from_numpy(gt_img_3.transpose((2, 0, 1)))
                    gt_img_3 = gt_img_3.float().div(255)

                    lr_16_3 = torch.from_numpy(lr_16_3.transpose((2, 0, 1)))
                    lr_16_3 = lr_16_3.float().div(255)


                    if is_cuda and args.gpu is not None:
                        ref_img_3 = ref_img_3.cuda(args.gpu).unsqueeze(0)
                        lr_16_3 = lr_16_3.cuda(args.gpu).unsqueeze(0)
                        gt_img_3 = gt_img_3.cuda(args.gpu).unsqueeze(0)

                    G1A_B = G1A_B.cuda(args.gpu)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    with torch.no_grad():
                        pred_42 = G1A_B(lr_16_3, ref_img_3)

                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time

                    pred_42 = pred_42.detach().cpu()
                    pred_42 = pred_42.squeeze(0)
                    pred_42 = pred_42.numpy().transpose(1,2,0)
                    pred_42 = (pred_42 * 255).clip(0, 255).astype(np.uint8)
                    pred_42 = Image.fromarray(pred_42, mode='RGB')
                    pred_42.save(f"{save_path_A[0]}pred_gt/{str(j).zfill(4)}_pred_gt.png")
                    print(f"Processed frame {j}/{num_samples}, inference time: {inference_time:.4f} sec")
                    print(j)

                avg_inference_time = total_inference_time / num_samples
                print(f"Average inference time per frame: {avg_inference_time * 1000:.2f} ms")
                # 计算模型的 MACs
    input_tensor1 = torch.randn(1, 3, 256, 256).cuda()
    input_tensor2 = torch.randn(1, 3, 256, 256).cuda()

    macs_G1A_B, params_G1A_B = profile(G1A_B, inputs=(input_tensor1,input_tensor2))  
    # 转换为 KMAC
    kmac_G1A_B = macs_G1A_B / (1e3*256*256)
    print(f"G1A_B: {kmac_G1A_B:.2f} KMACs")





