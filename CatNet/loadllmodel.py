from option import args1
from datacode import dataset1
import torchvision.utils as vutils
import os
import torch
from tqdm import tqdm
import torch.utils.data as data
from models import network

import glob
import sys
# from collections import OrderedDict
#from skimage.measure import compare_psnr
from natsort import natsort
import argparse
import options.options as option
#from Measure import Measure
# from Measure import Measure
# from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2,numpy
from skimage.measure import compare_psnr

import torch.nn as nn



# def fiFindByWildcard(wildcard):
#     return natsort.natsorted(glob.glob(wildcard, recursive=True))
#
def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt



def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    AAA = result
    return result  # 直方圖均衡化操作
def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt

def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)

def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]

def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])

def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]

def impad(img, top=0, bottom=0, left=0, right=0, color=255):
        return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')

parser = argparse.ArgumentParser()
parser.add_argument("--opt", default="confs/LOL_smallNet.yml")
args = parser.parse_args()
conf_path = args.opt
conf = conf_path.split('/')[-1].replace('.yml', '')
model, opt = load_model(conf_path)  # 加载模型
model.netG = model.netG.cuda()


class LLFlow1(nn.Module):
    def __init__(self):
        super(LLFlow1, self).__init__()
    def forward(self,x):

        lr1=x
        his = hiseq_color_cv2_img(lr1)

        lr_t = t(lr1)  #1,3,400,600
        lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
        his = t(his)  # 1,3,400,600

        lr_t = torch.cat([lr_t, his], dim=1)  # 1,6,400,600

        with torch.cuda.amp.autocast():
            sr_t1 = model.get_sr(lq=lr_t.cuda(), heat=None)

        return sr_t1




