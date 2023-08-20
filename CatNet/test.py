

#import data
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
from collections import OrderedDict
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
from skimage.measure import compare_psnr, compare_ssim
from loadllmodel import LLFlow1
import time
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

def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result







def test(model, args1, loader_test, device):

    torch.set_grad_enabled(False)
    testLoader = data.DataLoader(dataset=loader_test, batch_size=args1.batch_size, shuffle=True,
                                 num_workers=args1.n_threads, pin_memory=True)
    sum=0
    sum1=0
    index=0
    for batch, (lr, hr,x) in enumerate(tqdm(testLoader, ncols=80)):

        torch.set_grad_enabled(False)
        lr = lr.to(device)

        x1 = lr.cpu().numpy()
        x1 = x1.squeeze(0).transpose(1, 2, 0)
        lr1 = (x1 * 255).astype(numpy.uint8)
        llfow = LLFlow1()
        sr_t1 = llfow(lr1)
        mean_out = sr_t1.view(sr_t1.shape[0], -1).mean(dim=1).cuda()  # 1 rensor
        mean_gt = hr.view(hr.shape[0], -1).mean(dim=1).cuda()
        sr_t1 = torch.clamp(sr_t1 * (mean_gt / mean_out), 0, 1).cuda()

        eval_out,  xt_eval= model(sr_t1,lr)
        image3 = eval_out.clamp(0, 1).cuda()
        vutils.save_image(image3, './result/{}lol/{}.png'.format(args1.data_name, ''.join(x), index))

        eval_out = eval_out.clamp(0, 1).cuda()
        eval_out = eval_out.cpu().detach().numpy()
        eval_out1=eval_out.squeeze(0).transpose(1, 2, 0)
        #
        hr = hr.cpu().detach().numpy()
        hr1 = hr.squeeze(0).transpose(1, 2, 0)

        psnr = compare_psnr(eval_out, hr)
        ssim=compare_ssim(eval_out1,hr1,multichannel=True)

        sum = sum + psnr
        sum1 = sum1 + ssim
        index = index + 1


    print(sum/index)
    print(sum1 / index)


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda')
    global modele
    net = network.UTVNet().to(device)
    net.load_state_dict(torch.load("F:\ZCZ\8.12test\pretrained model/best.pt", map_location=device))

    test_input_dir = './LOL/VE-LOL/{}/'.format(args1.data_name)
    test_input_dir2 = ''
    test_gt_dir = './LOL/VE-LOL/high/'.format(args1.data_name)
    loaderTest = dataset1.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir,'test', '600', args1.data_name)

    test(net, args1, loaderTest, device)


if __name__ == '__main__':
    main()
