
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models import basicblock as B


def sum(x, device):
    pi = torch.tensor(math.pi)
    w = x.size()[-2]
    h = x.size()[-1]
    eh = 6.0 * (w - 2.0) * (h - 2.0)
    r = noise_esti(x, device)
    sr = torch.sum(torch.abs(r), (2, 3))[0]
    sumr = 2 * (torch.sqrt(pi / 2.0) * (1.0 / eh)) * (sr)

    return sumr#相当于粗噪声


def noise_esti(x, device):
    
    a = [1, -2, 1, -2, -4, -2, 1, -2, 1]
    #kernel = torch.tensor(a).reshape(1, 1, 3, 3).float().to(device)
    kernel = torch.tensor(a).repeat(1, x.size(1), 3, 3).float().to(device)
    b = F.conv2d(input=x, weight=kernel, stride=3, padding=1)
    return b#相当于公式中y卷积N


class IRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=24, nc=32):
        super(IRCNN, self).__init__()
        self.model = B.IRCNN(in_nc, out_nc, nc).cuda()
        self.device = torch.device('cuda')


    def forward(self, x):
        row, col = x.size()[-2], x.size()[-1]
        # lam, lam1, lam2 = sum(x, self.device)
        # # am = torch.zeros(1, row, col).to(self.device) + lam#
        # # am1 = torch.zeros(1, row, col).to(self.device) + lam1
        # # am2 = torch.zeros(1, row, col).to(self.device) + lam2
        #
        # n = self.model(x)#n相当于细噪声
        # print('1111',n[0:, 0:8:, :].shape)
        # l1 = torch.where((n[0:, 0:8:, :] + lam) > 0, (n[0:, 0:8:, :] + lam), am)
        # l2 = torch.where((n[0:, 8:16:, :] + lam1) > 0, (n[0:, 8:16:, :] + lam1), am1)
        # l3 = torch.where((n[0:, 16:24:, :] + lam2) > 0, (n[0:, 16:24:, :] + lam2), am2)
        # level = torch.cat((l1, l2, l3), 1)
        # return l1, l2, l3, level

        lam = sum(x, self.device)
        am = torch.zeros(1, row, col).to(self.device) + lam
        # am1 = torch.zeros(1, row, col).to(self.device) + lam1
        # am2 = torch.zeros(1, row, col).to(self.device) + lam2
        n = self.model(x)
        #l1 = torch.where((n[0:, 0:8:, :] + lam) > 0, (n[0:, 0:8:, :] + lam), am)
        l1 = torch.where((n[0:, 0:3:, :] + lam) > 0, (n[0:, 0:3:, :] + lam), am)
        # l2 = torch.where((n[0:, 8:16:, :] + lam1) > 0, (n[0:, 8:16:, :] + lam1), am1)
        # l3 = torch.where((n[0:, 16:24:, :] + lam2) > 0, (n[0:, 16:24:, :] + lam2), am2)
        # level = torch.cat((l1, l2, l3), 1)
        level = l1
        return  level