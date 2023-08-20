
#import torch.fft
import torch
import torch.nn as nn
import numpy
#import D2
from models import basicblock as B
from models import ns_model,ns_model1
from models import utv
from models import nli_model
from models import lc_model




class UTVNet(nn.Module):
    #def __init__(self):
    def __init__(self):
        super(UTVNet, self).__init__()
        self.a = utv.UNet()
        self.noiselevel = nli_model.IRCNN(3, 3, 32)
        self.denoise = ns_model1.UNet()
        self.LIGHT_car = lc_model.LIRCNN(3, 3, 48)
        self.LIGHT_tex = lc_model.LIRCNN(3, 3, 48)
        self.device = torch.device("cuda")



    def forward(self, xc,x):

        level = self.noiselevel(x)
        denoise = self.denoise(x - xc, level)
        out = denoise+xc
        #ccc=out
        return out,denoise


def make_model(args, parent=False):
    return UTVNet()
