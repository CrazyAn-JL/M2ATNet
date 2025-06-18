import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.CFFT import *
from net.IEN import Ienhance
from net.HV_DE import HV_En
from net.CFFT import HV_I

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Ienhance = Ienhance()
        self.HV_En = HV_En()
        self.HV_I = HV_I(32)
        self.i_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 1, 3, stride=1, padding=0,bias=False)
        )
        self.HV_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 2, 3, stride=1, padding=0,bias=False)
        )
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # hv = hvi[:,:2,:,:].to(dtypes)
        i_en =self.Ienhance(i)
        hv_en = self.HV_En(hvi)
        hv_en1 = self.HV_I(hv_en, i_en)
        i_en2 = self.i_conv(i_en)
        hv_en2 = self.HV_conv(hv_en1)
        output_hvi = torch.cat([hv_en2, i_en2], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi


