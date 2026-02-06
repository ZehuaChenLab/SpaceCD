from .module_ss2d_e import SS2D as SS2D_Prior
# from VMamba import SS2D
import torch
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import torch
import torch.nn.functional as F
# from functools import partial


###########################################################################
# 2d wavelet transform
def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

if  __name__ == "__main__":
    
    filters, _ = create_2d_wavelet_filter('db1', 1, 1, torch.float)
    print(filters.shape)
def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
###########################################################################


###########################################################################
# hh enhance module
class MultiScale_DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(MultiScale_DWConv, self).__init__()

        self.conv55 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=5, padding=2, stride=stride, dilation=1, groups=in_channels//4, bias=bias)
        self.conv33 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, padding=1, stride=stride, dilation=1, groups=in_channels//4, bias=bias)
        self.conv77 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=7, padding=3, stride=stride, dilation=1, groups=in_channels//4, bias=bias)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
    def forward(self, x):
        
        x1, x2, x3, x4 = x.chunk(chunks=4, dim=1)
        x2 = self.conv55(x2)
        x3 = self.conv33(x3)
        x4 = self.conv77(x4)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.pconv(x_cat)
        return out
###########################################################################

###########################################################################
# scale module
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
###########################################################################

###########################################################################
# WT convolution with fusion prior
class WTConv2d_prior(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, wt_type='db1'):
        super(WTConv2d_prior, self).__init__()

        self.in_channels = in_channels
        self.dilation = 1

        # create wavelet filters
        self.wt_filter, self.iwt_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        # create hh enhance module
        self.hh_enhance = MultiScale_DWConv(in_channels, in_channels, stride=1, bias=bias)
        
        # create prior mamba with ss2d to ll
        self.ll_lh_ss2d = SS2D_Prior(d_model=in_channels, prior=True)
        self.ll_hl_ss2d = SS2D_Prior(d_model=in_channels, prior=True)
        self.ll_hh_ss2d = SS2D_Prior(d_model=in_channels, prior=True)
        
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.scale = _ScaleModule(dims=(1, out_channels, 1, 1), init_scale=1.0)
        
        
        # 消融实验1, 不进行wavelet transform 仅进行ss2d增强 和 base_conv
        # self.ablation_ss2d = SS2D(d_model=in_channels,forward_type="v02_noz")
        
        # 消融实验2 不进行引导ss2d, 仅使用普通ss2d增强
        # self.ll_lh_ss2d = SS2D(d_model=in_channels,forward_type="v02_noz")
        # self.ll_hl_ss2d = SS2D(d_model=in_channels,forward_type="v02_noz")
        # self.ll_hh_ss2d = SS2D(d_model=in_channels,forward_type="v02_noz")
        
        # 消融实验3 不进行H-enhance, 仅使用先验ss2d增强
        
    def forward(self, x):

        x_wave = x.clone()
        
        # 保证输入尺寸为偶数
        curr_shape = x_wave.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            x_wave = F.pad(x_wave, curr_pads)

        # 2d wavelet transform
        curr_x = wavelet_2d_transform(x_wave, self.wt_filter)
        x_wave_ll = curr_x[:,:,0,:,:]
        x_wave_lh = curr_x[:,:,1,:,:]
        x_wave_hl = curr_x[:,:,2,:,:]
        x_wave_hh = curr_x[:,:,3,:,:]
        
        # enhance lh, hl, hh in spatial
        x_hh = torch.concat([x_wave_lh,
                             x_wave_hl,
                             x_wave_hh], dim=0)
        
        x_enhhance_hh = self.hh_enhance(x_hh)
        
        
        x_enhance_lh, x_enhance_hl, x_enhance_hh = torch.chunk(x_enhhance_hh, chunks=3, dim=0)
        
        # x_enhance_lh, x_enhance_hl, x_enhance_hh = torch.chunk(x_hh, chunks=3, dim=0)
        
        # # # 此时, ll不变，lh, hl, hh通过ss2d prior增强
        x_enhance_ll_lh = self.ll_lh_ss2d(x=x_wave_ll.permute(0,2,3,1), d=x_enhance_lh.permute(0,2,3,1)).permute(0,3,1,2)
        x_enhance_ll_hl = self.ll_hl_ss2d(x=x_wave_ll.permute(0,2,3,1), d=x_enhance_hl.permute(0,2,3,1)).permute(0,3,1,2)
        x_enhance_ll_hh = self.ll_hh_ss2d(x=x_wave_ll.permute(0,2,3,1), d=x_enhance_hh.permute(0,2,3,1)).permute(0,3,1,2)
        
        # 普通SS2D
        # x_enhance_ll_lh = self.ll_lh_ss2d(x=x_enhance_lh.permute(0,2,3,1)).permute(0,3,1,2)
        # x_enhance_ll_hl = self.ll_hl_ss2d(x=x_enhance_hl.permute(0,2,3,1)).permute(0,3,1,2)
        # x_enhance_ll_hh = self.ll_hh_ss2d(x=x_enhance_hh.permute(0,2,3,1)).permute(0,3,1,2)
        


        # inverse 2d wavelet transform
        curr_x = torch.cat([x_wave_ll.unsqueeze(2),
                            x_enhance_ll_lh.unsqueeze(2),
                            x_enhance_ll_hl.unsqueeze(2),
                            x_enhance_ll_hh.unsqueeze(2)], dim=2)
        next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
        
        # 消融实验1 不进行wavelet transform 仅进行ss2d增强
        # next_x_ll = self.ablation_ss2d(x.permute(0,2,3,1)).permute(0,3,1,2)
        
        x_final = self.scale(self.base_conv(next_x_ll + x))

        return x_final
###########################################################################
    
###########################################################################
# MLP with multi-scale depth-wise conv
class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        return x
###########################################################################

class WT_Center_Prior(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, wt_type='db1'):
        super(WT_Center_Prior, self).__init__()

        self.wt_conv = WTConv2d_prior(in_channels, out_channels, bias=bias, wt_type=wt_type)
        self.mlp = Mlp(out_channels, out_channels*2, out_channels)
        
    def forward(self, x):
        out = self.wt_conv(x)
        out = self.mlp(out)         
        return out

if __name__ == '__main__':
    x = torch.randn(2, 64, 64, 64).cuda()

    model = WT_Center_Prior(in_channels=64, out_channels=64, bias=True, wt_type='db1').cuda()

    y = model(x)
    print(y.shape)