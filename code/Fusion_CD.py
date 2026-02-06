import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x):
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
    
class SF_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super(SF_Fusion, self).__init__()
        
        middle_channels = in_channels * expand_ratio
        self.layer_norm_spatial = LayerNorm2d(in_channels)
        self.spatial_conv_31 = nn.Conv2d(in_channels, middle_channels, kernel_size=(3, 1), padding=(1, 0), bias=True, groups=in_channels//4)
        self.spatial_conv_13 = nn.Conv2d(in_channels, middle_channels, kernel_size=(1, 3), padding=(0, 1), bias=True, groups=in_channels//4)
        self.spatial_conv_33 = nn.Conv2d(in_channels, middle_channels, kernel_size=(3, 3), padding=(1, 1), bias=True, groups=in_channels//4)
        self.spatial_sg = SimpleGate()
        
        self.channel_weight = nn.Sequential(            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.mlp = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        
        # self.mlp = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x1, x2):
        # x1, x2: [B, C, H, W]
        _, _, H, W = x1.shape
        
        # 实现空间域特征提取
        x_spatial = torch.abs(x1 - x2)  # 计算差异图
        x_spatial_base = x_spatial.clone()
        x_spatial = self.layer_norm_spatial(x_spatial)
        # 通过三个卷积层进行特征提取
        x_spatial_31 = self.spatial_conv_31(x_spatial)
        x_spatial_13 = self.spatial_conv_13(x_spatial)
        x_spatial_33 = self.spatial_conv_33(x_spatial)
        
        # spatial selection
        ##############################################################
        x_spatial = self.spatial_sg(x_spatial_31 + x_spatial_13 + x_spatial_33)
        ##############################################################
        
        # x_spatial = x_spatial_31 + x_spatial_13 + x_spatial_33
        
        # 实现通道域特征提取
        x1_fft = torch.fft.rfft2(x1, norm='backward')
        x2_fft = torch.fft.rfft2(x2, norm='backward')
        # 将x1的幅值 与 x2的幅值进行比较
        x_channel_amp = torch.abs(torch.abs(x1_fft) - torch.abs(x2_fft))
        x_channel_phase = torch.angle(x_channel_amp)
        
        # channel selection
        ##############################################################
        x_channel_amp_weight = self.channel_weight(x_channel_amp)
        x_channel_amp_fusion = x_channel_amp_weight * x_channel_amp
        new_fft = torch.polar(x_channel_amp_fusion, x_channel_phase)
        ##############################################################
        
        # without channel selection
        # new_fft = torch.polar(x_channel_amp, x_channel_phase)
        
        # 必须指定 s=(H, W) 以确保尺寸完全恢复
        x_restored = torch.fft.irfft2(new_fft, s=(H, W), norm='backward')
    
        # 融合空间域与通道域特征
        # multi-scale dwconv
        x_fusion = self.mlp(x_spatial * x_restored + x_spatial_base)
        return x_fusion
        
        
        
if __name__ == '__main__':
    model = SF_Fusion(384, 192)
    x1 = torch.randn(1, 384, 128, 128)
    x2 = torch.randn(1, 384, 128, 128)
    out = model(x1, x2)
    from thop import profile
    flops, params = profile(model, inputs=(x1, x2))
    print(f'FLOPs: {flops / 1e9:.2f} G')
    print(f'Params: {params / 1e6:.2f} M')
    print(out.shape)  # 应该输出 torch.Size([1, 64, 128, 128])