import os
import torch.nn.functional as F
from functools import partial
from typing import Tuple
import torch
import torch.nn as nn
import sys

# 1. 获取当前脚本所在的绝对路径 (/home/smart/Road/networks)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 构造包含 'dinounet' 包的父目录路径
# 这里我们要指向: /home/smart/Road/networks/DinoUNet
lib_path = os.path.join(current_dir, 'DinoUNet')
# 3. 把这个路径加到 sys.path 的最前面，确保优先搜索
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
sys.path.append(".")
sys.path.append("..")

from dinounet.dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinounet.dinov3.hub.backbones import dinov3_vits16

from .Fusion_CD import SF_Fusion
# from .WTCenter import WT_Decoder, WT_Center
# from .WTCenterV2 import  WT_CenterV2
from .Wavelet_Mamba_Prior import WT_Center_Prior

nonlinearity = partial(F.silu, inplace=True)

def create_dinov3_encoder(pretrained_path: str, freeze_backbone: bool = True) -> Tuple[nn.Module, int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = dinov3_vits16(pretrained=False)

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"DINOv3权重文件不存在: {pretrained_path}")
    pretrained_weights = torch.load(pretrained_path, map_location=device)
    if "backbone" in next(iter(pretrained_weights.keys()), ""):
        pretrained_weights = {k.replace("backbone.", ""): v for k, v in pretrained_weights.items()}
    backbone.load_state_dict(pretrained_weights, strict=False)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    dinov3_adapter = DINOv3_Adapter(
        backbone=backbone,
        interaction_indexes=[2, 5, 8, 11],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
    )
    return dinov3_adapter, backbone.embed_dim


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
    
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


class DecoderBlock_MultiScale(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock_MultiScale, self).__init__()
        self.conv1 = MultiScaleDWConv(in_channels)
        self.conv_11 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = MultiScaleDWConv(in_channels // 4)
        self.conv31 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_11(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv31(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DinoCD(nn.Module):
    def __init__(self, img_size=512, num_classes=2, mode='baseline'):
        super(DinoCD, self).__init__()
        self.img_size=img_size
        self.mode = mode
        filters = [48, 96, 192, 384]
        
        # Encoder
        self.encoder, self.embed_dim = create_dinov3_encoder(
            pretrained_path = '/home/207lab/DINOv3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            freeze_backbone = True
        )
        
        # Skip Connections
        if mode == 'baseline':
            
            self.decoder4 = DecoderBlock(384, filters[2])
            self.decoder3 = DecoderBlock(filters[2], filters[1])
            self.decoder2 = DecoderBlock(filters[1], filters[0])
            self.decoder1 = DecoderBlock(filters[0], filters[0])
            
        elif mode == 'fusion':
            self.fusion_1 = SF_Fusion(in_channels=384, out_channels=192)
            self.fusion_2 = SF_Fusion(in_channels=384, out_channels=96)
            self.fusion_3 = SF_Fusion(in_channels=384, out_channels=48)
            
            self.decoder4 = DecoderBlock(384, filters[2])
            self.decoder3 = DecoderBlock(filters[2], filters[1])
            self.decoder2 = DecoderBlock(filters[1], filters[0])
            self.decoder1 = DecoderBlock(filters[0], filters[0])
            
        elif mode == 'center':
            self.center = WT_Center_Prior(in_channels=384, out_channels=384)
            self.decoder4 = DecoderBlock(384, filters[2])
            self.decoder3 = DecoderBlock(filters[2], filters[1])
            self.decoder2 = DecoderBlock(filters[1], filters[0])
            self.decoder1 = DecoderBlock(filters[0], filters[0])
            
        # elif mode == 'fusion_decoder':
        #     self.center = WT_Center(in_channels=384, out_channels=384)

        #     self.fusion_1 = SF_Fusion(in_channels=384, out_channels=192)
        #     self.fusion_2 = SF_Fusion(in_channels=384, out_channels=96)
        #     self.fusion_3 = SF_Fusion(in_channels=384, out_channels=48)
            
        #     self.decoder4 = WT_Decoder(384, filters[2])
        #     self.decoder3 = WT_Decoder(filters[2], filters[1])
        #     self.decoder2 = DecoderBlock(filters[1], filters[0])
        #     self.decoder1 = DecoderBlock(filters[0], filters[0])
        
        elif mode == 'fusion_center':
            
            self.center = WT_Center_Prior(in_channels=384, out_channels=384)

            self.fusion_1 = SF_Fusion(in_channels=384, out_channels=192)
            self.fusion_2 = SF_Fusion(in_channels=384, out_channels=96)
            self.fusion_3 = SF_Fusion(in_channels=384, out_channels=48)
            
            self.decoder4 = DecoderBlock_MultiScale(384, filters[2])
            self.decoder3 = DecoderBlock_MultiScale(filters[2], filters[1])
            self.decoder2 = DecoderBlock_MultiScale(filters[1], filters[0])
            self.decoder1 = DecoderBlock_MultiScale(filters[0], filters[0])
            
        # elif mode == 'fusion_center_v2':
            
        #     self.center =   WT_CenterV2(in_channels=384, out_channels=384)
            
        #     self.fusion_1 = SF_Fusion(in_channels=384, out_channels=192)
        #     self.fusion_2 = SF_Fusion(in_channels=384, out_channels=96)
        #     self.fusion_3 = SF_Fusion(in_channels=384, out_channels=48)
            
        #     self.decoder4 = DecoderBlock_MultiScale(384, filters[2])
        #     self.decoder3 = DecoderBlock_MultiScale(filters[2], filters[1])
        #     self.decoder2 = DecoderBlock_MultiScale(filters[1], filters[0])
        #     self.decoder1 = DecoderBlock_MultiScale(filters[0], filters[0])
            
            

        # final conv
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalacti1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalacti2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x1, x2):
        
        # Encoder
        feats_1 = self.encoder(x1)
        feats_2 = self.encoder(x2)
        
        feat_11, feat_12, feat_13, feat_14 = feats_1["1"], feats_1["2"], feats_1["3"], feats_1["4"]
        feat_21, feat_22, feat_23, feat_24 = feats_2["1"], feats_2["2"], feats_2["3"], feats_2["4"]

        if self.mode == 'baseline':
            
            d4 = self.decoder4(feat_14 + feat_24)
            d3 = self.decoder3(d4)
            d2 = self.decoder2(d3)
            d1 = self.decoder1(d2)
        
        elif self.mode == 'fusion':
            
            d4 = self.decoder4(feat_14 + feat_24)
            d3 = self.decoder3(d4 + self.fusion_1(feat_13, feat_23))
            d2 = self.decoder2(d3 + self.fusion_2(feat_12, feat_22))
            d1 = self.decoder1(d2 + self.fusion_3(feat_11, feat_21))
        
        elif self.mode == 'center':
            center_feat = self.center(feat_14 + feat_24)
            d4 = self.decoder4(center_feat +feat_14 + feat_24)
            d3 = self.decoder3(d4)
            d2 = self.decoder2(d3)
            d1 = self.decoder1(d2)
            
        # elif self.mode == 'fusion_decoder':
        #     d4 = self.decoder4(feat_14 + feat_24 )
        #     d3 = self.decoder3(d4 + self.fusion_1(feat_13, feat_23))
        #     d2 = self.decoder2(d3 + self.fusion_2(feat_12, feat_22))
        #     d1 = self.decoder1(d2 + self.fusion_3(feat_11, feat_21))
        
        elif self.mode == 'fusion_center':
            center_feat = self.center(feat_14 + feat_24)
            d4 = self.decoder4(center_feat)
            d3 = self.decoder3(d4 + self.fusion_1(feat_13, feat_23))
            d2 = self.decoder2(d3 + self.fusion_2(feat_12, feat_22))
            d1 = self.decoder1(d2 + self.fusion_3(feat_11, feat_21))

        # elif self.mode == 'fusion_center_v2':
        #     center_feat = self.center(feat_14, feat_24)
        #     d4 = self.decoder4(center_feat)
        #     d3 = self.decoder3(d4 + self.fusion_1(feat_13, feat_23))
        #     d2 = self.decoder2(d3 + self.fusion_2(feat_12, feat_22))
        #     d1 = self.decoder1(d2 + self.fusion_3(feat_11, feat_21))
        
        out = self.finaldeconv1(d1)
        out = self.finalacti1(out)
        out = self.finalconv2(out)
        out = self.finalacti2(out)
        out = self.finalconv3(out)
          
        return out
    
if __name__ == "__main__":
    x_1 = torch.randn(1, 3, 256, 256).cuda()
    x_2 = torch.randn(1, 3, 256, 256).cuda()

    model = DinoCD(img_size=256, num_classes=2, mode='fusion_center').cuda()
    output = model(x_1, x_2)
    print(output.shape)  # 输出的形状
    # # 假设 model 是你的模型，x_1 和 x_2 是输入
    #     output = model(x_1, x_2)
    #     print(output.shape)  # 输出的形状

    # 计算所有参数的总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 计算可训练参数的总数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # 计算不可训练参数的总数
    non_trainable_params = total_params - trainable_params
    print(f"Non-trainable parameters: {non_trainable_params}")

    # 计算 FLOPs 和参数量
    from thop import profile
    flops, params = profile(model, inputs=(x_1, x_2))
    print(f"FLOPs: {flops}, Parameters: {params}")