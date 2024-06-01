import numpy as np
import torch
import monai
import torch.nn as nn
from .STUNet import STUNet

class PCNet(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', organ_embedding = None,feat_concat = False,):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        self.feat_concat = feat_concat
        if backbone == 'swinunetr_small':
            self.backbone = monai.networks.nets.SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                feature_size=24,
                drop_rate=0,
                attn_drop_rate=0,
                dropout_path_rate=0,
            )
        elif backbone == 'swinunetr_base':
            self.backbone = monai.networks.nets.SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                dropout_path_rate=0.1,
            )
        elif backbone == 'swinunetr_large':
            self.backbone = monai.networks.nets.SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                feature_size=96,
                drop_rate=0,
                attn_drop_rate=0,
                dropout_path_rate=0,
            )
        elif backbone == 'segresnet':
            self.backbone = monai.networks.nets.SegResNet(
                blocks_down=[1,2,2,4],
                blocks_up=[1,1,1],
                init_filters=24,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                dropout_prob=0.2,
            )
        elif backbone == 'unet':
            self.backbone = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                channels=(16,32,64,128,256),
                strides=(2,2,2,2),
                num_res_units=2,
                norm=monai.networks.layers.Norm.BATCH,
            )
        elif backbone == 'unet_large':
            self.backbone = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                channels=(32,64,128,256,512),
                strides=(2,2,2,2),
                num_res_units=2,
                norm=monai.networks.layers.Norm.BATCH,
            )
        elif backbone == 'unetpp':
            self.backbone = monai.networks.nets.BasicUNetPlusPlus(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
                features=(32,32,64,128,256,32),
                dropout=0,
                upsample="deconv"
            )
        elif backbone == "vnet":
            self.backbone = monai.networks.nets.VNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=organ_embedding.shape[0],
            )
        elif backbone == "STUNet_small":
            model = STUNet(in_channels, 105, depth=[1,1,1,1,1,1], dims=[16, 32, 64, 128, 256, 256],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
            model.load_state_dict(torch.load("./STUNet/small_ep4k.model")["state_dict"])
            model.seg_outputs[0] = nn.Conv3d(256,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[1] = nn.Conv3d(128,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[2] = nn.Conv3d(64,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[3] = nn.Conv3d(32,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[4] = nn.Conv3d(16,organ_embedding.shape[0],kernel_size=1,stride=1)
            self.backbone = model
        elif backbone == "STUNet_base":
            model = STUNet(1, 105, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
            model.load_state_dict(torch.load("./models/STUNet/base_ep4k.model")["state_dict"])
            model.seg_outputs[0] = nn.Conv3d(512,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[1] = nn.Conv3d(256,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[2] = nn.Conv3d(128,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[3] = nn.Conv3d(64,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[4] = nn.Conv3d(32,organ_embedding.shape[0],kernel_size=1,stride=1)
            self.backbone = model

        elif backbone == "STUNet_large":
            model = STUNet(1, 105, depth=[2,2,2,2,2,2], dims=[64, 128, 256, 512, 1024, 1024],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
            # model.load_state_dict(torch.load("./models/STUNet/large_ep4k.model")["state_dict"])
            model.seg_outputs[0] = nn.Conv3d(1024,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[1] = nn.Conv3d(512,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[2] = nn.Conv3d(256,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[3] = nn.Conv3d(128,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[4] = nn.Conv3d(64,organ_embedding.shape[0],kernel_size=1,stride=1)
            self.backbone = model
            
        elif backbone == "STUNet_huge":
            model = STUNet(1,105, depth=[3,3,3,3,3,3], dims=[96, 192, 384, 768, 1536, 1536],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
            model.load_state_dict(torch.load("./models/STUNet/huge_ep4k.model")["state_dict"])
            model.seg_outputs[0] = nn.Conv3d(1536,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[1] = nn.Conv3d(768,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[2] = nn.Conv3d(384,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[3] = nn.Conv3d(192,organ_embedding.shape[0],kernel_size=1,stride=1)
            model.seg_outputs[4] = nn.Conv3d(96,organ_embedding.shape[0],kernel_size=1,stride=1)
            self.backbone = model
            
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.register_buffer("organ_embedding",organ_embedding)
        mat = torch.matmul(organ_embedding,organ_embedding.T)
        self.register_buffer("prompt_graph",mat)
        
        #self.organ_embedding = organ_embedding
        self.text_encoder = nn.Sequential(
            nn.Linear(512+512,256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,organ_embedding.shape[0])
        )
        self.segmentors = nn.Conv3d(organ_embedding.shape[0] * 2,organ_embedding.shape[0],kernel_size=1,bias=True)
        self.gap = nn.AdaptiveAvgPool3d(output_size=(8,8,8))
    def forward(self, x_in):
        x_in = self.backbone(x_in)
        x_feat = self.gap(x_in)
        b = x_in.shape[0]
        x_feat = x_feat.view(b,self.organ_embedding.shape[0],-1)
        x_feat = torch.mean(x_feat,dim=0)
        weight = self.text_encoder(
            torch.cat([x_feat,self.organ_embedding],dim=1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_out = torch.nn.functional.conv3d(x_in,weight)
        x_out = x_out.permute(0,2,3,4,1)
        x_out = torch.matmul(x_out,self.prompt_graph).permute(0,4,1,2,3)
        x = torch.concat([x_out,x_in],dim=1)
        return self.segmentors(x)