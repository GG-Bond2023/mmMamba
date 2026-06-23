import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones.swin import SwinTransformer
from segmentation.models.backbones.Swin import SwinTransformer as SwinTransformer_RADial
from .fourier_net import FFT_Net
from .vssm import MM_VSSM
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
# from models.decode_heads.radial_decode_head import RadialDecodeHead
from segmentation.models.decode_heads.radial_decode_head import RadialDecodeHead
NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

@MODELS.register_module()
class Segmentation_Header(RadialDecodeHead):
    def __init__(self, in_planes, **kwargs):
        super(Segmentation_Header, self).__init__(in_channels=in_planes, channels=in_planes, **kwargs)
        self.freespace = nn.Sequential(BasicBlock(in_planes= in_planes, planes = 128),
                                       BasicBlock(in_planes= 128, planes = 64),
                                       nn.Conv2d(64, 1, kernel_size=1)
                                       )
    def forward(self, x):
        Y = F.interpolate(x, (256, 224))
        seg_out = self.freespace(Y)
        return seg_out


@MODELS.register_module()
class Detection_Header(RadialDecodeHead):
    def __init__(self, in_channels, use_bn=True,reg_layer=2, input_angle_size=0, **kwargs):
        super(Detection_Header, self).__init__(in_channels=in_channels, channels=in_channels, **kwargs)
        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')


        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)
        return out

class block(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size):
        super(block,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=0)
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bnorm(x)
        x = self.activation(x)
        return x

@MODELS.register_module()
class ViT_RangeAngle_Decoder(nn.Module):
    def __init__(self):
        super(ViT_RangeAngle_Decoder,self).__init__()
        self.blk1 = block(2,32,2,2)
        self.blk2 = block(32,64,2,2)
        self.blk3 = block(64,256,2,2)
        self.dense = nn.Linear(512,896)
        self.dense_acc = nn.ReLU()

    def forward(self,x):
        x = self.dense(x)
        x = self.dense_acc(x).reshape(-1,2,16,28)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        return x

class Swin_RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(Swin_RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block4 = BasicBlock(48,128)

        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L4  = nn.Conv2d(384*1,224,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 224, kernel_size=1, stride=1,padding=0)

    def forward(self,features):
        T4 = self.L4(features[3]).transpose(1, 3)
        T3 = self.L3(features[2]).transpose(1, 3)
        T2 = self.L2(features[1]).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)

        return out


class vssm_RangeAngle_Decoder(nn.Module):
    def __init__(self, mimo_pre_encoder=False, **kwargs):
        super(vssm_RangeAngle_Decoder, self).__init__()
        dims = kwargs['dims']
        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block4 = BasicBlock(48,128)

        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L4  = nn.Conv2d(dims[3],224,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(dims[2], 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(dims[1], 224, kernel_size=1, stride=1,padding=0)

    def forward(self,features):
        T4 = self.L4(features[3]).transpose(1, 3)
        T3 = self.L3(features[2]).transpose(1, 3)
        T2 = self.L2(features[1]).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)

        return out



from .FPN_FFTRadNet import FPN_BackBone, FPN_RangeAngle_Decoder
@MODELS.register_module()
class FFTRadNet_FPN_backbone(nn.Module):
    def __init__(self, mimo_layer, channels, blocks, use_FFT_Net=False):
        super(FFTRadNet_FPN_backbone, self).__init__()

        self.DFT= FFT_Net() if use_FFT_Net else None
        self.FPN = FPN_BackBone(num_block=blocks, channels=channels, block_expansion=4, mimo_layer=mimo_layer,
                                use_bn=True)
        self.RA_decoder = FPN_RangeAngle_Decoder()

    def forward(self, x):
        if self.DFT is not None:
            x = self.DFT(x)
        features = self.FPN(x)
        RA = self.RA_decoder(features)

        return RA








@MODELS.register_module()
class FFTRadNet_ViT_backbone(nn.Module):
    def __init__(self, patch_size,in_chans,embed_dim, strides, depths,num_heads,drop_rates, norm_cfg):
        super(FFTRadNet_ViT_backbone, self).__init__()
        assert len(strides)==len(depths)
        self.DFT = FFT_Net()
        self.vit = SwinTransformer_RADial(
                 pretrain_img_size=None,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=drop_rates[0],
                 attn_drop_rate=drop_rates[1],
                 drop_path_rate=drop_rates[2],
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False
        )
        # self.vit = SwinTransformer(
        #          pretrain_img_size=None,
        #          patch_size=patch_size,
        #          in_channels=in_chans,
        #          embed_dims=embed_dim,
        #          depths=depths,
        #          strides = strides,
        #          num_heads=num_heads,
        #          window_size=7,
        #          mlp_ratio=4.,
        #          qkv_bias=True,
        #          qk_scale=None,
        #          drop_rate=drop_rates[0],
        #          attn_drop_rate=drop_rates[1],
        #          drop_path_rate=drop_rates[2],
        #          norm_cfg=dict(type='LN'),
        #          use_abs_pos_embed=False,
        #          patch_norm=True,
        #          out_indices=(0, 1, 2, 3),
        #          frozen_stages=-1,
        #          with_cp=False
        # )

        # del self.vit.norm0, self.vit.norm1, self.vit.norm2, self.vit.norm3
        self.RA_decoder = Swin_RangeAngle_Decoder()

        # if(self.detection_head):
        #     self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)
        #
        # if(self.segmentation_head):
        #     self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

    def forward(self,x):

        # out = {'Detection':[],'Segmentation':[]}
        x = self.DFT(x)
        features = self.vit(x)
        RA = self.RA_decoder(features)
        # if(self.detection_head):
            # out['Detection'] = self.detection_header(RA)

        # if(self.segmentation_head):
        #     Y =  F.interpolate(RA, (256, 224))
        #     out['Segmentation'] = self.freespace(Y)

        return RA


from .FPN_FFTRadNet import MIMO_PreEncoder

@MODELS.register_module()
class FFTRadNet_VSSM_backbone(nn.Module):
    def __init__(self, out_indices, pretrained, norm_layer, mimo_pre_encoder=False, **kwargs):
        super(FFTRadNet_VSSM_backbone, self).__init__()
        self.DFT = FFT_Net()

        self.mimo_pre_encoder = mimo_pre_encoder
        if self.mimo_pre_encoder==True:
            self.pre_enc = MIMO_PreEncoder(32, 192,
                                           kernel_size=(1, 12),
                                           dilation=(1, 16),
                                           use_bn=True)

        self.vssm = MM_VSSM(in_chans= 192 if mimo_pre_encoder else 32, out_indices=out_indices, pretrained=pretrained, norm_layer=norm_layer, **kwargs)
        self.RA_decoder = vssm_RangeAngle_Decoder(mimo_pre_encoder, **kwargs)
    def forward(self,x):
        x = self.DFT(x) # Output shape is [batch, 32, 512, 256]
        if self.mimo_pre_encoder:
            x = self.pre_enc(x)
        features = self.vssm(x) #
        RA = self.RA_decoder(features)
        return RA





from classification.models.videomamba import VisionMamba
@MODELS.register_module()
class FFTRadNet_VisionMamba_backbone(nn.Module):
    def __init__(self, mimo_pre_encoder=False, **kwargs):
        super(FFTRadNet_VisionMamba_backbone, self).__init__()
        self.DFT = FFT_Net()

        self.mimo_pre_encoder = mimo_pre_encoder
        if self.mimo_pre_encoder==True:
            self.pre_enc = MIMO_PreEncoder(32, 192,
                                           kernel_size=(1, 12),
                                           dilation=(1, 16),
                                           use_bn=True)
        self.vssm = VisionMamba(
            group_type = kwargs['group_type'],
            # k_group=kwargs['k_group'],
            depth=kwargs['depth'],
            embed_dim= kwargs['embed_dim'],
            dt_rank= kwargs['dt_rank'],  # Rank of the dynamic routing tensor
            d_inner= kwargs['d_inner'],  # 256,  # Inner dimension of the model
            d_state= kwargs['d_state'],  # 256,  # State dimension of the model
            scan_type=kwargs['scan_type'],
            conv3D_channel=kwargs['conv3D_channel'],
            conv3D_kernel=kwargs['conv3D_kernel'],
            )
        self.RA_decoder = vssm_RangeAngle_Decoder(mimo_pre_encoder, **kwargs)
    def forward(self,x):
        x = self.DFT(x) # Output shape is [batch, 32, 512, 256]
        if self.mimo_pre_encoder:
            x = self.pre_enc(x)
        features = self.vssm(x) #
        RA = self.RA_decoder(features)
        return RA
