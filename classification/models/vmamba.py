import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):#线性层与卷积层的混合实现
    '''
    继承自nn.Linear，但在前向传播时使用F.conv2d实现
    前向传播：将 2D 输入（B,C,H,W）视为多个独立的线性变换，通过 1x1 卷积实现
        权重形状：(out_channels, in_channels) → 重塑为 (out_channels, in_channels, 1, 1)
        等价于对每个空间位置应用相同的线性变换
    权重加载：允许从线性层或卷积层加载预训练权重，自动调整形状
    '''
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    '''
    针对 2D 数据的层归一化
    继承自nn.LayerNorm，但处理 4D 张量（图像）
    通道归一化：
        将输入从(B,C,H,W)重排为(B,H,W,C)
        应用标准层归一化（对最后一维 C 进行归一化）
        恢复原始维度顺序
    应用场景：在通道优先（NCHW）的 CNN 架构中使用层归一化
    '''
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    '''
    用于下采样的 patch 合并操作
    输入特征图下采样：将 2x2 相邻 patch 合并，通道数增加，空间尺寸减半
    初始化参数：
        dim：输入维度
        out_dim：输出维度（默认 2*dim）
        channel_first：控制通道顺序
    核心方法：
        _patch_merging_pad_channel_last：处理通道在后的情况（B,H,W,C）
        _patch_merging_pad_channel_first：处理通道在前的情况（B,C,H,W）
    操作流程：
        对输入进行填充（如果尺寸为奇数）
        将 2x2 相邻 patch 拆分为 4 个张量
        在通道维度拼接，通道数变为 4 倍
        应用归一化和线性变换（降维）
    '''
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)#如果out_dim<0（即用户未指定明确的输出维度，或使用默认值），则输出维度设为2*dim。否则，输出维度使用用户指定的out_dim。
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2 #0::2 从索引0开始，每隔2个元素选取一个元素        #0::1 从索引1开始，每隔2个元素选取一个元素
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    '''
    门控多层感知机
    门控机制：
        将输入通过第一个线性层扩展为两倍维度
        将结果分为两部分 x 和 z
        对 z 应用激活函数后，与 x 逐元素相乘作为门控操作
        门控结果通过第二个线性层输出
    通道顺序支持：
        channel_first=True 时，输入格式为 (B,C,H,W)，沿维度 1 分割
        channel_first=False 时，输入格式为 (B,H,W,C)，沿维度 - 1 分割
    应用场景：
        相比标准 MLP，gMLP 通过门控机制增强特征选择能力
        在视觉任务中，可替代自注意力机制，降低计算复杂度
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)# 第一个线性层，输出维度翻倍
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)# 第二个线性层
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x) # 线性变换，输出维度为2*hidden_features
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))# 沿通道维度将张量分为两部分
        x = self.fc2(x * self.act(z))# 门控机制：x与激活后的z逐元素相乘，再通过第二个线性层
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    '''
    空间维度 Softmax
    空间维度归一化：
        将输入张量的空间维度（H 和 W）展平
        对展平后的空间维度应用标准 Softmax
        恢复原始张量形状
    通道顺序支持：
        dim=-1：处理通道在后的输入 (B,H,W,C)，对最后一维（空间维度）归一化
        dim=1：处理通道在前的输入 (B,C,H,W)，对第二维（空间维度）归一化
    数学意义：
        对每个通道或每个位置，计算其在空间上的概率分布
        常用于注意力机制中，生成空间注意力图
    '''
    def forward(self, x: torch.Tensor):
        if self.dim == -1: # 通道在后的情况 (B,H,W,C)
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)# 对空间维度(H*W)应用softmax
        elif self.dim == 1:# 通道在前的情况 (B,C,H,W)
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)# 对空间维度(H*W)应用softmax
        else:
            raise NotImplementedError# 仅支持dim=1或dim=-1


# =====================================================
class mamba_init:#mamba_init的静态工具类，用于初始化 Mamba 模型中的关键参数，包括时间步长 (dt)、状态转移矩阵 (A) 和跳过连接参数 (D)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):#时间步长参数初始化
        '''
        初始化时间步长参数，控制序列建模中的状态更新速率
        通过dt_proj线性层将低秩表示映射到完整维度
        偏置初始化确保softplus(bias)在合理范围内（避免梯度消失 / 爆炸）

        dt的初始化：
            首先做一个可学习d_rank->d_inner的linear映射。
            初始化权重，要控制weight的初始化范围，使得初始化时，softplus(dt_proj(...))后方差不会爆炸。
            初始化偏差，delta = F.softplus(bias)确保delta>0，但不能让他太大，因为x>0, softplus(x)持续递增，所以希望初始化时 Δt 落在合理范围内（如 0.001～0.1），那么就要对bias进行逆softplus变化，使其落在0.001～0.1之间。
            （softplus：y=log(1+exp(x))   逆softplus：y=log(exp(x)-1)   但是官方给的是y=x+log(exp(x)-1)，原因是避免exp(x)-1趋近于0，导致y溢出，算是增加了一些数值稳定性的操作。
        '''
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)# 创建线性投影层  首先做一个可学习d_rank->d_inner的linear映射。  d_inner = int(ssm_ratio * d_model)
        # 初始化权重以保持方差  #初始化权重，要控制weight的初始化范围，使得初始化时，softplus(dt_proj(...))后方差不会爆炸。
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale# 缩放因子
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)# 常量初始化
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std) # 均匀分布初始化
        else:
            raise NotImplementedError
        #初始化偏差，delta = F.softplus(bias)确保delta>0，但不能让他太大，因为x>0, softplus(x)持续递增，
        #所以希望初始化时delta落在合理范围内（如 0.001～0.1），对delta进行逆softplus变化，就得到bias。
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max # 初始化偏置，使得softplus(bias)在[dt_min, dt_max]范围内
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)# 指数分布随机值
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))# softplus的逆变换
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)# 设置偏置值
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        '''
        状态转移矩阵初始化:
            初始化状态转移矩阵的对数形式 (log(A))
            使用状态索引的对数作为初始值，有助于控制衰减率
            支持分组操作，可创建多组不同的转移矩阵

        Mamba中用hippo矩阵来初始化A，A的对角线是1,2,...n，是上三角矩阵，
        A初始化时可让每一行都为从1开始递增序列。
        但是实际上更新的是A_log，当Aij>1时，A_logij>0，真正作用于系统矩阵A'的是对A_log取负指数，那么A'<0。
        A' < 0 保证系统状态不会发散，最终系统趋于稳定
        '''
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()# S4D实值初始化方法
        A_log = torch.log(A)  # Keep A_log in fp32  # 对状态索引取对数
        if copies > 0:# 复制多份用于分组
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)# 合并维度
        A_log = nn.Parameter(A_log) # 转换为可训练参数
        A_log._no_weight_decay = True# 标记为不应用权重衰减
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        '''
        跳过连接参数初始化:
            初始化跳过连接参数 D，控制输入直接传递到输出的比例
            初始化为 1，允许模型在训练初期保持较强的信息流
            同样支持分组操作
        '''
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)# D "skip"参数初始化 # 初始化为全1
        if copies > 0:# 复制多份用于分组
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1) # 合并维度
        D = nn.Parameter(D)  # Keep in fp32 # 转换为可训练参数
        D._no_weight_decay = True # 标记为不应用权重衰减
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):# 组合初始化dt、A、D参数
        '''
        组合初始化方法:
            一次性初始化 Mamba 模型所需的三组关键参数
            支持分组初始化，每组参数可学习不同的动态特性
            返回的参数直接用于构建 Mamba 模型的状态空间模型组件
        '''
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ] # 初始化dt投影层
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================# 初始化A和D参数
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class SS2Dv0:#基于状态空间模型的 2D 特征处理模块，结合了卷积操作和选择性扫描机制，能够高效处理空间特征
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,# 输入特征维度
        d_state=16,# 状态空间维度
        ssm_ratio=2.0,# 扩展比例
        dt_rank="auto",# 时间步长矩阵的秩
        # ======================
        dropout=0.0,# 正则化参数
        # ======================
        seq=False, # 是否处理序列数据
        force_fp32=True,# 是否强制使用FP32精度
        **kwargs,
    ):
        if "channel_first" in kwargs:# 确保不使用channel_first模式（要求channel_last）
            assert not kwargs["channel_first"]
        # 配置激活函数和参数
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # 计算内部维度和自动确定dt_rank
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # 设置前向传播函数，根据参数选择不同版本
        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================# 输入投影层：将输入特征映射到更高维度
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner, # 深度可分离卷积
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ) # 2D卷积层：处理空间特征

        # x proj ============================# X投影层：将特征投影到状态空间参数
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)  #将多个线性层（self.x_proj中的每个t）的权重矩阵提取出来，然后沿着新的维度（dim=0）堆叠成一个大张量，最后将其注册为模型的可训练参数（nn.Parameter）
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )# 初始化SSM的核心参数：时间步长(dt)、状态转移矩阵(A)和跳过连接(D)

        # out proj ======================================= # 输出投影层：将特征映射回原始维度
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        # 输入投影和特征分离
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)# 将输入分为x和z两部分
        z = self.act(z)# 对z应用激活函数
        # 对x进行卷积处理
        x = x.permute(0, 3, 1, 2).contiguous()# 调整维度为(B, C, H, W)
        x = self.conv2d(x) # (b, d, h, w) # 应用2D卷积
        x = self.act(x)# 应用激活函数
        # 使用Mamba后端的选择性扫描函数
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, D, H, W = x.shape# 获取张量形状信息
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W# 序列长度
        # 准备多个方向的输入（水平、垂直及其反转）
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # 添加反转方向的输入，形成4个方向
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        # 计算SSM参数：时间步长(dts)、输入映射(Bs)和输出映射(Cs)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        # 重塑张量以适应选择性扫描函数
        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        # 准备SSM核心参数
        As = -self.A_logs.float().exp() # (k * d, d_state) # 状态转移矩阵
        Ds = self.Ds.float() # (k * d)# 跳过连接参数
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)# 时间步长偏置
        # 强制使用FP32精度（如果需要）
        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        # 应用选择性扫描机制处理序列
        if seq:
            # 序列模式：分别处理每个方向
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:# 非序列模式：一次性处理所有方向
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # 合并不同方向的处理结果
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y # 融合4个方向的信息
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C) # 调整维度并应用归一化
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z # 应用门控机制并输出
        out = self.dropout(self.out_proj(y))# 最终投影和dropout
        return out


# support: v01-v05; v051d,v052d,v052dc; 
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    '''
    支持多版本的 2D 状态空间模型模块:
        该类实现了一个功能更丰富的 SS2D 模块（SS2Dv2），支持多种前向传播版本（v01-v05、v2、v3 等）和后缀配置（如激活函数、卷积类型），适用于不同场景的视觉特征处理。
    '''
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,# 输入特征维度
        d_state=16,# 状态空间维度
        ssm_ratio=2.0, # 特征扩展比例，d_inner = d_model * ssm_ratio
        dt_rank="auto", # 时间步长矩阵秩，auto时自动计算为d_model/16向上取整
        act_layer=nn.SiLU,
        # dwconv ===============# 深度卷积参数
        d_conv=3, # < 2 means no conv  # 卷积核大小，<=1时禁用卷积
        conv_bias=True, # 卷积层是否带偏置
        # ====================== # 正则化参数
        dropout=0.0,# Dropout率
        bias=False, # 线性层是否带偏置
        # dt init ==============# 时间步长（dt）初始化参数
        dt_min=0.001,# dt的最小值
        dt_max=0.1,# dt的最大值
        dt_init="random",# 初始化方式（random/constant）
        dt_scale=1.0,# 初始化缩放因子
        dt_init_floor=1e-4,# 初始化下界
        initialize="v0",# 初始化版本（v0/v1/v2）
        # ======================# 前向传播配置
        forward_type="v2",# 前向传播版本（如v05、v2等）
        channel_first=False,# 是否使用通道优先格式（NCHW）
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4 # 分组数，用于多组参数学习
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1# 是否启用深度卷积
        Linear = Linear2d if channel_first else nn.Linear# 根据channel_first选择线性层类型（Linear2d或nn.Linear）
        self.forward = self.forwardv2

        # tags for forward_type ==============================# 解析前向传播类型后缀（如_no32、_noz等）
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug ======================================= # 定义支持的前向传播版本映射
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"   # oflex backend，cross2d模式
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"  # 禁用einsum，提升吞吐量
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),  # 单向扫描
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),  # 双向扫描
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"), # 级联2D扫描
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None) # 根据类型选择核心前向函数

        # in proj ======================================= # 输入投影层：若不禁用z，则输出2*d_inner维度（用于拆分出门控信号z）
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================  # 深度卷积层（可选）
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================   # X投影层：将特征映射到SSM参数空间（dt_rank + 2*d_state）
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================  # 输出模块
        self.out_act = nn.GELU() if self.oact else nn.Identity()# 输出激活函数（可选）
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias) # 输出投影层
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()# Dropout层

        if initialize in ["v0"]:# 参数初始化版本选择
            # 使用mamba_init工具类初始化核心参数（A_logs, Ds, dt_projs）
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:# 简单初始化（用于测试或特定场景）
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:# 简单初始化（用于测试或特定场景）
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32 # 是否强制使用FP32精度
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input  # oflex后端专用参数，控制输出精度
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput  # 禁用einsum，改用卷积/线性层提升性能
        # ==============================
        selective_scan_backend = None,  # 扫描后端（mamba/oflex/torch）
        # ==============================
        scan_mode = "cross2d",  # 扫描模式（cross2d/unidi/bidi等）
        scan_force_torch = False,  # 强制使用PyTorch原生扫描
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug  # 解析扫描模式
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)   # FP32转换函数
        # 获取输入形状和参数形状
        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W  # 空间像素总数

        # 选择性扫描函数（根据后端动态选择）
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == -1:# cascade2d模式（级联扫描，先行后列）
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)# 拆分为4组参数
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            # 先行扫描
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), # 取前两组参数
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True, # 水平方向扫描
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            # 后列扫描
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:# 其他扫描模式（如cross2d、unidi、bidi）
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)##对x进行跨行扫描操作，将结果存储在xs中。注意xs与x只是扫描顺序不同，并没有改变x的值
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), #输入张量，形状依次为[批次大小，输入通道数，序列长度]
                                 self.x_proj_weight.view(-1, D, 1), #原始卷积核权重，形状依次为[输出通道数, 分组卷积参数, 一维卷积核长度]
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), #偏置项 将偏置张量展平为一维向量，形状为 [K*D]
                                 groups=K#分组卷积数     需要满足 输入通道数 / 分组卷积数 = 分组卷积参数
                                 )#x_dbl是对xs进行conv1d卷积，得到dts, Bs, Cs
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
            # 准备SSM核心参数
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state) # 状态转移矩阵（负数，确保衰减）
            Ds = self.Ds.to(torch.float) # (K * c)# 跳过连接参数
            Bs = Bs.contiguous().view(B, K, N, L) # 输入映射
            Cs = Cs.contiguous().view(B, K, N, L)# 输出映射
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)# 时间步长偏置

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs) # 强制转换为FP32
            # 应用选择性扫描
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))
        # 维度调整和归一化
        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)# 转为NHWC格式
        y = out_norm(y)  # 应用输出归一化（或其他后置操作，如Softmax）

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)# 输入投影
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)# 核心SSM处理
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z# 门控机制
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        '''
        输出归一化配置
        '''
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        # 解析后缀以确定输出归一化类型
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(# 3x3深度卷积
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):# 检查并移除前向传播类型中的后缀
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


# support: xv1a,xv2a,xv3a; 
# postfix: _cpos;_ocov;_ocov2;_ca,_ca1;_act;_mul;_onsigmoid,_onsoftmax,_ondwconv3,_onnone;
class SS2Dv3:
    def __initxv__(
        self,
        # basic dims ===========
        d_model=96, # 输入特征维度
        d_state=16,# 状态空间维度
        ssm_ratio=2.0,# 特征扩展比例，d_inner = d_model * ssm_ratio
        dt_rank="auto",# 时间步长矩阵秩，auto时自动计算为d_model/16向上取整
        # dwconv ===============
        d_conv=3, # < 2 means no conv   # 卷积核大小，<=1时禁用卷积
        conv_bias=True,# 卷积层是否带偏置
        # ======================
        dropout=0.0, # Dropout率
        bias=False, # 线性层是否带偏置
        # dt init ==============
        dt_min=0.001, # 时间步长最小值
        dt_max=0.1,# 时间步长最大值
        dt_init="random",# 初始化方式（random/constant）
        dt_scale=1.0,# 初始化缩放因子
        dt_init_floor=1e-4,# 初始化下界
        initialize="v0", # 初始化版本（v0/v1/v2）
        # ======================
        forward_type="v2",# 前向传播类型（如xv1a、xv2a等）
        channel_first=False,# 是否使用通道优先格式（NCHW）
        # ======================
        **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        k_group = 4 # 分组数，用于多组参数学习
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, channel_first)
        self.omul, forward_type = checkpostfix("_mul", forward_type)
        self.oact, forward_type = checkpostfix("_act", forward_type)
        self.f_omul = nn.Identity() if self.omul else None
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        mode = forward_type[:4]
        assert mode in ["xv1a", "xv2a", "xv3a"]

        self.forward = partial(self.forwardxv, mode=mode)
        self.dts_dim = dict(xv1a=self.dt_rank, xv2a=self.d_inner, xv3a=4 * self.dt_rank)[mode]
        d_inner_all = d_inner + self.dts_dim + 8 * d_state
        self.in_proj = Linear(d_model, d_inner_all, bias=bias)
        
        # conv =======================================
        self.cpos = False
        self.iconv = False
        self.oconv = False
        self.oconv2 = False
        if self.with_dconv:
            cact, forward_type = checkpostfix("_ca", forward_type)
            cact1, forward_type = checkpostfix("_ca1", forward_type)
            self.cact = nn.SiLU() if cact else nn.Identity()
            self.cact = nn.GELU() if cact1 else self.cact
                
            self.oconv2, forward_type = checkpostfix("_ocov2", forward_type)
            self.oconv, forward_type = checkpostfix("_ocov", forward_type)
            self.cpos, forward_type = checkpostfix("_cpos", forward_type)
            self.iconv = (not self.oconv) and (not self.oconv2)

            if self.iconv:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv:
                self.oconv2d = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv2:
                self.conv2d = nn.Conv2d(
                    in_channels=d_inner_all,
                    out_channels=d_inner_all,
                    groups=d_inner_all,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))


        if forward_type.startswith("xv2"):
            del self.dt_projs_weight
            self.dt_projs_weight = None

    def forwardxv(self, x: torch.Tensor, **kwargs):
        B, (H, W) = x.shape[0], (x.shape[2:4] if self.channel_first else x.shape[1:3])
        L = H * W
        force_fp32 = False
        delta_softplus = True
        out_norm = self.out_norm
        to_dtype = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex=True, backend=None)

        if self.iconv:
            x = self.cact(self.conv2d(x)) # (b, d, h, w)
        elif self.cpos:
            x = x + self.conv2d(x) # (b, d, h, w)

        x = self.in_proj(x)
        
        if self.oconv2:
            x = self.conv2d(x) # (b, d, h, w)

        us, dts, Bs, Cs = x.split([self.d_inner, self.dts_dim, 4 * self.d_state, 4 * self.d_state], dim=(1 if self.channel_first else -1))

        _us = us
        # Bs, Cs = Bs.view(B, H, W, 4, -1), Cs.view(B, H, W, 4, -1)
        # Bs, Cs = Bs.view(B, 4, -1, H, W), Cs.view(B, 4, -1, H, W)
        us = cross_scan_fn(us.contiguous(), in_channel_first=self.channel_first, out_channel_first=True).view(B, -1, L)
        Bs = cross_scan_fn(Bs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=True).view(B, 4, -1, L)
        Cs = cross_scan_fn(Cs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=True).view(B, 4, -1, L)
        dts = cross_scan_fn(dts.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=(self.dts_dim == 4 * self.dt_rank)).view(B, L, -1)
        if self.dts_dim == self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
        elif self.dts_dim == 4 * self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, 4, -1, H, W)
        y: torch.Tensor = cross_merge_fn(ys.contiguous(), in_channel_first=self.channel_first, out_channel_first=True)
        y = y.view(B, -1, H, W) if self.channel_first else y.view(B, H, W, -1)
        y = out_norm(y)
        
        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        y = (y.to(x.dtype) if to_dtype else y)
        
        y = self.out_act(y)
        
        if self.omul:
            y = y * _us

        if self.oconv:
            y = y + self.cact(self.oconv2d(_us))

        out = self.dropout(self.out_proj(y))
        return out


# mamba2 support ================================
class SS2Dm0:
    def __initm0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16, # now with mamba2, dstate should be bigger...
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        # ======================
        forward_type="m0",
        # ======================
        with_initial_state=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        self.with_dconv = d_conv > 1
        Linear = nn.Linear
        self.forward = self.forwardm0

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            ) 
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

    def forward_corem0(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        chunk_size = 64,
        dstate = 64,        
        # ==============================
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias, 
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,
        )
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias, 
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype)

    def forwardm0(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0):
    '''
    多版本状态空间模型的统一接口
        这个类实现了一个名为 "SS2D" 的模块，它通过多重继承整合了多个版本的 SS2D 实现（SS2Dv0、SS2Dv2、SS2Dv3、SS2Dm0），并根据forward_type参数选择不同的初始化方法，从而支持多种前向传播策略和配置选项。
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):# "VSSBlock" 的模块，它结合了状态空间模型 (SSM) 和多层感知机 (MLP) 来处理视觉特征。该模块支持灵活的配置选项，可以根据需求调整 SSM 和 MLP 的使用比例、归一化方式等。
    def __init__(
        self,
        hidden_dim: int = 0, # 隐藏层维度
        drop_path: float = 0, # Drop Path概率
        norm_layer: nn.Module = nn.LayerNorm, # 归一化层类型
        channel_first=False, # 是否使用通道优先的数据格式
        # ============================= # SSM参数
        ssm_d_state: int = 16,# SSM状态维度
        ssm_ratio=2.0, # SSM分支的扩展比例
        ssm_dt_rank: Any = "auto",# SSM时间步长矩阵的秩
        ssm_act_layer=nn.SiLU, # SSM激活函数
        ssm_conv: int = 3, # SSM卷积核大小
        ssm_conv_bias=True,# SSM卷积层是否使用偏置
        ssm_drop_rate: float = 0, # SSM分支的dropout率
        ssm_init="v0", # SSM初始化方法
        forward_type="v2", # SSM前向传播类型
        # =============================# MLP参数
        mlp_ratio=4.0, # MLP分支的扩展比例
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,# MLP分支的dropout率
        gmlp=False,# 是否使用门控MLP
        # =============================
        use_checkpoint: bool = False,# 是否使用梯度检查点
        post_norm: bool = False,# 是否使用后置归一化
        # =============================
        _SS2D: type = SS2D,# SSM模块类型
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0 # 是否启用SSM分支
        self.mlp_branch = mlp_ratio > 0 # 是否启用MLP分支
        self.use_checkpoint = use_checkpoint # 是否使用梯度检查点
        self.post_norm = post_norm # 是否使用后置归一化

        if self.ssm_branch: # 初始化SSM分支
            self.norm = norm_layer(hidden_dim) # 归一化层
            self.op = _SS2D(
                d_model=hidden_dim, # 输入特征维度
                d_state=ssm_d_state,  # SSM状态维度
                ssm_ratio=ssm_ratio,  # 扩展比例
                dt_rank=ssm_dt_rank, # 时间步长矩阵的秩
                act_layer=ssm_act_layer,# 激活函数
                # ==========================
                d_conv=ssm_conv, # 卷积核大小
                conv_bias=ssm_conv_bias,# 是否使用卷积偏置
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,# 通道优先标志
            )
        # Drop Path正则化
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch: # 初始化MLP分支
            _MLP = Mlp if not gmlp else gMlp # 根据标志选择标准MLP或门控MLP
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)# 计算MLP隐藏层维度
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:  # 处理SSM分支
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x))) # 后置归一化：先应用SSM操作，再归一化并添加残差连接
            else:
                x = x + self.drop_path(self.op(self.norm(x)))# 前置归一化：先归一化，再应用SSM操作并添加残差连接
        if self.mlp_branch:# 处理MLP分支
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN   # 后置归一化：先应用MLP操作，再归一化并添加残差连接
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN# 前置归一化：先归一化，再应用MLP操作并添加残差连接
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self,
        patch_size=4,  # 图像patch大小
        in_chans=3,  # 输入通道数（RGB=3）
        num_classes=1000,  # 分类类别数
        depths=[2, 2, 9, 2],  # 每个阶段的层数
        dims=[96, 192, 384, 768],  # 每个阶段的特征维度
        # SSM相关参数
        ssm_d_state=16,  # SSM状态维度
        ssm_ratio=2.0,  # SSM扩张比
        ssm_dt_rank="auto",  # SSM时间步长矩阵秩
        ssm_act_layer="silu",  # SSM激活函数
        ssm_conv=3,  # SSM卷积核大小
        ssm_conv_bias=True,  # SSM卷积是否使用偏置
        ssm_drop_rate=0.0,  # SSM dropout率
        ssm_init="v0",  # SSM初始化方式
        forward_type="v2",  # 前向传播版本
        # MLP相关参数
        mlp_ratio=4.0,  # MLP隐藏层扩张比
        mlp_act_layer="gelu",  # MLP激活函数
        mlp_drop_rate=0.0,  # MLP dropout率
        gmlp=False,  # 是否使用gMLP
        # 其他参数
        drop_path_rate=0.1,  # 随机深度丢弃率
        patch_norm=True,  # patch嵌入后是否使用归一化
        norm_layer="LN",  # 归一化层类型（LN=LayerNorm）
        downsample_version: str = "v2",  # 下采样版本
        patchembed_version: str = "v1",  # patch嵌入版本
        use_checkpoint=False,  # 是否使用检查点以节省内存
        posembed=False,  # 是否使用位置编码

        imgsize=224,  # 输入图像大小
        _SS2D=SS2D,  # SSM二维实现类
        # =========================
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"]) # 判断归一化层是否为通道优先（channel-first）格式（如 BN/LN2D）
        self.num_classes = num_classes# 保存分类类别数
        self.num_layers = len(depths)
        if isinstance(dims, int):# 处理 dims 参数：若为整数，则自动生成逐层翻倍的维度列表
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1] # 最后一个阶段的特征维度（用于分类头）
        self.dims = dims# 保存各阶段特征维度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule #随机深度正则化，计算得到的是每一层的残差块的丢弃概率
        # 这两段代码定义了两种模块字典，分别用于归一化层（_NORMLAYERS）和激活函数（_ACTLAYERS）的动态选择。这种设计使模型能够根据配置灵活切换不同的层类型，增强了代码的可扩展性和复用性。
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )# 定义归一化层映射字典（支持 LN、LN2D、BN）

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        ) # 定义激活函数映射字典（支持 silu、gelu、relu 等）
        # 根据字符串名称获取实际的归一化层和激活函数类
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        # 初始化位置编码（若启用 posembed）
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)# 根据版本选择 patch 嵌入模块（v1/v2 不同实现）
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),# 无下采样时返回恒等映射
        ).get(downsample_version, None) # 根据版本选择下采样模块（v1/v2/v3 或无下采样）

        self.layers = nn.ModuleList()# 初始化各阶段的层（layers）
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()# 计算当前阶段是否需要下采样（非最后一个阶段时添加下采样模块）

            self.layers.append(self._make_layer(#创建当前阶段的层（包含多个VSSBlock和下采样模块）
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],#当前阶段的随机深度率
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================# 传递 SSM 相关参数
                ssm_d_state=ssm_d_state, # SSM状态维度
                ssm_ratio=ssm_ratio,# 扩展比例
                ssm_dt_rank=ssm_dt_rank,# 时间步长矩阵的秩
                ssm_act_layer=ssm_act_layer,# 激活函数
                ssm_conv=ssm_conv, # 卷积核大小
                ssm_conv_bias=ssm_conv_bias,# 是否使用卷积偏置
                ssm_drop_rate=ssm_drop_rate,# dropout率
                ssm_init=ssm_init,# 初始化方法
                forward_type=forward_type,# 前向传播类型
                # =================# 传递 MLP 相关参数
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # ================= # 传递 SSM 二维实现类
                _SS2D=_SS2D,
            ))
        # 定义分类头（Classifier）：归一化+池化+线性层
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C   #对最后阶段的特征归一化
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),# 调整通道顺序
            avgpool=nn.AdaptiveAvgPool2d(1),# 全局平均池化
            flatten=nn.Flatten(1),# 展平特征
            head=nn.Linear(self.num_features, num_classes),# 分类头
        ))

        self.apply(self._init_weights)# 应用权重初始化函数

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):#（位置编码初始化）
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size) # 计算 patch 网格的行数和列数（如 img_size=224, patch_size=4 → 56x56）
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))# 创建可学习的位置编码参数（形状：1 x embed_dims x H x W）
        trunc_normal_(pos_embed, std=0.02) # 用截断正态分布初始化位置编码（标准差 0.02）
        return pos_embed

    def _init_weights(self, m: nn.Module):#权重初始化
        if isinstance(m, nn.Linear):# 线性层权重初始化（截断正态分布）
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: # 偏置初始化为 0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):# 归一化层权重初始化（全 1）和偏置初始化（全 0）
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first # 构建 patch 嵌入模块：卷积层 + 归一化（可选）+ 通道顺序调整
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True), # 分块卷积
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)), # 转换为 (B, H, W, C) 格式
            (norm_layer(embed_dim) if patch_norm else nn.Identity()), # 归一化
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2 # 步长为 patch_size 的一半（重叠分块）
        kernel_size = stride + 1# 卷积核大小（如 stride=2 → kernel=3，带重叠）
        padding = 1 # 填充保持空间尺寸
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding), # 第一层卷积：降维到 embed_dim/2
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)), # 通道顺序调整（根据是否 channel-first 和是否归一化）
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),# 归一化
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),# 恢复通道顺序
            nn.GELU(), # 激活函数
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),# 第二层卷积：升维到 embed_dim
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),# 转换为 (B, H, W, C)
            (norm_layer(embed_dim) if patch_norm else nn.Identity()), # 归一化
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first# 下采样模块：卷积层（stride=2）+ 归一化 + 通道顺序调整
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)), # 转为 (B, C, H, W) 格式
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),# 减半空间尺寸，翻倍通道数
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),# 转回 (B, H, W, C)
            norm_layer(out_dim), # 归一化
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):#（v3 版本的下采样模块，使用 3x3 卷积）
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1), # 3x3 卷积下采样
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(#创建单个阶段的层，包含多个VSSBlock
        dim=96, # 当前阶段的特征维度
        drop_path=[0.1, 0.1], # 该阶段各层的随机深度率
        use_checkpoint=False, # 是否启用检查点
        norm_layer=nn.LayerNorm,# 归一化层类型
        downsample=nn.Identity(), # 下采样模块（默认恒等映射）
        channel_first=False, # 是否为通道优先格式
        # ===========================#SSM参数
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================== #MLP参数
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================== #SSM二维实现类
        _SS2D=SS2D,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path) # 该阶段包含的层数（如 depths 中的某个值）
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(# 创建每个层的 VSSBlock（核心模块，包含 SSM 和 MLP）
                hidden_dim=dim, 
                drop_path=drop_path[d], # 当前层的随机深度率
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        # 返回该阶段的层（包含所有 blocks 和下采样模块）
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),# 多个 block 组成的序列
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)# 将输入图像分割为patch并嵌入到特征空间
        if self.pos_embed is not None:# 如果使用位置编码
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed# 根据通道顺序调整位置编码的维度排列
            x = x + pos_embed# 将位置编码添加到patch嵌入中
        for layer in self.layers:# 依次通过各个阶段的层
            x = layer(x) # 每个层包含多个VSSBlock和可能的下采样
        x = self.classifier(x)# 通过分类头得到最终预测结果
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):#使用fvcore库计算模型的浮点运算量
        # shape = self.__input_shape__[1:]# 定义支持的操作及其FLOPs计算函数
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS# 忽略这些操作的FLOPs计算
            "aten::neg": None, # as relu is in _IGNORED_OPS# 通常这些操作的计算量较小
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)# 复制模型避免修改原始模型
        model.cuda().eval()# 将模型移至GPU并设置为评估模式

        input = torch.randn((1, *shape), device=next(model.parameters()).device)# 创建随机输入张量
        params = parameter_count(model)[""]# 计算模型参数数量
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)# 使用fvcore库的flop_count计算FLOPs

        del model, input
        return sum(Gflops.values()) * 1e9# 返回总FLOPs (以Giga为单位)
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code # 用于从旧版本训练代码加载检查点
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """自定义权重加载逻辑，处理不同版本模型之间的兼容性"""
        def check_name(src, state_dict: dict = state_dict, strict=False):# 检查键是否存在于state_dict中
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):# 重命名state_dict中的键以匹配当前模型结构
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        # 处理位置编码：如果大小不匹配，则进行插值调整
        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            # 使用双三次插值调整位置编码的大小以匹配当前模型
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)
        # 重命名patch_embed模块中的键
        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):# 重命名layers模块中的键，处理旧版本到新版本的兼容性
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm") # 重命名分类头中的键
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)# 调用父类的方法完成权重加载


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        # 将norm_layer参数传递给父类
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        # 确定通道顺序（channel_first）
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )# 定义归一化层映射字典
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices# 保存需要输出的特征层索引
        for i in out_indices: # 为每个需要输出的特征层添加归一化层
            layer = norm_layer(self.dims[i]) # 创建归一化层，维度为对应层的特征维度
            layer_name = f'outnorm{i}' # 归一化层的名称，如outnorm0, outnorm1等
            self.add_module(layer_name, layer)# 将归一化层添加到模块中

        del self.classifier # 删除原始VSSM模型中的分类头，因为骨干网络不需要分类头
        self.load_pretrained(pretrained) # 加载预训练权重

    def load_pretrained(self, ckpt=None, key="model"):
        """
        加载预训练权重
        参数:
            ckpt: 预训练模型路径
            key: 权重字典中的键名（通常为"model"）
        """
        if ckpt is None:# 如果没有提供预训练模型路径，则不加载
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu")) # 加载预训练权重并映射到CPU
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)# 加载权重，strict=False表示允许部分加载（忽略不匹配的层）
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):# 定义辅助函数，用于逐层处理并获取中间输出
            x = l.blocks(x)# 通过当前层的所有blocks
            y = l.downsample(x)# 通过下采样层（如果有）
            return x, y # 返回处理后的特征和下采样后的特征

        x = self.patch_embed(x)# 图像patch嵌入
        outs = []# 存储输出特征的列表
        # 逐层处理
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)# o是当前层的输出，x是下采样后的特征
            if i in self.out_indices:# 如果当前层索引在out_indices中，则添加到输出列表
                norm_layer = getattr(self, f'outnorm{i}')# 获取对应的归一化层
                out = norm_layer(o) # 应用归一化
                if not self.channel_first: # 如果不是通道优先格式，则调整维度顺序为(B, C, H, W)
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())# 添加到输出列表
        # 如果没有指定输出索引，则返回最后一层的特征
        if len(self.out_indices) == 0:
            return x
        
        return outs# 返回多尺度特征列表


# =====================================================
def vanilla_vmamba_tiny():
    return VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vanilla_vmamba_small():
    return VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vanilla_vmamba_base():
    return VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# =====================================================
def vmamba_tiny_s2l5(channel_first=True):
    return VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# =====================================================
def vmamba_tiny_s1l8(channel_first=True):
    return VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# mamba2 support =====================================================
# FLOPS count do not work now for mamba2!
def vmamba_tiny_m2():
    return VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=128, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


if __name__ == "__main__":
    model_ref = vmamba_tiny_s1l8()

    model = VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    print(parameter_count(model)[""])
    print(model.flops()) # wrong
    model.cuda().train()
    model_ref.cuda().train()

    def bench(model):
        import time
        inp = torch.randn((128, 3, 224, 224)).cuda()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim1 = time.time() - tim

        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim2 = time.time() - tim

        return tim1 / 30, tim2 / 30
    
    print(bench(model_ref))
    print(bench(model))

    breakpoint()


