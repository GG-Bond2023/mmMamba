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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mmMamba import cross_scan_fn, cross_merge_fn



# try:
#     from .csm_triton import cross_scan_fn, cross_merge_fn
# except:
#     from csm_triton import cross_scan_fn, cross_merge_fn

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
class Linear2d(nn.Linear):  # Hybrid implementation of linear layer and convolution
    '''
    Inherits from nn.Linear but uses F.conv2d for forward pass.
    Forward pass: treats 2D input (B,C,H,W) as multiple independent linear transforms via 1x1 convolution.
        Weight shape: (out_channels, in_channels) -> reshaped to (out_channels, in_channels, 1, 1)
        Equivalent to applying the same linear transform to every spatial position.
    Weight loading: allows loading pretrained weights from either linear or conv layers, auto-adjusting shape.
    '''

    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    '''
    Layer normalization for 2D data.
    Inherits from nn.LayerNorm but handles 4D tensors (images).
    Channel normalization:
        Rearranges input from (B,C,H,W) to (B,H,W,C)
        Applies standard layer normalization (normalizing over the last dimension C)
        Restores original dimension order.
    Use case: applying layer normalization in channel-first (NCHW) CNN architectures.
    '''

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    '''
    Patch merging operation for downsampling.
    Input feature map downsampling: merges 2x2 adjacent patches, increases channels, halves spatial dimensions.
    Initialization parameters:
        dim: input dimension
        out_dim: output dimension (default 2*dim)
        channel_first: controls channel order
    Core methods:
        _patch_merging_pad_channel_last: handles channel-last case (B,H,W,C)
        _patch_merging_pad_channel_first: handles channel-first case (B,C,H,W)
    Operation flow:
        Pad input (if dimensions are odd)
        Split 2x2 adjacent patches into 4 tensors
        Concatenate along channel dimension, channels become 4x
        Apply normalization and linear transformation (dimensionality reduction)
    '''

    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim,
                                bias=False)  # If out_dim<0 (user did not specify explicit output dim, or using default), output dim is set to 2*dim. Otherwise, use user-specified out_dim.
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
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2; 0::2 starts at index 0, picks every 2nd element; 1::2 starts at index 1, picks every 2nd element
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
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
    Gated Multi-Layer Perceptron.
    Gating mechanism:
        Expands input to double dimension via first linear layer.
        Splits result into two parts: x and z.
        Applies activation to z, then element-wise multiplies with x as gating.
        Gated result passes through second linear layer.
    Channel order support:
        channel_first=True: input format (B,C,H,W), splits along dimension 1.
        channel_first=False: input format (B,H,W,C), splits along dimension -1.
    Use cases:
        Compared to standard MLP, gMLP enhances feature selection via gating.
        In vision tasks, can replace self-attention, reducing computational complexity.
    '''

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)  # First linear layer, doubles output dimension
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)  # Second linear layer
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)  # Linear transform, output dim is 2*hidden_features
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # Split tensor into two parts along channel dim
        x = self.fc2(x * self.act(z))  # Gating: element-wise multiply x with activated z, then second linear layer
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    '''
    Spatial dimension Softmax.
    Spatial dimension normalization:
        Flattens spatial dimensions (H and W) of input tensor.
        Applies standard Softmax on flattened spatial dimension.
        Restores original tensor shape.
    Channel order support:
        dim=-1: handles channel-last input (B,H,W,C), normalizes over last dim (spatial).
        dim=1: handles channel-first input (B,C,H,W), normalizes over second dim (spatial).
    Mathematical meaning:
        Computes probability distribution over spatial positions for each channel/position.
        Commonly used in attention mechanisms to generate spatial attention maps.
    '''

    def forward(self, x: torch.Tensor):
        if self.dim == -1:  # Channel-last case (B,H,W,C)
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)  # Apply softmax on spatial dims (H*W)
        elif self.dim == 1:  # Channel-first case (B,C,H,W)
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)  # Apply softmax on spatial dims (H*W)
        else:
            raise NotImplementedError  # Only supports dim=1 or dim=-1


# =====================================================
class mamba_init:  # Static utility class for initializing key Mamba model parameters: time step (dt), state transition matrix (A), and skip connection parameter (D)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                dt_init_floor=1e-4):  # Time step parameter initialization
        '''
        Initialize time step parameters, controlling state update rate in sequence modeling.
        Maps low-rank representation to full dimension via dt_proj linear layer.
        Bias initialization ensures softplus(bias) stays in a reasonable range (avoids vanishing/exploding gradients).

        dt initialization details:
            First create a learnable linear mapping d_rank->d_inner.
            Weight init: control the init range so that variance after softplus(dt_proj(...)) does not explode.
            Bias init: delta = F.softplus(bias) ensures delta>0, but must not be too large since softplus(x) monotonically
            increases for x>0. We want delta in a reasonable range (e.g., 0.001~0.1) at initialization,
            so we apply inverse softplus to bias to keep it in [0.001, 0.1].
            (softplus: y=log(1+exp(x)); inverse softplus: y=log(exp(x)-1);
            but the official implementation uses y=x+log(exp(x)-1) to avoid numerical overflow when exp(x)-1 approaches 0.)
        '''
        dt_proj = nn.Linear(dt_rank, d_inner,
                            bias=True)  # Create linear projection layer: learnable mapping d_rank->d_inner. d_inner = int(ssm_ratio * d_model)
        # Initialize weights to preserve variance; control weight init range so variance after softplus(dt_proj(...)) does not explode.
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale  # Scaling factor
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)  # Constant initialization
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)  # Uniform distribution initialization
        else:
            raise NotImplementedError
        # Initialize bias: delta = F.softplus(bias) ensures delta>0, but we want delta in a reasonable range
        # (e.g., 0.001~0.1) at init, so apply inverse softplus to bias.
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  # Exponential distribution random values
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus transform
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)  # Set bias values
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        '''
        State transition matrix initialization:
            Initializes the log form of the state transition matrix (log(A)).
            Uses the log of state indices as initial values, helping control decay rates.
            Supports grouped operation, can create multiple different transition matrices.

        In Mamba, the HIPPO matrix initializes A, whose diagonal is 1,2,...n, forming an upper triangular matrix.
        A can be initialized with each row as an increasing sequence starting from 1.
        However, what is actually updated is A_log. When Aij>1, A_logij>0, and the actual system matrix A'
        takes the negative exponent of A_log, so A'<0. A'<0 ensures system states do not diverge,
        eventually converging to stability.
        For example, when d_state=3, the initial A sequence is [1, 2, 3]. Through subsequent negative
        exponential transform (exp(-exp(A_log))), we get decreasing decay rates, matching the requirement
        that "more distant information decays faster" in sequence modeling.
        '''
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner,
                                                                                                1).contiguous()  # S4D real-valued initialization method
        A_log = torch.log(A)  # Keep A_log in fp32; take log of state indices
        if copies > 0:  # Replicate for grouping
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)  # Merge dimensions
        A_log = nn.Parameter(A_log)  # Convert to trainable parameter
        A_log._no_weight_decay = True  # Mark as not applying weight decay
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        '''
        Skip connection parameter initialization:
            Initializes skip connection parameter D, controlling the ratio of input directly passed to output.
            Initialized to 1, allowing strong information flow in early training.
            Also supports grouped operation.
        '''
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)  # D "skip" parameter init; initialized to all ones
        if copies > 0:  # Replicate for grouping
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)  # Merge dimensions
        D = nn.Parameter(D)  # Keep in fp32; convert to trainable parameter
        D._no_weight_decay = True  # Mark as not applying weight decay
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                    k_group=4):  # Combined initialization of dt, A, D parameters
        '''
        Combined initialization method:
            Initializes all three key parameter groups needed by the Mamba model at once.
            Supports grouped initialization, each group can learn different dynamic characteristics.
            Returned parameters are directly used to build the SSM components of the Mamba model.
        '''
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]  # Initialize dt projection layers
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================  # Initialize A and D parameters
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class SS2Dv0:  # 2D feature processing module based on state space model, combining convolution and selective scan for efficient spatial feature processing
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,  # Input feature dimension
            d_state=16,  # State space dimension
            ssm_ratio=2.0,  # Expansion ratio
            dt_rank="auto",  # Time step matrix rank
            # ======================
            dropout=0.0,  # Regularization parameter
            # ======================
            seq=False,  # Whether to process sequential data
            force_fp32=True,  # Whether to force FP32 precision
            **kwargs,
    ):
        if "channel_first" in kwargs:  # Ensure not using channel_first mode (requires channel_last)
            assert not kwargs["channel_first"]
        # Configure activation function and parameters
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
        # Calculate inner dim and auto-determine dt_rank
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # Set forward function, selecting different versions based on parameters
        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================  # Input projection layer: maps input features to higher dimension
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,  # Depthwise separable convolution
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )  # 2D convolution layer: processes spatial features

        # x proj ============================  # X projection layer: projects features to state space parameters
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj],
                                                      dim=0))  # (K, N, inner); extract weight matrices from multiple linear layers, stack along new dim (dim=0), and register as trainable parameter
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )  # Initialize SSM core parameters: time step (dt), state transition matrix (A), and skip connection (D)

        # out proj =======================================  # Output projection layer: maps features back to original dim
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        # Input projection and feature separation
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d); split input into x and z
        z = self.act(z)  # Apply activation to z
        # Apply convolution to x
        x = x.permute(0, 3, 1, 2).contiguous()  # Adjust dims to (B, C, H, W)
        x = self.conv2d(x)  # (b, d, h, w); apply 2D convolution
        x = self.act(x)  # Apply activation
        # Use Mamba backend selective scan function
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape  # Get tensor shape info
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W  # Sequence length
        # Prepare multi-directional input (horizontal, vertical and their reversals)
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # Add reversed directions, forming 4 directions
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        # Compute SSM parameters: time step (dts), input mapping (Bs), and output mapping (Cs)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        # Reshape tensors for selective scan function
        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)
        # Prepare SSM core parameters
        As = -self.A_logs.float().exp()  # (k * d, d_state); state transition matrix
        Ds = self.Ds.float()  # (k * d); skip connection parameter
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d); time step bias
        # Force FP32 precision (if needed)
        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        # Apply selective scan to process sequences
        if seq:
            # Sequential mode: process each direction separately
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
        else:  # Non-sequential mode: process all directions at once
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # Merge results from different directions
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y  # Fuse information from 4 directions

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C); adjust dims and apply normalization
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z  # Apply gating mechanism and output
        out = self.dropout(self.out_proj(y))  # Final projection and dropout
        return out


# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    '''
    Multi-version 2D State Space Model module:
        This class implements a feature-rich SS2D module (SS2Dv2), supporting multiple forward versions
        (v01-v05, v2, v3, etc.) and suffix configurations (e.g., activation functions, convolution types),
        suitable for visual feature processing in different scenarios.
    '''
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,  # Input feature dimension
            d_state=16,  # State space dimension
            ssm_ratio=2.0,  # Feature expansion ratio, d_inner = d_model * ssm_ratio
            dt_rank="auto",  # Time step matrix rank; when auto, computed as ceil(d_model/16)
            act_layer=nn.SiLU,
            # dwconv ===============  # Depthwise convolution parameters
            d_conv=3,  # < 2 means no conv; convolution kernel size, <=1 disables convolution
            conv_bias=True,  # Whether convolution layer uses bias
            # ======================  # Regularization parameters
            dropout=0.0,  # Dropout rate
            bias=False,  # Whether linear layers use bias
            # dt init ==============  # Time step (dt) initialization parameters
            dt_min=0.001,  # Minimum dt value
            dt_max=0.1,  # Maximum dt value
            dt_init="random",  # Initialization method (random/constant)
            dt_scale=1.0,  # Initialization scaling factor
            dt_init_floor=1e-4,  # Initialization lower bound
            initialize="v0",  # Initialization version (v0/v1/v2)
            # ======================  # Forward pass configuration
            forward_type="v2",  # Forward pass version (e.g., v05, v2, etc.)
            channel_first=False,  # Whether to use channel-first format (NCHW)
            Scantype=None,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.Scantype = Scantype
        self.k_group = 4  # Number of groups for multi-group parameter learning
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1  # Whether to enable depthwise convolution
        Linear = Linear2d if channel_first else nn.Linear  # Select linear layer type based on channel_first (Linear2d or nn.Linear)
        self.forward = self.forwardv2

        # tags for forward_type ==============================  # Parse forward type suffixes (e.g., _no32, _noz, etc.)
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================  # Define supported forward version mapping
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),
            # selective_scan_backend="oflex", scan_mode="cross2d"  # oflex backend, cross2d mode
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # selective_scan_backend="oflex", scan_mode="cross2d"  # Disable einsum to improve throughput
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),  # Unidirectional scan
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),  # Bidirectional scan
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),  # Cascade 2D scan
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)  # Select core forward function based on type

        # in proj =======================================  # Input projection layer: if z is not disabled, output 2*d_inner dims (for splitting out gating signal z)
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================  # Depthwise convolution layer (optional)
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

        # x proj ============================   # X projection layer: maps features to SSM parameter space (dt_rank + 2*d_state)
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================  # Output module
        self.out_act = nn.GELU() if self.oact else nn.Identity()  # Output activation function (optional)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)  # Output projection layer
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()  # Dropout layer

        if initialize in ["v0"]:  # Parameter initialization version selection
            # Use mamba_init utility class to initialize core parameters (A_logs, Ds, dt_projs)
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:  # Simple initialization (for testing or specific scenarios)
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:  # Simple initialization (for testing or specific scenarios)
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32; whether to force FP32 precision
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input; oflex backend parameter, controls output precision
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput; disable einsum, use conv/linear for better performance
            # ==============================
            selective_scan_backend=None,  # Scan backend (mamba/oflex/torch)
            # ==============================
            scan_mode="cross2d",  # Scan mode (cross2d/unidi/bidi, etc.)
            scan_force_torch=False,  # Force use of PyTorch native scan
            # ==============================
            **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        scan_mode = self.Scantype
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1, hilbert=4, z_order=5, hilbert_z_order=6).get(scan_mode, None) if isinstance(scan_mode,
                                           str) else scan_mode  # for debug; parse scan mode
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)  # FP32 conversion function
        # Get input shape and parameter shapes
        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W  # Total number of spatial pixels

        # Selective scan function (dynamically selected based on backend)
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if _scan_mode == -1:  # cascade2d mode (cascade scan: first rows, then columns)
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
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
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
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

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)  # Split into 4 parameter groups
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()  # added0510 to avoid nan
            # First: row scan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),  # Take first two parameter groups
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,  # Horizontal direction scan
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            # Second: column scan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:  # Other scan modes (e.g., cross2d, unidi, bidi)
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                               force_torch=scan_force_torch)  # Cross-scan x, store results in xs. Note: xs differs from x only in scan order, values unchanged
            assert self.k_group == xs.size()[1]
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L),  # Input tensor, shape: [batch_size, input_channels, sequence_length]
                                 self.x_proj_weight.view(-1, D, 1),  # Conv kernel weights, shape: [output_channels, grouped_conv_params, conv1d_kernel_size]
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                                 # Bias term: flatten bias tensor to 1D vector, shape [K*D]
                                 groups=K  # Number of grouped conv groups; must satisfy: input_channels / groups = grouped_conv_params
                                 )  # x_dbl is the conv1d result of xs, yielding dts, Bs, Cs
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
            # Prepare SSM core parameters
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state); state transition matrix (negative, ensures decay)
            Ds = self.Ds.to(torch.float)  # (K * c); skip connection parameter
            Bs = Bs.contiguous().view(B, K, N, L)  # Input mapping
            Cs = Cs.contiguous().view(B, K, N, L)  # Output mapping
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)  # Time step bias

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)  # Force conversion to FP32
            # Apply selective scan
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                             force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))
        # Dimension adjustment and normalization
        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C); convert to NHWC format
        y = out_norm(y)  # Apply output normalization (or other post-processing, e.g., Softmax)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)  # Input projection
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)  # Core SSM processing
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z  # Gating mechanism
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        '''
        Output normalization configuration
        '''

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        # Parse suffix to determine output normalization type
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(  # 3x3 depthwise convolution
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
    def checkpostfix(tag, value):  # Check and remove suffix from forward type
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
            d_model=96,  # Input feature dimension
            d_state=16,  # State space dimension
            ssm_ratio=2.0,  # Feature expansion ratio, d_inner = d_model * ssm_ratio
            dt_rank="auto",  # Time step matrix rank; when auto, computed as ceil(d_model/16)
            # dwconv ===============
            d_conv=3,  # < 2 means no conv   # Convolution kernel size, <=1 disables convolution
            conv_bias=True,  # Whether convolution layer uses bias
            # ======================
            dropout=0.0,  # Dropout rate
            bias=False,  # Whether linear layers use bias
            # dt init ==============
            dt_min=0.001,  # Time step minimum value
            dt_max=0.1,  # Time step maximum value
            dt_init="random",  # Initialization method (random/constant)
            dt_scale=1.0,  # Initialization scaling factor
            dt_init_floor=1e-4,  # Initialization lower bound
            initialize="v0",  # Initialization version (v0/v1/v2)
            # ======================
            forward_type="v2",  # Forward pass type (e.g., xv1a, xv2a, etc.)
            channel_first=False,  # Whether to use channel-first format (NCHW)
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
        k_group = 4  # Number of groups for multi-group parameter learning
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
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
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
            x = self.cact(self.conv2d(x))  # (b, d, h, w)
        elif self.cpos:
            x = x + self.conv2d(x)  # (b, d, h, w)

        x = self.in_proj(x)

        if self.oconv2:
            x = self.conv2d(x)  # (b, d, h, w)

        us, dts, Bs, Cs = x.split([self.d_inner, self.dts_dim, 4 * self.d_state, 4 * self.d_state],
                                  dim=(1 if self.channel_first else -1))

        _us = us
        # Bs, Cs = Bs.view(B, H, W, 4, -1), Cs.view(B, H, W, 4, -1)
        # Bs, Cs = Bs.view(B, 4, -1, H, W), Cs.view(B, 4, -1, H, W)
        us = cross_scan_fn(us.contiguous(), in_channel_first=self.channel_first, out_channel_first=True).view(B, -1, L)
        Bs = cross_scan_fn(Bs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                           one_by_one=True).view(B, 4, -1, L)
        Cs = cross_scan_fn(Cs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                           one_by_one=True).view(B, 4, -1, L)
        dts = cross_scan_fn(dts.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                            one_by_one=(self.dts_dim == 4 * self.dt_rank)).view(B, L, -1)
        if self.dts_dim == self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
        elif self.dts_dim == 4 * self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)  # (K * c)

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
            d_state=16,  # now with mamba2, dstate should be bigger...
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.GELU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
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
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)),
                                              requires_grad=False)

    def forward_corem0(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            chunk_size=64,
            dstate=64,
            # ==============================
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
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
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode,
                           force_torch=scan_force_torch)  # (B, H, W, 4, D)
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
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False,
                                         scans=_scan_mode, force_torch=scan_force_torch)

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
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0):
    '''
    Unified interface for multi-version state space models.
        This class implements a module called "SS2D" that integrates multiple versions of SS2D
        implementations (SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0) via multiple inheritance, and selects
        different initialization methods based on the forward_type parameter, supporting
        various forward propagation strategies and configuration options.
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
            d_conv=3,  # < 2 means no conv
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





# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS3Dv2:
    # Multi-version 3D state space model module:
    #     This class implements a feature-rich SS3D module (SS3Dv2), supporting multiple forward versions
    #     (v01-v05, v2, v3, etc.) and suffix configurations (e.g., activation functions, convolution types),
    #     suitable for visual feature processing in different scenarios.
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,  # Input feature dimension
            d_state=16,  # State space dimension
            ssm_ratio=2.0,  # Feature expansion ratio, d_inner = d_model * ssm_ratio
            dt_rank="auto",  # Time step matrix rank; when auto, computed as ceil(d_model/16)
            act_layer=nn.SiLU,
            # dwconv ===============# Depthwise convolution parameters
            d_conv=3,  # < 2 means no conv  # Convolution kernel size, <=1 disables convolution
            conv_bias=True,  # Whether convolution layer uses bias
            # ====================== # Regularization parameters
            dropout=0.0,  # Dropout rate
            bias=False,  # Whether linear layers use bias
            # dt init ==============# Time step (dt) initialization parameters
            dt_min=0.001,  # Minimum dt value
            dt_max=0.1,  # Maximum dt value
            dt_init="random",  # Initialization method (random/constant)
            dt_scale=1.0,  # Initialization scaling factor
            dt_init_floor=1e-4,  # Initialization lower bound
            initialize="v0",  # Initialization version (v0/v1/v2)
            # ======================# Forward pass configuration
            forward_type="v2",  # Forward pass version (e.g., v05, v2, etc.)
            channel_first=False,  # Whether to use channel-first format (NCHW)
            # ======================
            **kwargs
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4  # Number of groups for multi-group parameter learning
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1  # Whether to enable depthwise convolution
        Linear = Linear2d if channel_first else nn.Linear  # Select linear layer type based on channel_first (Linear2d or nn.Linear)
        self.forward = self.forwardv2

        self.num_modalities = 3 # Supports H+W, C+H, C+W

        # tags for forward_type ==============================# Parse forward type suffixes (e.g., _no32, _noz, etc.)
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug ======================================= # Define supported forward version mapping
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True, fuse_mode = kwargs["fuse_mode"]),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",  fuse_mode = kwargs["fuse_mode"]),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex",  fuse_mode = kwargs["fuse_mode"]),
            v04=partial(self.forward_corev2, force_fp32=False,  fuse_mode = kwargs["fuse_mode"]),
            # selective_scan_backend="oflex", scan_mode="cross2d"   # oflex backend, cross2d mode
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True,  fuse_mode = kwargs["fuse_mode"]),
            # selective_scan_backend="oflex", scan_mode="cross2d"  # Disable einsum to improve throughput
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi",  fuse_mode = kwargs["fuse_mode"]),  # Unidirectional scan
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi",  fuse_mode = kwargs["fuse_mode"]),  # Bidirectional scan
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d",  fuse_mode = kwargs["fuse_mode"]),  # Cascade 2D scan
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3,  fuse_mode = kwargs["fuse_mode"]),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core",  fuse_mode = kwargs["fuse_mode"]),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex",  fuse_mode = kwargs["fuse_mode"]),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)  # Select core forward function based on type

        # in proj ======================================= # Input projection layer: if z is not disabled, output 2*d_inner dims (for splitting out gating signal z)
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================  # Depthwise convolution layer (optional)
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


        '''
        # x proj ============================   # X projection layer: maps features to SSM parameter space (dt_rank + 2*d_state)
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # x_proj_weight currently: shape (K, out_dim, in_dim)
        # create per-modality parameter (M, K, out_dim, in_dim)
        # simple initialization: copy the same initial projection into each modality;
        # user can reinitialize differently if desired.
        self.x_proj_weight_modal = nn.Parameter(self.x_proj_weight.unsqueeze(0).repeat(self.num_modalities, 1, 1, 1))
        '''
        # Create k_group LazyLinear layers per modality (no need to pre-specify in_features)
        # Structure: self.proj_layers[m][k] represents the projection layer of the k-th group of the m-th modality
        self.x_proj_out_dim = self.dt_rank +2*self.d_state# Define output dimension of x_proj
        # Two-level ModuleList: outer=number of modalities (M), inner=k_group LazyLinear per modality
        self.x_proj_layers = nn.ModuleList()
        for _ in range(self.num_modalities):
            # Create k_group LazyLinear per modality (no need to pre-specify in_features, auto-inferred)
            modal_layers = nn.ModuleList(
                [nn.LazyLinear(out_features=self.x_proj_out_dim, bias=False) for _ in range(self.k_group)]
            )
            self.x_proj_layers.append(modal_layers)



        # If you had an x_proj_bias present, duplicate it too (optional)
        x_proj_bias = getattr(self, "x_proj_bias", None)
        if x_proj_bias is not None:
            self.x_proj_bias_modal = nn.Parameter(x_proj_bias.view(1, -1).repeat(self.num_modalities, 1))
        else:
            self.x_proj_bias_modal = None



        # out proj =======================================  # Output module
        self.out_act = nn.GELU() if self.oact else nn.Identity()  # Output activation function (optional)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)  # Output projection layer
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()  # Dropout layer

        # This will create separate A_logs, Ds and dt_projs for each modality.
        # --------------------------
        if initialize in ["v0"]:  # Parameter initialization version selection
            # Use mamba_init utility class to initialize core parameters (A_logs, Ds, dt_projs)
            # self.A_logs_list = []
            # self.Ds_list = []
            # self.dtpw_list = []#short for dt_projs_weight
            # self.dtpb_list = []#short for dt_projs_bias
            '''
            for m in range(self.num_modalities):
                A_logs_m, Ds_m, dtpw_m, dtpb_m = mamba_init.init_dt_A_D(
                    self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                    k_group=self.k_group,
                )
                # A_logs_m: nn.Parameter((k_group * d_inner, d_state))
                # Ds_m: nn.Parameter((k_group * d_inner))
                # dtpw_m: nn.Parameter((k_group, d_inner, dt_rank))
                # dtpb_m: nn.Parameter((k_group, d_inner))
                A_logs_list.append(A_logs_m.detach())
                Ds_list.append(Ds_m.detach())
                dtpw_list.append(dtpw_m.detach())
                dtpb_list.append(dtpb_m.detach())

            # stack and register as parameters with modality dim first
            self.A_logs_modal = nn.Parameter(torch.stack(A_logs_list, dim=0))  # (M, k_group * d_inner, d_state)
            self.A_logs_modal._no_weight_decay = True
            self.Ds_modal = nn.Parameter(torch.stack(Ds_list, dim=0))  # (M, k_group * d_inner)
            self.Ds_modal._no_weight_decay = True
            self.dt_projs_weight_modal = nn.Parameter(
                torch.stack(dtpw_list, dim=0))  # (M, k_group, d_inner, dt_rank)
            self.dt_projs_bias_modal = nn.Parameter(torch.stack(dtpb_list, dim=0))  # (M, k_group, d_inner)
            # cleanup temp lists
            del A_logs_list, Ds_list, dtpw_list, dtpb_list
            '''
            # Use lazy init method, perform real initialization in forward
            self.initialized =False    # Mark whether initialization has been completed (core flag)
            # Store initialization config (used in subsequent forward calls)
            self.init_config = {
                "d_state": self.d_state,
                "dt_rank": self.dt_rank,
                "d_inner": self.d_inner,
                "dt_scale": dt_scale,
                "dt_init": dt_init,
                "dt_min": dt_min,
                "dt_max": dt_max,
                "dt_init_floor": dt_init_floor,
                "k_group": self.k_group,
                "num_modalities":self.num_modalities,
                "ssm_ratio":ssm_ratio
            }

        elif initialize in ["v1", "v2"]: # Simple initialization (for testing or specific scenarios)
            raise NotImplementedError
            # for simpler inits, just tile simple inits across modalities
            self.Ds_modal = nn.Parameter(torch.ones((self.num_modalities, self.k_group * self.d_inner)))
            self.A_logs_modal = nn.Parameter(torch.zeros((self.num_modalities, self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight_modal = nn.Parameter(0.1 * torch.rand((self.num_modalities, self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias_modal = nn.Parameter(0.1 * torch.rand((self.num_modalities, self.k_group, self.d_inner)))

        # --------------------------
        # Add a small fusion head for the 3 modalities (map 3 * d_inner -> d_inner)
        # --------------------------
        # fusion conv: fuse modality outputs (channel_first layout expected later)
        self.fusion_conv = nn.Conv2d(self.d_inner * self.num_modalities, self.d_inner, kernel_size=1, bias=False)



    def _lazy_init(self, x: torch.Tensor):
        """Initialize per-modality parameters based on actual input x dimensions"""
        device = x.device
        # Input x shape: [batch, channel, height, width]
        batch, channel, height, width = x.shape
        num_modalities = self.init_config["num_modalities"]
        # Compute corresponding d_inner for each modality (associated with channel/height/width respectively)
        # Modality order: 0->(B, C, H, W), 1->(B, W, C, H), 2->(B, H, C, W)
        d_inner_list = [
            int(self.init_config["ssm_ratio"] * channel),# m=0: [C, H,W], correlated with C
            int(                                width),  # m=1: [W, C,H], correlated with W
            int(                                height)  # m=2: [H ,C,W], correlated with H
            ]
        # Initialize parameters per modality
        for m in range(num_modalities):
            d_inner = d_inner_list[m]
            # Call initialization utility (using current modality's d_inner)
            A_logs_m, Ds_m, dtpw_m, dtpb_m = mamba_init.init_dt_A_D(
                d_state=self.init_config["d_state"],
                dt_rank=self.init_config["dt_rank"] if self.init_config["dt_rank"] != "auto" else max(d_inner // 16, 1),
                d_inner=d_inner,
                dt_scale=self.init_config["dt_scale"],
                dt_init=self.init_config["dt_init"],
                dt_min=self.init_config["dt_min"],
                dt_max=self.init_config["dt_max"],
                dt_init_floor=self.init_config["dt_init_floor"],
                k_group=self.init_config["k_group"],
            )#A_logs_m dims are [k_group*ssm_ratio*d_inner, d_state]
            # Ds_m dims are [k_group*ssm_ratio*d_inner, 1]
            # dtpw_m dims are [k_group, ssm_ratio*d_inner, dt_rank]
            # dtpb_m dims are [k_group, ssm_ratio*d_inner, 1]

            '''
            # Store initialized parameters (detach() removed from original code, retaining trainability)
            self.A_logs_list.append(A_logs_m)
            self.Ds_list.append(Ds_m)
            self.dtpw_list.append(dtpw_m)
            self.dtpb_list.append(dtpb_m)
            '''

            A_logs_m = A_logs_m.to(device)
            Ds_m = Ds_m.to(device)
            dtpw_m = dtpw_m.to(device)
            dtpb_m = dtpb_m.to(device)

            setattr(self, f"A_logs_m{m}", nn.Parameter(A_logs_m))
            getattr(self, f"A_logs_m{m}")._no_weight_decay = True

            setattr(self, f"Ds_m{m}", nn.Parameter(Ds_m))
            getattr(self, f"Ds_m{m}")._no_weight_decay = True

            setattr(self, f"dt_projs_weight_m{m}", nn.Parameter(dtpw_m))

            setattr(self, f"dt_projs_bias_m{m}", nn.Parameter(dtpb_m))

            setattr(self, f"d_inner_m{m}", d_inner)
        '''
        # Stack parameters and register as module attributes (preserving modality dimension from original code)
        self.A_logs_modal = nn.Parameter(torch.stack(self.A_logs_list, dim=0))  # (M, k_group * d_inner, d_state)
        self.A_logs_modal._no_weight_decay = True

        self.Ds_modal = nn.Parameter(torch.stack(self.Ds_list, dim=0))  # (M, k_group * d_inner)
        self.Ds_modal._no_weight_decay = True

        self.dt_projs_weight_modal = nn.Parameter(
            torch.stack(self.dtpw_list, dim=0))  # (M, k_group, d_inner, dt_rank)

        self.dt_projs_bias_modal = nn.Parameter(
            torch.stack(self.dtpb_list, dim=0))  # (M, k_group, d_inner)

        # Clean up temporary lists, mark initialization as complete
        del self.A_logs_list, self.Ds_list, self.dtpw_list, self.dtpb_list
        self.initialized = True
        '''
        self.d_inner_list = d_inner_list

        # 5. Mark initialization complete, clean up temporary config
        self.initialized = True

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32 # Whether to force FP32 precision
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input  # oflex backend parameter, controls output precision
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput  # Disable einsum, use conv/linear layers for better performance
            # ==============================
            selective_scan_backend=None,  # Scan backend (mamba/oflex/torch)
            # ==============================
            scan_mode="cross2d",  # Scan mode (cross2d/unidi/bidi, etc.)
            scan_force_torch=False,  # Force use of PyTorch native scan
            # ==============================
            fuse_mode ="Null",
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug  # Parse scan mode
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)  # FP32 conversion function
        # Get input shape and parameter shapes
        B, Ch, H, W = x.shape
        N = self.d_state
        K, R = self.k_group, self.dt_rank
        L = H * W  # Total number of spatial pixels

        # Selective scan function (dynamically selected based on backend)
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)
        if _scan_mode == -1:  # cascade2d mode (cascade scan: first rows, then columns)
            x_proj_bias = getattr(self, "x_proj_bias", None)
            raise NotImplementedError("This branch has not been verified yet. Lijie Yang")
            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
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
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
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

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)  # Split into 4 parameter groups
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()  # added0510 to avoid nan
            # First: row scan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),  # Take first two parameter groups
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,  # Horizontal direction scan
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            # Second: column scan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:  # Other scan modes (e.g., cross2d, unidi, bidi)
            x_proj_bias = getattr(self, "x_proj_bias", None)

            # We'll compute 3 modality outputs and fuse them.
            modality_outputs = []

            # We uniformly pass in_channel_first=True (NCHW) to cross_scan_fn,
            # and use permute to place the two dimensions to be serialized at the H, W positions:
            # Modality 0 (H+W):  original (B,C,H,W) -> unchanged
            # Modality 1 (C+H):  place (C, H) as space -> construct (B, C_in, H_in, W_in) = (B, W, C, H)
            #               hence perm=(0,3,1,2) transforms original (B,C,H,W) -> (B,W,C,H)
            # Modality 2 (C+W):  place (C, W) as space -> construct (B, C_in, H_in, W_in) = (B, H, C, W)
            #               hence perm=(0,2,1,3) transforms original (B,C,H,W) -> (B,H,C,W)
            permutes = [
                (None, True),            # modality 0: no perm, in_channel_first=True
                ((0, 3, 1, 2), True),    # modality 1: perm -> (B, W, C, H), in_channel_first=True
                ((0, 2, 1, 3), True),    # modality 2: perm -> (B, H, C, W), in_channel_first=True
            ]

            for m in range(self.num_modalities):
                perm, in_chan_first = permutes[m]
                if perm is not None:
                    x_m = x.permute(*perm).contiguous()   # Note: x may have been previously transformed to channel-first format (B, D, H, W)
                else:
                    x_m = x

                _, Ch_m, H_m, W_m = x_m.shape
                L_m = H_m * W_m # Total number of spatial pixels


                # Now x_m is NCHW (because in_chan_first=True), channel dimension is x_m.shape[1]
                # cross_scan_fn expects input as (B,C,H,W) or (B,H,W,C) — we keep (B,C,H,W) here
                xs = cross_scan_fn(x_m, in_channel_first=in_chan_first, out_channel_first=True, scans=_scan_mode,
                                   force_torch=scan_force_torch)
                assert K == xs.size()[1]
                B, K, d_inner_m, L_m = xs.shape  # Dynamically get the current modality's d_inner_m (key: different modalities may differ)

                # d_inner_m = getattr(self, f"d_inner_m{m}")
                # assert d_inner_m == xs.shape[2]

                # Get the current modality's k_group LazyLinear layers
                current_modal_x_proj = self.x_proj_layers[m]  # Inner ModuleList: (K LazyLinear layers)
                x_dbl_list = []  # Store x_proj result for each group
                for k in range(K):
                    # Extract the k-th group features: (B, d_inner_m, L_m)
                    xs_k = xs[:, k, :, :]  # (B, d_inner_m, L_m)
                    # Reshape to LazyLinear input format: (B*L_m, d_inner_m) (Linear processes the last dimension)
                    xs_k_reshaped = xs_k.permute(0, 2, 1).reshape(-1, d_inner_m)  # (B*L_m, d_inner_m)  The LazyLinear layer below processes d_inner_m, permute ensures the sequence order from cross_scan_fn is preserved
                    # Call the k-th LazyLinear layer (auto-infers d_inner_m and initializes weights on first call)
                    x_proj_k = current_modal_x_proj[k](xs_k_reshaped)  # (B*L_m, x_proj_out_dim)    [131072,35]
                    # Restore dimensions: (B, L_m, x_proj_out_dim) -> (B, 1, x_proj_out_dim, L_m) (for subsequent concatenation)
                    x_proj_k_reshaped = x_proj_k.reshape(B, L_m, -1).permute(0, 2, 1).unsqueeze(1)  # (B, 1, out_dim, L_m)     [4,1,35,32768]
                    x_dbl_list.append(x_proj_k_reshaped)

                # Concatenate all groups results: (B, K, x_proj_out_dim, L_m) (same shape as original x_dbl)
                x_dbl = torch.cat(x_dbl_list, dim=1)
                # ------------------------------------------------------------------------------------------

                # Split SSM parameters (dts, Bs, Cs) — original logic preserved (x_dbl shape unchanged)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # dts:(B,K,R,L_m); Bs/Cs:(B,K,N,L_m)

                # dt projection (modality-specific parameters) — original logic preserved
                dt_pw_m = getattr(self, f"dt_projs_weight_m{m}")   # (K, d_inner_m, R)
                if no_einsum:
                    # Dimensions after adapting LazyLinear: dts -> (B, K*d_inner_m, L_m)
                    dt_pw_m_reshaped = dt_pw_m.reshape(K*self.d_inner_list[m], -1, 1)  # (K*R, d_inner_m, 1)

                    dts_reshaped = dts.contiguous().reshape(B, -1, L_m)  # (B, K*R, L_m)
                    dts = F.conv1d(dts_reshaped, dt_pw_m_reshaped, groups=K)
                    '''
                    # Dimensions after adapting LazyLinear: dts -> (B, K*d_inner_m, L_m)
                    dts_reshaped = dts.permute(0, 1, 3, 2).reshape(B, K*R, L_m)  # (B, K*R, L_m)
                    dt_pw_m_reshaped = dt_pw_m.permute(0, 2, 1).reshape(K*R, d_inner_m, 1)  # (K*R, d_inner_m, 1)
                    dts = F.conv1d(dts_reshaped, dt_pw_m_reshaped, groups=K).reshape(B, K, d_inner_m, L_m)
                    '''

                else:
                    raise NotImplementedError
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_pw_m)  # (B,K,d_inner_m,L_m)


                # SSM parameters (per modality)
                xs = xs.view(B, -1, L_m)
                dts = dts.contiguous().view(B, -1, L_m)
                As = -getattr(self, f"A_logs_m{m}").to(torch.float).exp()
                Ds = getattr(self, f"Ds_m{m}").to(torch.float)
                Bs = Bs.contiguous().view(B, K, N, L_m)
                Cs = Cs.contiguous().view(B, K, N, L_m)
                delta_bias = getattr(self, f"dt_projs_bias_m{m}") .view(-1).to(torch.float)

                if force_fp32:
                    xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

                # Call selective scan (preserving original calling convention)
                ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias,
                                       delta_softplus)
                # Note: selective_scan_fn returns the same flat/unfolded form as before, then reshape

                # Reshape ys to (B, K, C', H_m, W_m) or (B, K, N, H_m, W_m) compatible with cross_merge_fn
                ys = ys.view(B, K, -1, H_m, W_m)

                # merge back into a compact tensor equivalent to x_m via cross_merge_fn
                y_m = cross_merge_fn(ys, in_channel_first=in_chan_first, out_channel_first=True, scans=_scan_mode,
                                      force_torch=scan_force_torch)

                # Now y_m may be (B, C_in, H_m*W_m) or (B, H_m*W_m, C_in) or (B, C_in, H_m, W_m)
                # Uniformly restore to (B, C_in, H_m, W_m), here in_chan_first=True so (B, C, L) -> (B, C, H_m, W_m)
                if y_m.dim() == 3:
                    # (B, C_in, L)
                    y_m = y_m.view(B, -1, H_m, W_m)
                elif y_m.dim() == 4:
                    # (B, C_in, H_m, W_m) is already NCHW, no transformation needed
                    pass
                else:
                    raise RuntimeError(f"unexpected shape from cross_merge_fn: {y_m.shape}")

                # Inverse permute y_m from x_m layout back to original x layout (apply inverse permutation)
                if perm is not None:
                    inv = [0] * len(perm)
                    for i, p in enumerate(perm):
                        inv[p] = i
                    y_m = y_m.permute(*inv).contiguous()

                # Final y_m should return to (B, d_inner, H, W)
                assert y_m.shape[0] == B and y_m.shape[1] == self.d_inner and y_m.shape[2] == H and y_m.shape[3] == W, \
                    f"modal y_m shape mismatch: {y_m.shape} (expected B,{self.d_inner},{H},{W})"


                # if m==0:
                #     y_m *= 0

                modality_outputs.append(y_m)
                # --- Optional debug (enable during development) ---
                # print(f"modal {m}: perm={perm}, in_chan_first={in_chan_first}, x.shape={x.shape}, x_m.shape={x_m.shape}, y_m.shape={y_m.shape}")


            # print("modality_outputs: ", [modality_outputs[a][1, 10, 16, 8] for a in range(3)])

            if fuse_mode == 'fusion_conv':#Default mode
                # fuse modalities: concatenate along channel (dim=1) and fuse
                y_cat = torch.cat(modality_outputs, dim=1)  # (B, 3 * d_inner, H, W)
                y = self.fusion_conv(y_cat)                 # (B, d_inner, H, W)
            elif fuse_mode == 'fusion_add':
                y_cat = torch.stack(modality_outputs, dim=0)  # (3, B, d_inner, H, W)
                y = torch.sum(y_cat, dim=0)                 # (B, d_inner, H, W)
            else:
                raise NotImplementedError

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    modality_outputs=modality_outputs, y_cat=y_cat, y=y
                ))


        # Dimension adjustment and normalization
        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)# Convert to NHWC format
        y = out_norm(y)  # Apply output normalization (or other post-processing, e.g., Softmax)

        return y.to(x.dtype)


    def forwardv2(self, x: torch.Tensor, **kwargs):
        # Perform lazy initialization on first forward pass
        if not self.initialized and hasattr(self, "init_config"):
            self._lazy_init(x)

        x = self.in_proj(x)  # Input projection
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x,**kwargs)  # Core SSM processing
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z  # Gating mechanism
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        '''
        Output normalization configuration
        '''

        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        # Parse suffix to determine output normalization type
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(  # 3x3 depthwise convolution
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
    def checkpostfix(tag, value):  # Check and remove suffix from forward type
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

class SS3D(nn.Module, SS3Dv2):
    '''
    Unified interface for multi-version state space models.
        This class implements a module called "SS3D" that integrates multiple versions of SS3D
        implementations (SS2Dv2) via multiple inheritance, and selects different initialization
        methods based on the forward_type parameter, supporting various forward propagation
        strategies and configuration options.
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # 3D convolution params ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # other params ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v2",
            # forward type ======================
            forward_type="v2",
            channel_first=False,
            # 3D-specific parameters ======================
            spatial_dims=3,  # 3D spatial dimensions
            # kernel_size=(3, 3, 3),  # 3D convolution kernel size
            # fuse_mode="Null",
            **kwargs,
    ):
        nn.Module.__init__(self)
        # Consolidate all parameters
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias,
            dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, initialize=initialize,
            forward_type=forward_type, channel_first=channel_first,
            spatial_dims=spatial_dims, #, kernel_size=kernel_size
        )

        '''        
        # Select different initialization methods based on forward_type
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type),** kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(** kwargs)
        # Default to v2 version initialization
        else:
            self.__initv2__(**kwargs)
        '''
        self.__initv2__(**kwargs)








# =====================================================
class VSSBlock(
    nn.Module):  # "VSSBlock" module combines State Space Model (SSM) and Multi-Layer Perceptron (MLP) to process visual features. This module supports flexible configuration options, allowing adjustment of SSM and MLP usage ratios, normalization methods, etc.
    def __init__(
            self,
            hidden_dim: int = 0,  # Hidden layer dimension
            drop_path: float = 0,  # Drop Path probability
            norm_layer: nn.Module = nn.LayerNorm,  # Normalization layer type
            channel_first=False,  # Whether to use channel-first data format
            # ============================= # SSM parameters
            ssm_d_state: int = 16,  # SSM state dimension
            ssm_ratio=2.0,  # SSM branch expansion ratio
            ssm_dt_rank: Any = "auto",  # SSM time step matrix rank
            ssm_act_layer=nn.SiLU,  # SSM activation function
            ssm_conv: int = 3,  # SSM convolution kernel size
            ssm_conv_bias=True,  # Whether SSM conv layer uses bias
            ssm_drop_rate: float = 0,  # SSM branch dropout rate
            ssm_init="v0",  # SSM initialization method
            forward_type="v2",  # SSM forward propagation type
            # =============================# MLP parameters
            mlp_ratio=4.0,  # MLP branch expansion ratio
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,  # MLP branch dropout rate
            gmlp=False,  # Whether to use gated MLP
            # =============================
            use_checkpoint: bool = False,  # Whether to use gradient checkpointing
            post_norm: bool = False,  # Whether to use post-normalization
            # =============================
            _SS2D: type = SS2D,  # SSM module type
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0  # Whether to enable SSM branch
        self.mlp_branch = mlp_ratio > 0  # Whether to enable MLP branch
        self.use_checkpoint = use_checkpoint  # Whether to use gradient checkpointing
        self.post_norm = post_norm  # Whether to use post-normalization

        if self.ssm_branch:  # Initialize SSM branch
            self.norm = norm_layer(hidden_dim)  # Normalization layer
            self.op = _SS2D(
                d_model=hidden_dim,  # Input feature dimension
                d_state=ssm_d_state,  # SSM state dimension
                ssm_ratio=ssm_ratio,  # Expansion ratio
                dt_rank=ssm_dt_rank,  # Time step matrix rank
                act_layer=ssm_act_layer,  # Activation function
                # ==========================
                d_conv=ssm_conv,  # Convolution kernel size
                conv_bias=ssm_conv_bias,  # Whether to use convolution bias
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
                channel_first=channel_first,  # Channel-first flag
                **kwargs,
            )
        # Drop Path regularization
        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:  # Initialize MLP branch
            _MLP = Mlp if not gmlp else gMlp  # Select standard MLP or gated MLP based on flag
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)  # Calculate MLP hidden layer dimension
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:  # Process SSM branch
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))  # Post-normalization: apply SSM first, then normalize and add residual
            else:
                x = x + self.drop_path(self.op(self.norm(x)))  # Pre-normalization: normalize first, then apply SSM and add residual
        if self.mlp_branch:  # Process MLP branch
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN   # Post-normalization: apply MLP first, then normalize and add residual
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN# Pre-normalization: normalize first, then apply MLP and add residual
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,  # Image patch size
            in_chans=3,  # Number of input channels (RGB=3)
            num_classes=1000,  # Number of classification classes
            depths=[2, 2, 9, 2],  # Number of layers per stage
            dims=[96, 192, 384, 768],  # Feature dimensions per stage
            # SSM related parameters
            ssm_d_state=16,  # SSM state dimension
            ssm_ratio=2.0,  # SSM expansion ratio
            ssm_dt_rank="auto",  # SSM time step matrix rank
            ssm_act_layer="silu",  # SSM activation function
            ssm_conv=3,  # SSM convolution kernel size
            ssm_conv_bias=True,  # Whether SSM convolution uses bias
            ssm_drop_rate=0.0,  # SSM dropout rate
            ssm_init="v0",  # SSM initialization method
            forward_type="v2",  # Forward propagation version
            # MLP related parameters
            mlp_ratio=4.0,  # MLP hidden layer expansion ratio
            mlp_act_layer="gelu",  # MLP activation function
            mlp_drop_rate=0.0,  # MLP dropout rate
            gmlp=False,  # Whether to use gMLP
            # Other parameters
            drop_path_rate=0.1,  # Stochastic depth drop rate
            patch_norm=True,  # Whether to use normalization after patch embedding
            norm_layer="LN",  # Normalization layer type (LN=LayerNorm)
            downsample_version: str = "v2",  # Downsampling version
            patchembed_version: str = "v1",  # Patch embedding version
            use_checkpoint=False,  # Whether to use checkpointing to save memory
            posembed=False,  # Whether to use positional encoding

            imgsize=224,  # Input image size
            _SS2D=SS2D,  # SSM 2D implementation class
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])  # Determine if normalization layer is channel-first format (e.g., BN/LN2D)
        self.num_classes = num_classes  # Save number of classification classes
        self.num_layers = len(depths)
        if isinstance(dims, int):  # Handle dims parameter: if int, auto-generate layer-wise doubling dimension list
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]  # Feature dimension of the last stage (for classification head)
        self.dims = dims  # Save feature dimensions for each stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule #Stochastic depth regularization, calculates the drop probability for each residual block
        # These two dictionaries define module mappings for dynamic selection of normalization layers (_NORMLAYERS) and activation functions (_ACTLAYERS). This design allows the model to flexibly switch between different layer types based on configuration, enhancing code extensibility and reusability.
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )  # Define normalization layer mapping dict (supports LN, LN2D, BN)

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )  # Define activation function mapping dict (supports silu, gelu, relu, etc.)
        # Get actual normalization layer and activation function classes from string names
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        # Initialize positional encoding (if posembed is enabled)
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        self.patch_h = kwargs.get("patch_h", patch_size)
        self.patch_w = kwargs.get("patch_w", patch_size)
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)  # Select patch embedding module based on version (v1/v2 different implementations)
        # self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
        #                                      channel_first=self.channel_first)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first,patch_h=self.patch_h,patch_w=self.patch_w)



        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),  # Return identity mapping when no downsampling
        ).get(downsample_version, None)  # Select downsampling module based on version (v1/v2/v3 or none)

        self.layers = nn.ModuleList()  # Initialize layers for each stage
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()  # Determine if current stage needs downsampling (add downsampling for non-final stages)

            self.layers.append(self._make_layer(  # Create current stage layers (containing multiple VSSBlocks and downsampling module)
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # Stochastic depth rate for current stage
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================# Pass SSM related parameters
                ssm_d_state=ssm_d_state,  # SSM state dimension
                ssm_ratio=ssm_ratio,  # Expansion ratio
                ssm_dt_rank=ssm_dt_rank,  # Time step matrix rank
                ssm_act_layer=ssm_act_layer,  # Activation function
                ssm_conv=ssm_conv,  # Convolution kernel size
                ssm_conv_bias=ssm_conv_bias,  # Whether to use convolution bias
                ssm_drop_rate=ssm_drop_rate,  # dropout rate
                ssm_init=ssm_init,  # Initialization method
                forward_type=forward_type,  # Forward propagation type
                # =================# Pass MLP related parameters
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # ================= # Pass SSM 2D implementation class
                _SS2D=_SS2D,
                **kwargs
            ))
        # Define classifier: normalization + pooling + linear layer
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),  # B,H,W,C   #Normalize features of the last stage
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),  # Adjust channel order
            avgpool=nn.AdaptiveAvgPool2d(1),  # Global average pooling
            flatten=nn.Flatten(1),  # Flatten features
            head=nn.Linear(self.num_features, num_classes),  # Classification head
        ))  # classifier will be removed in backbone_VSSM, backbone does not need classification head

        if "SS3D" in _SS2D.__name__:#In this mode, weights use lazyInit, so weight initialization is done in forward
            # self._is_initialized = False  # Mark whether weights have been initialized
            pass
        else:
            self.apply(self._init_weights)  # Apply weight initialization function
            # self._is_initialized = True  # Mark whether weights have been initialized
        # self.apply(self._init_weights)  # Apply weight initialization function


    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):  # (Positional encoding initialization)
        patch_height, patch_width = (img_size // patch_size,
                                     img_size // patch_size)  # Calculate patch grid rows and columns (e.g., img_size=224, patch_size=4 -> 56x56)
        pos_embed = nn.Parameter(
            torch.zeros(1, embed_dims, patch_height, patch_width))  # Create learnable positional encoding parameters (shape: 1 x embed_dims x H x W)
        trunc_normal_(pos_embed, std=0.02)  # Initialize positional encoding with truncated normal distribution (std 0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):  # Weight initialization
        if isinstance(m, nn.Linear):  # Linear layer weight initialization (truncated normal distribution)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # Bias initialized to 0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # Normalization layer weight initialization (all 1) and bias initialization (all 0)
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
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False,**kwargs,):
        # if channel first, then Norm and Output are both channel_first # Build patch embedding module: convolution + normalization (optional) + channel order adjustment
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=(kwargs.get("patch_h"),kwargs.get("patch_w")), stride=(kwargs.get("patch_h"),kwargs.get("patch_w")), bias=True),  # Patch convolution
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),  # Convert to (B, H, W, C) format
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),  # Normalization
        )
    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2  # Stride is half of patch_size (overlapping patches)
        kernel_size = stride + 1  # Convolution kernel size (e.g., stride=2 -> kernel=3, with overlap)
        padding = 1  # Padding to maintain spatial size
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            # First convolution layer: reduce dimension to embed_dim/2
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            # Channel order adjustment (based on channel_first and normalization flags)
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),  # Normalization
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),  # Restore channel order
            nn.GELU(),  # Activation function
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            # Second convolution layer: increase dimension to embed_dim
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),  # Convert to (B, H, W, C)
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),  # Normalization
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first# Downsampling module: convolution (stride=2) + normalization + channel order adjustment
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),  # Convert to (B, C, H, W) format
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),  # Halve spatial size, double channel count
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),  # Convert back to (B, H, W, C)
            norm_layer(out_dim),  # Normalization
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm,
                            channel_first=False):  # (v3 version downsampling module, using 3x3 convolution)
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),  # 3x3 convolution downsampling
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(  # Create a single stage of layers, containing multiple VSSBlocks
            dim=96,  # Feature dimension of current stage
            drop_path=[0.1, 0.1],  # Stochastic depth rate for each layer in this stage
            use_checkpoint=False,  # Whether to enable checkpointing
            norm_layer=nn.LayerNorm,  # Normalization layer type
            downsample=nn.Identity(),  # Downsampling module (default identity mapping)
            channel_first=False,  # Whether to use channel-first format
            # ===========================#SSM parameters
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================== #MLP parameters
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================== #SSM 2D implementation class
            _SS2D=SS2D,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)  # Number of layers in this stage (e.g., a value from depths)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(  # Create VSSBlock for each layer (core module, contains SSM and MLP)
                hidden_dim=dim,
                drop_path=drop_path[d],  # Stochastic depth rate for current layer
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
                **kwargs,
            ))
        # Return the layers of this stage (containing all blocks and downsampling module)
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),  # Sequence of multiple blocks
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)  # Split input image into patches and embed into feature space
        if self.pos_embed is not None:  # If using positional encoding
            pos_embed = self.pos_embed.permute(0, 2, 3,
                                               1) if not self.channel_first else self.pos_embed  # Adjust positional encoding dimension order based on channel format
            x = x + pos_embed  # Add positional encoding to patch embedding
        for layer in self.layers:  # Iterate through each stage's layers
            x = layer(x)  # Each layer contains multiple VSSBlocks and optional downsampling
        x = self.classifier(x)  # Get final prediction through classification head
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):  # Calculate model FLOPs using fvcore library
        # shape = self.__input_shape__[1:]# Define supported operations and their FLOPs calculation functions
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS# Ignore FLOPs calculation for these operations
            "aten::neg": None,  # as relu is in _IGNORED_OPS# These operations typically have negligible computational cost
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)  # Copy model to avoid modifying original
        model.cuda().eval()  # Move model to GPU and set to eval mode

        input = torch.randn((1, *shape), device=next(model.parameters()).device)  # Create random input tensor
        params = parameter_count(model)[""]  # Calculate model parameter count
        Gflops, unsupported = flop_count(model=model, inputs=(input,),
                                         supported_ops=supported_ops)  # Use fvcore's flop_count to compute FLOPs

        del model, input
        return sum(Gflops.values()) * 1e9  # Return total FLOPs (in Giga units)
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code # Used to load checkpoints from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Custom weight loading logic to handle compatibility between different model versions"""

        def check_name(src, state_dict: dict = state_dict, strict=False):  # Check if key exists in state_dict
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):  # Rename keys in state_dict to match current model structure
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

        # Handle positional encoding: resize via interpolation if dimensions mismatch
        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            # Use bicubic interpolation to resize positional encoding to match current model
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)
        # Rename keys in patch_embed module
        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):  # Rename keys in layers module for backward compatibility
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")  # Rename classification head keys
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)  # Call parent method to complete weight loading


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", SelectiveScanMode="2D", **kwargs):
        # Pass norm_layer parameter to parent class
        # print(">>> Backbone_VSSM __init__ called with kwargs:", kwargs)
        kwargs.update(norm_layer=norm_layer)
        if SelectiveScanMode == "2D":
            pass  # Default
        elif SelectiveScanMode == "3D":
            kwargs.update(_SS2D=SS3D)
        else:
            raise NotImplementedError("Set SelectiveScanMode with '2D' or '3D' ")
        super().__init__(**kwargs)
        # Determine channel order (channel_first)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )  # Define normalization layer mapping dict
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices  # Save indices of feature layers to output
        for i in out_indices:  # Add normalization layer for each output feature layer
            layer = norm_layer(self.dims[i])  # Create normalization layer with corresponding feature dimension
            layer_name = f'outnorm{i}'  # Normalization layer name, e.g., outnorm0, outnorm1, etc.
            self.add_module(layer_name, layer)  # Add normalization layer to module

            down_h = nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i], kernel_size=3, stride=(int(2/self.patch_h), int(2/self.patch_w)), padding=1, groups=self.dims[i]),  # depthwise conv
                nn.BatchNorm2d(self.dims[i], self.dims[i]),
                nn.GELU(),
                nn.Conv2d(self.dims[i], self.dims[i], kernel_size=1),  # pointwise conv
            )

            self.add_module(f'down_h{i}', down_h)











        del self.classifier  # Delete classifier defined in VSSM model, backbone does not need it
        self.load_pretrained(pretrained)  # Load pretrained weights

    def load_pretrained(self, ckpt=None, key="models"):
        """
        Load pretrained weights
        Args:
            ckpt: Path to pretrained model
            key: Key name in the weight dictionary (usually "models")
        """
        if ckpt is None:  # Don't load if no pretrained path provided
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))  # Load pretrained weights and map to CPU
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)  # Load weights, strict=False allows partial loading (ignore mismatched layers)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):  # Define helper function for layer-by-layer processing and intermediate output
            x = l.blocks(x)  # Process through all blocks of current layer
            y = l.downsample(x)  # Process through downsampling layer (if present)
            return x, y  # Return processed features and downsampled features

        x = self.patch_embed(x)  # Image patch embedding
        outs = []  # List to store output features
        # Process layer by layer
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # o is the output of current layer, x is the downsampled features
            if i in self.out_indices:  # If current layer index is in out_indices, add to output list
                norm_layer = getattr(self, f'outnorm{i}')  # Get corresponding normalization layer
                # out = norm_layer(o)  # Apply normalization
                if not self.channel_first:  # If not channel-first format, adjust dimension order to (B, C, H, W)
                    out = out.permute(0, 3, 1, 2)
                # Asymmetric downsampling (H -> H/2)
                down_h = getattr(self, f'down_h{i}')
                out = down_h(o)
                outs.append(out.contiguous())  # Add to output list
        # If no output indices specified, return the last layer's features
        if len(self.out_indices) == 0:
            return x

        return outs  # Return multi-scale feature list

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
    print(model.flops())  # wrong
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