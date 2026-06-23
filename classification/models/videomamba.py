# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os  # OS interaction (paths, environment variables, etc.)
import torch  # PyTorch main package
import torch.nn as nn  # Neural network module shorthand
from functools import partial  # For fixing partial function parameters
from torch import Tensor  # Type hint: Tensor
from typing import Optional  # Type hint: Optional
import torch.utils.checkpoint as checkpoint  # Gradient checkpointing, saves memory
from einops import rearrange, repeat  # einops for elegant tensor rearrangement and repetition

from einops import rearrange  # Re-import (redundant, but harmless)
from timm.models.vision_transformer import _cfg  # Default config function/structure for ViT in timm
from timm.models.registry import register_model  # Model registry from timm
from timm.models.layers import trunc_normal_  # Truncated normal initialization

from timm.models.layers import DropPath, to_2tuple  # DropPath (stochastic depth), to_2tuple helper
from timm.models.vision_transformer import _load_weights  # timm weight loading utility

import math  # Math functions

from mamba_ssm.modules.mamba_simple import Mamba  # SSM / Mamba module (custom)
from models.csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex  # Custom scan operation implementations

# Try to import Triton-based high-performance layernorm implementation, fall back to None if unavailable
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

MODEL_PATH = 'your_model_path'  # Model file base path (must be replaced with actual path)
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}  # Pretrained model file path dictionary


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        # Create a linear layer to project dt parameters to d_inner dimension, bias will store inverse-softplus initial values
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale  # Standard deviation for weight initialization (related to input rank)
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)  # Constant weight initialization
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)  # Uniform random weight initialization
        else:
            raise NotImplementedError  # Unknown initialization method error

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  # Sample in log space [dt_min, dt_max], exp, then clamp to prevent being too small
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse function of softplus, used to get initial bias value
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)  # Write computed bias into linear.bias (gradient-blocked)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True  # Can be enabled if marking as not-to-be-reset by framework

        return dt_proj  # Return initialized projection

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization (logarithmic representation of SSM A diagonal terms)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()  # Generate matrix of shape (d_inner, d_state), each column is 1..d_state, repeated d_inner times
        A_log = torch.log(A)  # Store as logarithmic form (fp32)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # If multiple copies (k_group) are needed, add a leading dimension
            if merge:
                A_log = A_log.flatten(0, 1)  # Merge first two dimensions to become (copies * d_inner, d_state)
        A_log = nn.Parameter(A_log)  # Convert to trainable parameter (may not be optimized subsequently)
        A_log._no_weight_decay = True  # Mark as not applying weight decay
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter, initialized to ones (commonly used for skip/residual parameters in SSM)
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)  # Repeat copies times
            if merge:
                D = D.flatten(0, 1)  # Merge into a 1D vector
        D = nn.Parameter(D)  # Convert to trainable parameter (keep fp32)
        D._no_weight_decay = True  # Do not apply weight decay
        return D


class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type=None,
                 k_group=None,
                 dim=None,
                 dt_rank=None,
                 d_inner=None,
                 d_state=None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()  # Initialize parent class
        act_layer = nn.SiLU  # Activation function class (SwiGLU/SiLU)
        dt_min = 0.001  # dt lower bound
        dt_max = 0.1  # dt upper bound
        dt_init = "random"  # dt initialization mode
        dt_scale = 1.0  # dt scaling
        dt_init_floor = 1e-4  # Minimum threshold for dt initialization
        bias = False  # Linear layers default to no bias (except some layers)
        self.force_fp32 = force_fp32  # Whether to force conversion to fp32 (for numerical stability)
        self.seq = seq  # Sequence mode flag (temporarily unused)
        self.k_group = k_group  # Number of groups (e.g., different scan/direction counts)
        self.group_type = group_type  # Group type (e.g., 'Cube')
        self.scan_type = scan_type  # Scan type (multiple strategies)

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        # Input projection: project dim to d_inner*2, used to split into x and gate/z parts (similar to GLU)
        self.act: nn.Module = act_layer()  # Activation instance
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=(1, 1, 1), **kwargs,
        )  # Depthwise separable 3D convolution (groups=d_inner means independent convolution per channel)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]  # Create a linear layer for each group, output contains dt_rank + 2*d_state (to be split into dt, B, C later)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj  # Delete original list, but keep weight tensors to save memory

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]  # Create dt projection for each group (using dt_init above)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs  # Delete list, only keep merged weight and bias parameters

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)  # Output normalization (LayerNorm)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)  # Project d_inner back to original dim
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()  # Optional dropout

    def flatten_spectral_spatial(self, x):
        x = rearrange(x, 'b c t h w -> b c (h w) t')  # Merge (h,w) into one dimension, order becomes (b, c, n_spatial, t)
        x = rearrange(x, 'b c n m -> b c (n m)')  # Then merge n and m into a single time/sequence dimension, yielding (b, c, L)
        return x  # Return flattened tensor

    def flatten_spatial_spectral(self, x):
        x = rearrange(x, 'b c t h w -> b c t (h w)')  # First merge spatial into last dimension (h*w)
        x = rearrange(x, 'b c n m -> b c (n m)')  # Then merge t with the above, yielding (b, c, L)
        return x

    def reshape_spectral_spatial(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()  # Swap channel and sequence axes, ensure memory contiguity
        y = y.view(B, H * W, T, -1)  # Reshape to (B, H*W, T, channels')
        y = rearrange(y, 'b o t c -> b t o c')  # Rearrange to (B, T, H*W, channels')
        y = y.view(B, T, H, W, -1)  # Finally restore to (B, T, H, W, channels')
        return y

    def reshape_spatial_spectral(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()  # Swap dimensions, ensure contiguity
        y = y.view(B, T, H, W, -1)  # Directly reshape to (B, T, H, W, channels')
        return y

    def scan(self, x, scan_type=None, group_type=None):
        # Select different flattening and mirroring combination strategies based on scan_type, return stacked scan versions
        if scan_type == 'Spectral-priority':
            x = self.flatten_spectral_spatial(x)  # Spectral priority
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)  # forward + reversed (flip along last sequence dimension)
        elif scan_type == 'Spatial-priority':
            x = self.flatten_spatial_spectral(x)  # Spatial priority
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Cross spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spa, dims=[-1])], dim=1)  # Cross: spe forward + spa reverse
        elif scan_type == 'Cross spatial-spectral':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spa, torch.flip(x_spe, dims=[-1])], dim=1)
        elif scan_type == 'Parallel spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            # parallel: spe forward, spe reverse, spa forward, spa reverse -> 4 branches
            xs = torch.stack([x_spe, torch.flip(x_spe, dims=[-1]), x_spa, torch.flip(x_spa, dims=[-1])], dim=1)
        return xs  # Return shape (B, K, channels, L) or similar stacked tensor

    def forward(self, x: Tensor, SelectiveScan=SelectiveScanMamba):
        x = self.in_proj(x)  # Input projection -> project dim to d_inner*2 (for splitting into gate/main channel)
        x, z = x.chunk(2, dim=-1)  # Split equally along last dimension into x (main) and z (gate/scaling factor)
        z = self.act(z)  # Apply activation (SiLU) to z, obtaining scaling factor

        # forward conv1d
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # Rearrange to (B, channels, T, H, W) for Conv3d use
        x = self.conv3d(x)  # Per-channel 3D convolution (groups=d_inner), channel count unchanged
        x = self.act(x)  # Activation

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            # Wrapper for SelectiveScan.apply call, simplifying subsequent calls
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, T, H, W = x.shape  # Unpack current x shape (note D here represents channel)
        L = T * H * W  # Sequence length (time * space)
        D, N = self.A_logs.shape  # A_logs shape (D, N), D = k_group * d_inner (or similar), N = d_state
        K, D, R = self.dt_projs_weight.shape  # dt_projs_weight shape (K, inner, rank)

        # scan
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type)  # Obtain stacked scans, shape (B, K, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # Use einsum to linearly map xs with x_proj_weight, output (B, K, out_dim, L)
        # out_dim = dt_rank + 2 * d_state, to be split into dts, Bs, Cs later
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)  # Add bias if available

        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # Split along channel dimension into dt (R), B (N), C (N)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        # Project dts from rank space through dt_projs_weight to d_inner (or D) space -> shape (B, K, D, L)

        xs = xs.view(B, -1, L)  # Flatten xs merging K and chan, yielding (B, K*C, L)
        dts = dts.contiguous().view(B, -1, L)  # Also flatten dts to (B, K*D, L), ensure memory contiguity
        Bs = Bs.contiguous()  # Ensure memory contiguity (may be used for subsequent Apply)
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())  # A_logs stored in logarithmic form, take exp and negate for stability (negative real part)
        Ds = self.Ds.float()  # Convert D parameter to float
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # Flatten dt proj bias to 1D vector

        # Some assertions (commented out), usable for debugging shape correctness
        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)  # Small helper: convert multiple tensors to fp32

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)  # Force these scan tensors to fp32 (numerical stability)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  # Call SelectiveScan, returns (B, K * D, L) -> reshape to (B, K, D, L)
        assert out_y.dtype == torch.float  # Ensure output is float (fp32)

        if self.group_type == 'Cube':  # Merge logic for group_type 'Cube'
            if self.scan_type == 'Spectral-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # Sum forward and reverse branches
                y = self.reshape_spectral_spatial(y, B, H, W, T)  # Restore to (B, T, H, W, C)
                y = self.out_norm(y)  # LayerNorm
                ###### equal to above ####
                # y_fwd = out_y[:, 0] ...
                # Commented-out code provides equivalent implementation for readability
            elif self.scan_type == 'Spatial-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                y = self.reshape_spatial_spectral(y, B, H, W, T)
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spectral-spatial':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spectral_spatial(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spatial_spectral(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spatial-spectral':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spatial_spectral(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spectral_spatial(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Parallel spectral-spatial':
                ye = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                ye = self.reshape_spectral_spatial(ye, B, H, W, T)
                ya = out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1])
                ya = self.reshape_spatial_spectral(ya, B, H, W, T)
                y = ye + ya
                y = self.out_norm(y)

        y = y * z  # Element-wise scaling of y by previously computed gate/z (similar to attention gating)
        out = self.dropout(self.out_proj(y))  # Output projection back to original dim and apply dropout

        return out  # Return the output of this Block

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        # Placeholder function: may be used for allocating cache during inference (depends on self.mixer, mixer definition not shown in code)


class VisionMamba(nn.Module):
    def __init__(
            self,
            group_type=None,
            # k_group=None,
            depth=None,
            embed_dim=None,
            dt_rank: int = None,
            d_inner: int = None,
            d_state: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            # video
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls=False,
            scan_type=None,
            conv3D_channel: int = None,
            conv3D_kernel: int = None,
            **kwargs,
    ):
        super().__init__()  # Initialize parent class
        self.residual_in_fp32 = residual_in_fp32  # Whether residual part uses fp32
        self.fused_add_norm = fused_add_norm  # Whether to use fused add+norm (performance optimization)
        self.use_checkpoint = use_checkpoint  # Whether to use checkpoint (memory saving)
        self.checkpoint_num = checkpoint_num  # Checkpoint config (reserved)
        self.Pos_Cls = Pos_Cls  # Whether to use position or classification token (reserved)
        self.scan_type = scan_type  # Scan method
        self.d_model = self.num_features = self.embed_dim = embed_dim  # Unified model embedding dimension name

        if scan_type =="Parallel spectral-spatial":
            self.k_group = 4  # Number of groups
        else:
            raise NotImplementedError("scan_type currently supports 'Parallel spectral-spatial' only.")

        self.group_type = group_type  # Group type

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )  # Video front-end: 1-channel input (e.g., grayscale or single-channel feature), Conv3d -> BN -> ReLU

        self.embedding_spatial_spectral = nn.Sequential(nn.Linear(conv3D_channel, embed_dim))
        # Linear layer maps conv3d's final channel count to embed_dim, as feature vector for each spatial position

        self.norm = nn.LayerNorm(embed_dim)  # LayerNorm for block input normalization
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling to 1x1 (spatial average)
        self.flatten = nn.Flatten(1)  # Flatten starting from dimension 1 (preserve batch dim)

        self.pos_drop = nn.Dropout(p=drop_rate)  # Input dropout (position/patch dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth drop rate per layer
        inter_dpr = [0.0] + dpr  # Can be used for layerwise drop rate (reserved here)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # DropPath applies stochastic depth to residuals

        # mamba blocks
        self.layers = nn.ModuleList(
            [
                Block(
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=self.k_group,
                    dim=embed_dim,
                    d_state=d_state,
                    d_inner=d_inner,
                    dt_rank=dt_rank,
                    bimamba=bimamba,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )  # Create a layer sequence of depth Blocks

    def forward_features(self, x, inference_params=None):
        x = x.unsqueeze(1)# # conv3d_features requires x to be (B, channels, FFT, H, W) for Conv3d use
        x = self.conv3d_features(x)  # Front-end Conv3d+BN+ReLU: input e.g. (B,1,T,H,W) -> (B, C, T', H', W')
        # scan
        x = rearrange(x, 'b c t h w -> b t h w c')  # Transform to (B, T, H, W, C), convenient for spatial position linear embedding
        x = self.embedding_spatial_spectral(x)  # Linear mapping for each (T,H,W) position -> obtain embed_dim features
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1, 1, 64]  stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_token, x), dim=1)  ##[10, 1793, 64]
        # x = x + self.pos_embed
        x = self.pos_drop(x)  # Apply positional dropout

        # mamba impl
        for idx, layer in enumerate(self.layers):  ## Iterate through each Block
            # Residual connection + stochastic depth + block output

            # x = x + self.drop_path(layer(self.norm(x)))

            feature = self.norm(x)
            feature = layer(feature)
            feature = self.drop_path(feature)
            feature = feature + x
        # First permute back to Conv3d style (B, C, T, H, W), then avgpool -> flatten -> return feature vector
        # return self.flatten(self.avgpool(x.permute(0, 4, 1, 2, 3)).mean(dim=2))  # Final output shape (B, C)
        return self.flatten(self.avgpool(feature.permute(0, 4, 1, 2, 3)).mean(dim=2))  # Final output shape (B, C)

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  # Extract feature vector
        return x, feature  # Return (logits, feature)
