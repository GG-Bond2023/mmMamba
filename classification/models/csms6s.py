import time
import torch
import warnings



# PyTorch cross scan implementation =============
class CrossScan(torch.autograd.Function):
    """Custom cross scan operation implementing custom forward and backward logic"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        Forward pass: generates multi-directional scan features from input tensor
        Args:
            ctx: context object for saving information needed by backward pass
            x: input tensor of shape (B, C, H, W), where B=batch size, C=channels, H=height, W=width
        Returns:
            xs: multi-directional scan features of shape (B, 4, C, H*W)
        """
        B, C, H, W = x.shape  # Get input tensor shape
        ctx.shape = (B, C, H, W)  # Save shape information for backward pass
        xs = x.new_empty((B, 4, C, H * W))  # Create output tensor, pre-allocate memory
        xs[:, 0] = x.flatten(2, 3)  # Direction 0: flatten spatial dimensions (H,W)->(H*W)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # Direction 1: transpose H and W then flatten
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # Directions 2-3: flip first two direction features
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        """
        Backward pass: compute gradient of input tensor
        Args:
            ctx: context object containing shape information from forward pass
            ys: gradient of output tensor, shape (B, 4, C, H*W)
        Returns:
            y: gradient of input tensor, shape (B, C, H, W)
        """
        B, C, H, W = ctx.shape  # Restore shape from forward pass
        L = H * W  # Length of flattened spatial dimensions
        # Merge gradients of flipped directions (directions 2-3 correspond to first two directions)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # Merge gradients of the two original directions (direction 1 needs transposition back to original dims)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)  # Reshape to input tensor shape


class CrossMerge(torch.autograd.Function):
    """Custom cross merge operation corresponding to CrossScan, merging multi-directional features"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        """
        Forward pass: merge multi-directional scan features
        Args:
            ctx: context object, saves shape information
            ys: multi-directional features, shape (B, K, D, H, W), K is number of directions
        Returns:
            y: merged feature, shape (B, D, H*W)
        """
        B, K, D, H, W = ys.shape  # Get input shape
        ctx.shape = (H, W)  # Save spatial dimensions for backward pass
        ys = ys.view(B, K, D, -1)  # Flatten spatial dimensions
        # Merge flipped direction features (directions 2-3 correspond to first two directions)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # Merge the two original direction features (direction 1 needs transposition back to original dims)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        """
        Backward pass: split merged feature gradient into multi-directional gradients
        Args:
            ctx: context object containing spatial dimension information
            x: gradient of merged feature, shape (B, D, H*W)
        Returns:
            xs: split multi-directional gradients, shape (B, 4, D, H, W)
        """
        H, W = ctx.shape  # Restore spatial dimensions
        B, C, L = x.shape  # Get input gradient shape
        xs = x.new_empty((B, 4, C, L))  # Create output gradient tensor
        xs[:, 0] = x  # Direction 0 gradient
        # Direction 1 gradient: transpose spatial dimensions then flatten
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # Directions 2-3: flip first two directions
        xs = xs.view(B, 4, C, H, W)  # Reshape to original multi-directional feature shape
        return xs


# Ablation variants below =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    """Cross scan ablation variant: uses only 2 directions (with flip)"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # Repeat input into 2 directions, then concatenate their flipped versions, total 4 directions (but only 2 base directions)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        # Merge flipped direction gradients
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # Sum over direction dimension to get input gradient
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    """Cross merge ablation variant: corresponds to 2-direction scan merge"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # Merge flipped direction features
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # Sum over direction dimension
        return ys.contiguous().sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        # Repeat input into 2 directions, then concatenate their flipped versions
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    """Cross scan ablation variant: uses only 1 direction (repeated 4 times)"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # Repeat input 4 times to simulate 4 directions (but all are the same direction)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        # Sum gradients over 4 directions
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    """Cross merge ablation variant: corresponds to 1-direction scan merge"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        # Sum features over 4 directions
        return ys.view(B, 4, -1, H * W).sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        # Repeat input 4 times, corresponding to 4 direction gradients
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)




WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda_oflex # import torch package first, then import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
    print("Can not import selective_scan_cuda_oflex. This affects speed.", flush=True)
try:
    import selective_scan_cuda_core # import torch package first, then import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda # import torch package first, then import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False



def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    """
    Check if tensor contains NaN or Inf values, used for debugging
    Args:
        tag: identifier for the tensor name/location being checked
        x: tensor to check
        enable: whether to enable checking
    """
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()  # Trigger debug breakpoint




def selective_scan_torch(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True, 
    *args,
    **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)# Save forward pass inputs and intermediate states for backward pass.
        return out# Return only scan result out, intermediate state x is used for gradient computation
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True,
    backend=None,
):
    WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA)
    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


def print_jit_input_names(inputs):
    """Print names of JIT compile-time input parameters, used for debugging"""
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):  # Print names of first 10 input parameters
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


# Reference implementation of selective scan FLOPs calculation only
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    Reference version of selective scan FLOPs computation (for comparative verification)
    Difference from flops_selective_scan_fn: uses einsum path analysis to compute FLOPs
    """
    import numpy as np  # Import numpy for einsum path analysis

    # Einsum FLOPs calculation logic adapted from fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        """Compute FLOPs for einsum operation via numpy's einsum path analysis"""
        np_arrs = [np.zeros(s) for s in input_shapes]  # Create empty arrays to simulate inputs
        # Compute optimal einsum path and extract FLOPs
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # Divide by 2 because multiply-add is counted as 1 FLOP
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex  # Does not support complex operations

    flops = 0  # Initialize FLOPs

    # Compute FLOPs for einsum: bdl,dn->bdln
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        # Grouped einsum FLOPs: bdl,bnl,bdl->bdln
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        # Ungrouped einsum FLOPs: bdl,bdnl,bdl->bdln
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    # FLOPs inside the loop
    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops  # Multiply by sequence length L
    if with_D:
        flops += B * D * L  # Add D parameter computation
    if with_Z:
        flops += B * D * L  # Add Z parameter computation
    return flops




# Cross selective scan implementation ===============================
class SelectiveScanMamba(torch.autograd.Function):
    """Selective scan operation based on base CUDA implementation, supports automatic differentiation"""

    @staticmethod
    @torch.cuda.amp.custom_fwd  # Support automatic mixed precision training
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        """
        Forward pass: calls CUDA kernel to perform selective scan
        Args:
            ctx: context object, saves information needed by backward pass
            u, delta, A, B, C, D: input tensors for selective scan
            delta_bias: bias for delta
            delta_softplus: whether to apply softplus activation to delta
            nrows, backnrows: row count parameters for parallel computation
            oflex: whether to enable oflex optimization
        Returns:
            out: output of selective scan
        """
        ctx.delta_softplus = delta_softplus  # Save whether softplus is used
        # Call CUDA forward function
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        # Save tensors needed for backward pass
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd  # Support automatic mixed precision training backward pass
    def backward(ctx, dout, *args):
        """
        Backward pass: compute gradients of input tensors
        Args:
            ctx: context object containing information saved from forward pass
            dout: gradient of output tensor
        Returns:
            gradients of input tensors
        """
        # Restore tensors saved from forward pass
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        # Ensure gradient tensor is contiguous
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # Call CUDA backward function to compute gradients
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    """Selective scan operation based on core version CUDA implementation"""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        # Call core version CUDA forward function
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # Call core version CUDA backward function
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    """Selective scan operation based on oflex version CUDA implementation"""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        # Call oflex version CUDA forward function
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # Call oflex version CUDA backward function
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)



def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    flops_fn = flops_selective_scan_ref if backend == "naive" else flops_selective_scan_fn
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


if __name__ == "__main__":
    def params(B, K, C, N, L, device = torch.device("cuda"), itype = torch.float):
        As = (-0.5 * torch.rand(K * C, N, device=device, dtype=torch.float32)).requires_grad_()
        Bs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
        Cs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
        Ds = torch.randn((K * C), device=device, dtype=torch.float32).requires_grad_()
        u = torch.randn((B, K * C, L), device=device, dtype=itype).requires_grad_()
        delta = (0.5 * torch.rand((B, K * C, L),  device=device, dtype=itype)).requires_grad_()
        delta_bias = (0.5 * torch.rand((K * C), device=device, dtype=torch.float32)).requires_grad_()
        return u, delta, As, Bs, Cs, Ds, delta_bias

    def bench(func, xs, Warmup=30, NTimes=20):
        import time
        torch.cuda.synchronize()
        for r in range(Warmup):
            for x in xs:
                func(x)
        torch.cuda.synchronize()
        tim0 = time.time()
        for r in range(NTimes):
            for x in xs:
                func(x)
        torch.cuda.synchronize()
        return (time.time() - tim0) / NTimes

    def check():
        u, delta, As, Bs, Cs, Ds, delta_bias = params(1, 4, 16, 8, 512, itype=torch.float16)
        u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1 = [x.clone().detach().requires_grad_() for x in [u, delta, As, Bs, Cs, Ds, delta_bias]]
        
        # out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="torch")
        out = selective_scan_fn(u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1, True, backend="oflex")
        out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="mamba")
        print((out_ref - out).abs().max())
        out.sum().backward()
        out_ref.sum().backward()
        for x, y in zip([u, As, Bs, Cs, Ds, delta, delta_bias], [u1, As1, Bs1, Cs1, Ds1, delta1, delta_bias1]):
            print((x.grad - y.grad).abs().max())

        u, delta, As, Bs, Cs, Ds, delta_bias = params(128, 4, 96, 8, 56 * 56)
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="oflex"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="mamba"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="torch"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))

    check()

