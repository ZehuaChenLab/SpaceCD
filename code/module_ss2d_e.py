"""
修改 dts -> dt + dt_add_map
prior -> Dw3*3 -> SiLU -> Pw1x1 -> LN -> tanh -> dt_add_map
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class LayerNorm2d(nn.Module):
    def __init__(self, n_channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x):
        # x: (B,C,H,W)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv_encoder=5,
        d_conv_decoder=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.1,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        decoder=None,
        prior=True,  # True: 启用先验注入；False: 完全不使用 d
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if selective_scan_fn is None:
            raise RuntimeError("selective_scan_fn is None")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv_encoder = d_conv_encoder
        self.d_conv_decoder = d_conv_decoder
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.decoder = decoder
        self.prior_use = prior

        # -------------------------
        # SS2D main branch
        # -------------------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # depthwise conv on x_inner
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=(3 - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # -------------------------
        # ✅ A：per-direction prior -> dt_add_map
        # 输出 (B, 4*d_inner, H, W) -> reshape (B,4,d_inner,H,W)
        # -------------------------
        if self.prior_use:
            self.prior_dt = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=3,
                    padding=1,
                    groups=self.d_model,
                    bias=bias,
                    **factory_kwargs,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=self.d_model,
                    out_channels=4 * self.d_inner,  # ✅ per-direction
                    kernel_size=1,
                    bias=bias,
                    **factory_kwargs,
                ),
                # 加一个LN
                # LayerNorm2d(4 * self.d_inner),
            )
            # ✅ 每个方向、每个通道一个可学习的缩放（初始为0：先验初期不影响）
            self.alpha_dt = nn.Parameter(
                torch.zeros(4, self.d_inner, 1, 1, device=device, dtype=torch.float32), requires_grad=True
            )

        # x_proj weights (K=4)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # dt_projs weights (K=4)
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))      # (K=4, inner)
        del self.dt_projs

        # SSM params
        # ✅ 传 device，避免 A_logs/Ds 初始化在 CPU 导致 device mismatch
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, device=device, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, device=device, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None

    # -------------------------
    # init helpers
    # -------------------------
    @staticmethod
    def dt_init(
        dt_rank, d_inner, dt_scale=1.0, dt_init="random",
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
        **factory_kwargs
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # -------------------------
    # core scan
    # -------------------------
    def forward_core(self, x: torch.Tensor, dt_add_map: Optional[torch.Tensor] = None):
        """
        x: (B, d_inner, H, W)
        dt_add_map: (B, 4, d_inner, H, W)  # ✅ A：每方向一张 dt 增量图（可为 None）
        return: y1,y2,y3,y4 each shaped (B, d_inner, L) where L=H*W
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B,4,d_inner,L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # (B,K,dt_rank,L) -> (B,K,d_inner,L)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # ✅ A：加入 per-direction dt_add_map
        if dt_add_map is not None:
            if dt_add_map.shape != (B, K, self.d_inner, H, W):
                raise ValueError(
                    f"dt_add_map shape mismatch: got {tuple(dt_add_map.shape)}, "
                    f"expect {(B, K, self.d_inner, H, W)}"
                )
            dt_add = dt_add_map.view(B, K, self.d_inner, L)  # (B,K,d_inner,L)
            dts = dts + dt_add

        # flatten for selective_scan
        xs = xs.float().view(B, -1, L)                         # (B, K*d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L)          # (B, K*d_inner, L)
        Bs = Bs.float().view(B, K, -1, L)                      # (B, K, d_state, L)
        Cs = Cs.float().view(B, K, -1, L)                      # (B, K, d_state, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)     # (K*d_inner)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor] = None):
        """
        x: (B,H,W,d_model)
        d: (B,H,W,d_model) 仅用于 prior 条件化（prior_use=True 时需要）
        """
        B, H, W, C = x.shape

        # in_proj -> split
        xz = self.in_proj(x)                   # (B,H,W,2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)       # (B,H,W,d_inner), (B,H,W,d_inner)

        # conv branch
        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()  # (B,d_inner,H,W)
        x_inner = self.act(self.conv2d(x_inner))            # (B,d_inner,H,W)

        # ✅ A：生成 per-direction dt_add_map（可为 None）
        dt_add_map = None
        if self.prior_use:
            if d is None:
                raise ValueError("prior_use=True 时 forward 必须传入 d (B,H,W,d_model)，仅用于 dt 条件化")
            d = d.permute(0, 3, 1, 2).contiguous()          # (B,d_model,H,W)

            dt_add_map = self.prior_dt(d)                   # (B, 4*d_inner, H, W)
            dt_add_map = dt_add_map.view(B, 4, self.d_inner, H, W)  # (B,4,d_inner,H,W)
            dt_add_map = torch.tanh(dt_add_map) * self.alpha_dt     # broadcast (4,d_inner,1,1)

        # scan core（注入 dt）
        y1, y2, y3, y4 = self.forward_core(x_inner, dt_add_map=dt_add_map)
        y = y1 + y2 + y3 + y4                                # (B,d_inner,L)

        # to (B,H,W,d_inner)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        gate = F.silu(z)  # (B,H,W,d_inner)

        # 仅保留标准 y*gate（不注入 prior feature）
        y = y * gate

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    

if __name__ == '__main__':
    x = torch.randn(2, 16, 16, 64).cuda()
    d = torch.randn(2, 16, 16, 64).cuda()

    model = SS2D(d_model=64, prior=True).cuda()

    y = model(x, d=d)
    print(y.shape)