# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.


@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Analyzing and Improving the Training Dynamics of Diffusion Models".


def pixel_norm(x: torch.Tensor, eps: float = 1e-4, dim=1) -> torch.Tensor:
    return x / (torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps))


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) / 0.596


def mp_add(a: torch.Tensor, b: torch.Tensor, t: float = 0.3) -> torch.Tensor:
    scale = np.sqrt(t**2 + (1 - t) ** 2, dtype=np.float32)
    return ((1 - t) * a + t * b) / scale


def mp_sum(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    normalized_weights = np.array(weights, dtype=np.float32) / np.sum(
        weights, dtype=np.float32
    )
    scale = np.sqrt(np.sum(normalized_weights**2, dtype=np.float32), dtype=np.float32)
    res = sum([w * t for w, t in zip(normalized_weights, tensors)]) / scale
    return res


def mp_cat(a: torch.Tensor, b: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    N_a, N_b = a[0].numel(), b[0].numel()
    scale = np.sqrt((N_a + N_b) / (t**2 + (1 - t) ** 2), dtype=np.float32)
    out = torch.cat(
        [
            (1 - t) / np.sqrt(N_a, dtype=np.float32) * a,
            t / np.sqrt(N_b, dtype=np.float32) * b,
        ],
        dim=1,
    )
    return out * scale


# Weight normalization from Karras Paper


def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


@persistence.persistent_class
class KarrasConv2d(torch.nn.Conv2d):
    def __init__(self, C_in, C_out, k):
        super().__init__(C_in, C_out, k, bias=False)
        torch.nn.init.normal_(self.weight, mean=0, std=1)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / np.sqrt(fan_in)
        x = torch.nn.functional.conv2d(x, w.to(x.dtype), padding="same")
        return x


@persistence.persistent_class
class KarrasLinear(torch.nn.Linear):
    def __init__(self, C_in, C_out):
        super().__init__(C_in, C_out, bias=False)
        torch.nn.init.normal_(self.weight, mean=0, std=1)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / np.sqrt(fan_in)
        x = torch.nn.functional.linear(x, w.to(x.dtype))
        return x


@persistence.persistent_class
class KarrasFourierEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.randn(embedding_dim))
        self.register_buffer("phases", torch.rand(embedding_dim))

    def forward(self, x):
        x = torch.outer(x.flatten(), self.freqs) + self.phases
        x = torch.cos(2 * torch.pi * x) * np.sqrt(2, dtype=np.float32)
        return x


@persistence.persistent_class
class KarrasClassEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.linear = KarrasLinear(num_embeddings, embedding_dim)

    def forward(self, class_labels: torch.Tensor):
        class_emb = torch.nn.functional.one_hot(
            class_labels.flatten(), self.num_embeddings
        )
        return self.linear(class_emb * np.sqrt(self.num_embeddings, dtype=np.float32))


@persistence.persistent_class
class KarrasAugmentEmbedding(torch.nn.Module):
    def __init__(self, augment_dim: int, embedding_dim: int):
        super().__init__()
        self.linear = KarrasLinear(augment_dim, embedding_dim)

    def forward(self, augment_labels: torch.Tensor):
        return self.linear(augment_labels)


@persistence.persistent_class
class KarrasEmbedding(torch.nn.Module):
    def __init__(
        self,
        fourier_dim: int,
        embedding_dim: int,
        num_classes: int = 0,
        augment_dim: int = 0,
        add_factor: float = 0.5,
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.add_factor = add_factor
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.augment_dim = augment_dim
        self.fourier_embed = KarrasFourierEmbedding(fourier_dim)
        self.sigma_embed = KarrasLinear(fourier_dim, embedding_dim)
        self.class_embed = None
        if num_classes != 0:
            self.class_embed = KarrasClassEmbedding(num_classes, embedding_dim)
        self.augment_embed = None
        if augment_dim != 0:
            self.augment_embed = KarrasAugmentEmbedding(augment_dim, embedding_dim)

    def forward(self, c_noise, class_labels=None, augment_labels=None):
        embedding = self.fourier_embed(c_noise)
        embedding = self.sigma_embed(embedding)

        embeddings = [embedding]
        if class_labels is not None:
            if self.class_embed is None:
                raise ValueError("class_labels is not None, but num_classes is None. ")
            class_embedding = self.class_embed(class_labels)
            embeddings.append(class_embedding)

        if augment_labels is not None:
            if self.augment_embed is None:
                raise ValueError(
                    "augment_labels is not None, but augment_dim is None. "
                )
            augment_embedding = self.augment_embed(augment_labels)
            embeddings.append(augment_embedding)

        embedding = mp_sum(embeddings, [1] * len(embeddings))
        out = mp_silu(embedding)
        return out


@persistence.persistent_class
class KarrasCosineAttention(torch.nn.Module):
    def __init__(self, embed_dimension: int, head_dim: int = 64):
        super().__init__()
        assert embed_dimension % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = embed_dimension // head_dim
        self.embed_dimension = embed_dimension
        self.c_attn = KarrasConv2d(embed_dimension, 3 * embed_dimension, 1)
        self.c_proj = KarrasConv2d(embed_dimension, embed_dimension, 1)

    def forward(self, x):
        input = x
        b, c, h, w = x.shape
        x_proj = self.c_attn(x)  # (b, c, h, w) -> (b, 3*c, h, w)
        x_proj = x_proj.view(b, -1, h * w).transpose(1, 2)  # (b, h*w, 3*c)

        q, k, v = x_proj.chunk(3, -1)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = pixel_norm(q), pixel_norm(k), pixel_norm(v)

        res = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # (b, num_heads, h*w, head_dim)

        res = res.transpose(-1, -2).reshape(b, -1, h, w)
        res = self.c_proj(res)

        out = mp_add(input, res)
        return out


@persistence.persistent_class
class Upsample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


@persistence.persistent_class
class KarrasEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        down: bool = False,
        attention: bool = False,
        head_dim: int = 64,
        dropout_rate: float = 0.0,
        add_factor: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.add_factor = add_factor

        self.resample = (
            torch.nn.AvgPool2d(kernel_size=2, stride=2) if down else torch.nn.Identity()
        )

        self.conv_1x1 = (
            KarrasConv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

        self.conv_3x3_1 = KarrasConv2d(out_channels, out_channels, 3)
        self.conv_3x3_2 = KarrasConv2d(out_channels, out_channels, 3)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.attention = (
            KarrasCosineAttention(out_channels, head_dim)
            if attention
            else torch.nn.Identity()
        )

        # embedding layer
        self.embed = KarrasLinear(embedding_dim, out_channels)
        self.gain = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = self.resample(input)
        x = self.conv_1x1(x)
        x = pixel_norm(x)

        # Residual branch
        res = x
        res = mp_silu(res)
        res = self.conv_3x3_1(res)
        res = res * (self.embed(embedding) * self.gain.to(x.dtype) + 1).unsqueeze(
            -1
        ).unsqueeze(-1)
        res = mp_silu(res)
        res = self.dropout(res)
        res = self.conv_3x3_2(res)

        out = mp_add(x, res, self.add_factor)
        out = self.attention(out)
        return out


@persistence.persistent_class
class KarrasDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        up: bool = False,
        attention: bool = False,
        head_dim: int = 64,
        skip_channels: int = 0,
        dropout_rate: float = 0.0,
        add_factor: float = 0.3,
        cat_factor: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.skip_channels = skip_channels
        self.add_factor = add_factor
        self.cat_factor = cat_factor

        self.resample = Upsample() if up else torch.nn.Identity()

        total_input_channels = in_channels + skip_channels
        self.conv_1x1 = (
            KarrasConv2d(total_input_channels, out_channels, 1)
            if total_input_channels != out_channels
            else torch.nn.Identity()
        )

        self.conv_3x3_1 = KarrasConv2d(total_input_channels, out_channels, 3)
        self.conv_3x3_2 = KarrasConv2d(out_channels, out_channels, 3)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.attention = (
            KarrasCosineAttention(out_channels, head_dim)
            if attention
            else torch.nn.Identity()
        )

        # embedding layer
        self.embed = KarrasLinear(embedding_dim, out_channels)
        self.gain = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input: torch.Tensor,
        embedding: torch.torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            input = mp_cat(input, skip, self.cat_factor)
        x = self.resample(input)
        res = x
        x = self.conv_1x1(x)

        res = mp_silu(res)
        res = self.conv_3x3_1(res)

        res = res * (self.embed(embedding) * self.gain.to(x.dtype) + 1).unsqueeze(
            -1
        ).unsqueeze(-1)
        res = mp_silu(res)
        res = self.dropout(res)
        res = self.conv_3x3_2(res)

        out = mp_add(x, res, self.add_factor)
        out = self.attention(out)
        return out


@persistence.persistent_class
class KarrasUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        emb_channels = model_channels * channel_mult_emb
        self.embedding = KarrasEmbedding(
            fourier_dim=model_channels,
            embedding_dim=emb_channels,
            num_classes=label_dim,
            augment_dim=augment_dim,
        )
        block_kwargs = dict(
            embedding_dim=emb_channels,
            head_dim=64,
            dropout_rate=dropout,
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 1
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = KarrasConv2d(cin, cout, 3)
            else:
                self.enc[f"{res}x{res}_down"] = KarrasEncoderBlock(
                    cout, cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block_{idx}"] = KarrasEncoderBlock(
                    cin, cout, attention=(res in attn_resolutions), **block_kwargs
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = KarrasDecoderBlock(
                    cout, cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = KarrasDecoderBlock(
                    cout, cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = KarrasDecoderBlock(
                    cout, cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block_{idx}"] = KarrasDecoderBlock(
                    cin,
                    cout,
                    skip_channels=skips.pop(),
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )

        self.conv_out = KarrasConv2d(cout, out_channels, 3)
        self.gain_out = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        dtype = x.dtype
        emb = self.embedding(noise_labels, class_labels, augment_labels).to(dtype)
        skips = []
        ones_tensor = torch.ones_like(x[:, 0:1, :, :])
        x = torch.cat((x, ones_tensor), dim=1)
        # Encoder
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, KarrasEncoderBlock) else block(x)
            skips.append(x)
        # Decoder
        for block in self.dec.values():
            if block.skip_channels != 0:
                skip = skips.pop()
                x = block(x, emb, skip)
            else:
                x = block(x, emb)
        x = self.conv_out(x) * self.gain_out
        return x.to(dtype)


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[
            1,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


# ----------------------------------------------------------------------------
# Uncertainty-based loss weighting according to the paper Analyzing and
# Improving the Training Dynamics of Diffusion Models


class UncertaintyMLP(torch.nn.Module):
    def __init__(
        self,
        num_channels,
    ):
        super().__init__()
        self.map_noise = KarrasFourierEmbedding(embedding_dim=num_channels)
        self.mlp = KarrasLinear(num_channels, 1)

    def forward(self, noise_labels):
        emb = self.map_noise(noise_labels)
        u_sigma = self.mlp(emb)
        return u_sigma


# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion


@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        dtype="fp32",  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.dtype = dtype
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.dtype == "fp16" and not force_fp32 and x.device.type == "cuda")
            else (
                torch.bfloat16
                if (self.dtype == "bf16" and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        dtype="fp32",  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
        model_type="SongUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.dtype = dtype
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.dtype == "fp16" and not force_fp32 and x.device.type == "cuda")
            else (
                torch.bfloat16
                if (self.dtype == "bf16" and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".


@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        dtype="fp32",  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.dtype = dtype
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.dtype == "fp16" and not force_fp32 and x.device.type == "cuda")
            else (
                torch.bfloat16
                if (self.dtype == "bf16" and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        dtype="fp32",  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        uncertainty=False,  # Use uncertainty-based loss weighting?
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.dtype = dtype
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )
        if uncertainty:
            self.uncertainty = UncertaintyMLP(
                num_channels=model_kwargs["model_channels"]
            )

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        force_fp32=False,
        return_uncertainty=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.dtype == "fp16" and not force_fp32 and x.device.type == "cuda")
            else (
                torch.bfloat16
                if (self.dtype == "bf16" and not force_fp32 and x.device.type == "cuda")
                else torch.float32
            )
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        if return_uncertainty:
            u_sigma = self.uncertainty(c_noise.flatten())
            return D_x, u_sigma
        else:
            return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
