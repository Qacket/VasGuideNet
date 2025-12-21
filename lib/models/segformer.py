# lib/models/segformer.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List

# ------------------------------
# 3D Building Blocks
# ------------------------------

class OverlapPatchEmbeddings3D(nn.Module):
    """
    3D overlap patch embedding:
    Conv3d -> [B, C, D', H', W'] -> flatten to [B, N, C] then LayerNorm(C)
    """
    def __init__(self, in_ch: int, dim: int, kernel_size: Tuple[int, int, int], stride: Tuple[int, int, int], padding: Tuple[int, int, int]):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, dim, D', H', W']
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, d, h, w


class DWConv3d(nn.Module):
    """
    Depth-wise 3D conv used inside MixFFN3D.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape
        x = x.transpose(1, 2).contiguous().view(b, c, d, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class MixFFN3D(nn.Module):
    """
    SegFormer-style MixFFN for 3D: Linear -> DWConv3d -> GELU -> Linear
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv3d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, d, h, w)
        x = self.act(x)
        x = self.fc2(x)
        return x


class EfficientSelfAtten3D(nn.Module):
    """
    Efficient self-attention in 3D: Q from full tokens; K,V from spatially reduced tokens via Conv3d(stride=reduction_ratio).
    """
    def __init__(self, dim: int, heads: int, reduction_ratio: int):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.reduction_ratio = reduction_ratio

        if reduction_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.heads, c // self.heads).permute(0, 2, 1, 3)  # [B, heads, N, d_k]

        if self.reduction_ratio > 1:
            xt = x.transpose(1, 2).contiguous().view(b, c, d, h, w)  # [B, C, D, H, W]
            xs = self.sr(xt)  # [B, C, D', H', W']
            bs, cs, ds, hs, ws = xs.shape
            xs = xs.flatten(2).transpose(1, 2).contiguous()  # [B, N_s, C]
            xs = self.norm(xs)
        else:
            xs = x

        kv = self.kv(xs).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)  # [2, B, heads, N_s, d_k]
        k, v = kv[0], kv[1]  # [B, heads, N_s, d_k]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N_s]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)  # [B, N, C]
        out = self.proj(out)
        return out


class TransformerBlock3D(nn.Module):
    """
    One transformer block: LN -> EfficientSelfAtten3D -> residual; LN -> MixFFN3D -> residual
    """
    def __init__(self, dim: int, heads: int, reduction_ratio: int, ffn_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten3D(dim, heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN3D(dim, dim * ffn_ratio)

    def forward(self, x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), d, h, w)
        x = x + self.mlp(self.norm2(x), d, h, w)
        return x


class MiT3D(nn.Module):
    """
    3D Mix Transformer encoder with 4 stages.
    Each stage: overlap patch embedding (3D) -> multiple TransformerBlock3D
    """
    def __init__(self, in_channels: int, dims: List[int], layers: List[int], heads: List[int], reduction: List[int]):
        super().__init__()
        # Patch Embedding configs (all stride=2 to keep decode manageable)
        ks = [(7, 7, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        st = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        pd = [(3, 3, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]

        self.patch_embed1 = OverlapPatchEmbeddings3D(in_channels, dims[0], ks[0], st[0], pd[0])
        self.block1 = nn.ModuleList([TransformerBlock3D(dims[0], heads[0], reduction[0]) for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.patch_embed2 = OverlapPatchEmbeddings3D(dims[0], dims[1], ks[1], st[1], pd[1])
        self.block2 = nn.ModuleList([TransformerBlock3D(dims[1], heads[1], reduction[1]) for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.patch_embed3 = OverlapPatchEmbeddings3D(dims[1], dims[2], ks[2], st[2], pd[2])
        self.block3 = nn.ModuleList([TransformerBlock3D(dims[2], heads[2], reduction[2]) for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.patch_embed4 = OverlapPatchEmbeddings3D(dims[2], dims[3], ks[3], st[3], pd[3])
        self.block4 = nn.ModuleList([TransformerBlock3D(dims[3], heads[3], reduction[3]) for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, C, D, H, W]
        outs = []

        x, d1, h1, w1 = self.patch_embed1(x)  # [B, N1, C1]
        for blk in self.block1:
            x = blk(x, d1, h1, w1)
        x1 = self.norm1(x).transpose(1, 2).contiguous().view(-1, x.shape[-1], d1, h1, w1)  # [B, C1, D1, H1, W1]
        outs.append(x1)

        x, d2, h2, w2 = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, d2, h2, w2)
        x2 = self.norm2(x).transpose(1, 2).contiguous().view(-1, x.shape[-1], d2, h2, w2)
        outs.append(x2)

        x, d3, h3, w3 = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, d3, h3, w3)
        x3 = self.norm3(x).transpose(1, 2).contiguous().view(-1, x.shape[-1], d3, h3, w3)
        outs.append(x3)

        x, d4, h4, w4 = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, d4, h4, w4)
        x4 = self.norm4(x).transpose(1, 2).contiguous().view(-1, x.shape[-1], d4, h4, w4)
        outs.append(x4)

        return outs  # [x1, x2, x3, x4]


class ConvModule3D(nn.Module):
    def __init__(self, c1: int, c2: int, k: int):
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, padding=k // 2, bias=False)
        self.bn = nn.InstanceNorm3d(c2, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Decoder3D(nn.Module):
    """
    Fuse 4-stage features: project to same embed_dim, upsample to the highest resolution (stage1), concatenate and fuse.
    Then upsample once to original resolution.
    """
    def __init__(self, dims: List[int], embed_dim: int, num_classes: int):
        super().__init__()
        self.proj1 = nn.Conv3d(dims[0], embed_dim, kernel_size=1)
        self.proj2 = nn.Conv3d(dims[1], embed_dim, kernel_size=1)
        self.proj3 = nn.Conv3d(dims[2], embed_dim, kernel_size=1)
        self.proj4 = nn.Conv3d(dims[3], embed_dim, kernel_size=1)

        self.fuse = ConvModule3D(embed_dim * 4, embed_dim, k=1)
        self.dropout = nn.Dropout3d(0.1)
        self.head = nn.Conv3d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor], in_shape: Tuple[int, int, int]) -> torch.Tensor:
        # feats: [x1, x2, x3, x4] at resolutions [D1,H1,W1] ... [D4,H4,W4]
        x1, x2, x3, x4 = feats
        b, _, d1, h1, w1 = x1.shape

        c1 = self.proj1(x1)
        c2 = F.interpolate(self.proj2(x2), size=(d1, h1, w1), mode='trilinear', align_corners=False)
        c3 = F.interpolate(self.proj3(x3), size=(d1, h1, w1), mode='trilinear', align_corners=False)
        c4 = F.interpolate(self.proj4(x4), size=(d1, h1, w1), mode='trilinear', align_corners=False)

        c = self.fuse(torch.cat([c4, c3, c2, c1], dim=1))
        c = self.dropout(c)
        logits_stage1 = self.head(c)  # [B, num_classes, D1, H1, W1]

        # Upsample once more to original input resolution
        D, H, W = in_shape
        logits = F.interpolate(logits_stage1, size=(D, H, W), mode='trilinear', align_corners=False)
        return logits


# ------------------------------
# Public Model
# ------------------------------

segformer3d_settings = {
    # [channel dims, num blocks per stage, heads per stage, reduction ratio per stage, decoder embed dim]
    'B0_3D': [[32, 64, 160, 256], [2, 2, 2, 2], [1, 2, 5, 8], [4, 4, 2, 1], 256],
    'B1_3D': [[64, 128, 320, 512], [2, 2, 2, 2], [1, 2, 5, 8], [4, 4, 2, 1], 256],
}

class SegFormer3D(nn.Module):
    """
    A clean 3D SegFormer-like network:
    - Encoder: MiT3D (overlap 3D patch embedding + efficient 3D attention + MixFFN3D)
    - Decoder: fuse 4 scales to highest, then upsample to original size
    - IO: input [B, C, X, Y, Z] -> output [B, num_classes, X, Y, Z]
    """
    def __init__(self, model_name: str = 'B0_3D', num_classes: int = 2, in_channels: int = 1) -> None:
        super().__init__()
        assert model_name in segformer3d_settings, f"model_name must be one of {list(segformer3d_settings.keys())}"
        dims, layers, heads, reduction, embed_dim = segformer3d_settings[model_name]

        self.encoder = MiT3D(in_channels=in_channels, dims=dims, layers=layers, heads=heads, reduction=reduction)
        self.decoder = Decoder3D(dims=dims, embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        in_shape = x.shape[2:]  # (D, H, W)
        feats = self.encoder(x)
        logits = self.decoder(feats, in_shape)
        return logits

