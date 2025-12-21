import logging
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch.nn.functional as F


from torch_geometric.utils import to_dense_batch


import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from einops import rearrange
from collections import Counter

# SegFormer的高效自注意力
import torch
import torch.nn as nn


class EfficientSelfAtten3D(nn.Module):
    def __init__(self, dim, heads, reduction_ratio):
        super().__init__()
        self.heads = heads  # [1,2,5,8]
        self.reduction_ratio = reduction_ratio  # 2
        self.scale = (
            dim // heads
        ) ** -0.5  # dims = [64, 128, 320, 512]   so  self.scale = 0.125
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        # kernel_size = reduction_ratio, stride = reduction_ratio D,H,W各变为原来的1/reduction_ratio
        self.sr = nn.Conv3d(dim, dim, reduction_ratio, reduction_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(
            B, N, self.heads, C // self.heads
        )  # [B, heads, N, C//heads]
        p_x = x.clone().permute(0, 2, 1).reshape(B, C, W, H, D)
        sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
        x = self.norm(sp_x)  # [B, N//(reduce_ratio**3), C]
        kv = (
            self.kv(x)
            .reshape(B, -1, 2, self.heads, C // self.heads)
            .permute(2, 0, 1, 3, 4)
        )  # [2, B, heads, N//(reduce_ratio**3), C//heads]
        k, v = kv[0], kv[1]  # [B, heads, N//(reduce_ratio**3), C//heads]
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        attention = torch.softmax(energy * self.scale, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(B, N, C)
        out = self.proj(out)
        return out


class Scale_reduce3D(nn.Module):
    # 做上下文桥接模块中EfficientAtten需要的缩放模块
    def __init__(
        self,
        dim,
        bridge_reduction_ratios,
        heads,
        d_base_feat_size,
        d_base_depth_size,
        part=None,
    ):
        super().__init__()
        self.dim = dim  # 64  -> [64, 128, 320, 512]
        self.heads = heads  # [1, 2, 5, 8]
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.bridge_reduction_ratios = bridge_reduction_ratios  #  [1  2  4  8]
        self.part = part
        if len(self.bridge_reduction_ratios) == 4:
            self.sr0 = nn.Conv3d(
                dim * heads[0],
                dim * heads[0],
                bridge_reduction_ratios[3],
                bridge_reduction_ratios[3],
            )
            self.sr1 = nn.Conv3d(
                dim * heads[1],
                dim * heads[1],
                bridge_reduction_ratios[2],
                bridge_reduction_ratios[2],
            )
            self.sr2 = nn.Conv3d(
                dim * heads[2],
                dim * heads[2],
                bridge_reduction_ratios[1],
                bridge_reduction_ratios[1],
            )

        # 这里需要保证head输入进来和len(self.bridge_reduction_ratios)==4时相同
        elif len(self.bridge_reduction_ratios) == 3:
            self.sr0 = nn.Conv3d(
                dim * heads[1],
                dim * heads[1],
                bridge_reduction_ratios[2],
                bridge_reduction_ratios[2],
            )
            self.sr1 = nn.Conv3d(
                dim * heads[2],
                dim * heads[2],
                bridge_reduction_ratios[1],
                bridge_reduction_ratios[1],
            )

        # 这里是为了把中间的上下文桥接模块拆开两部分
        # 这里需要保证head输入进来和len(self.bridge_reduction_ratios)==4时相同
        elif len(self.bridge_reduction_ratios) == 2:
            if self.part == 0:
                self.sr0 = nn.Conv3d(
                    dim * heads[0],
                    dim * heads[0],
                    bridge_reduction_ratios[1],
                    bridge_reduction_ratios[1],
                )
                self.sr1 = nn.Conv3d(
                    dim * heads[1],
                    dim * heads[1],
                    bridge_reduction_ratios[0],
                    bridge_reduction_ratios[0],
                )
            elif self.part == 1:
                self.sr0 = nn.Conv3d(
                    dim * heads[2],
                    dim * heads[2],
                    bridge_reduction_ratios[1],
                    bridge_reduction_ratios[1],
                )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if len(self.bridge_reduction_ratios) == 4:
            range_size = [
                self.depth_size[i] * self.patch_size[i] ** 2 * self.heads[i]
                for i in range(len(self.patch_size))
            ]
            tem0 = (
                x[:, sum(range_size[:0]) : sum(range_size[:1]), :]
                .reshape(
                    B,
                    self.depth_size[0],
                    self.patch_size[0],
                    self.patch_size[0],
                    C * self.heads[0],
                )
                .permute(0, 4, 1, 2, 3)
            )
            tem1 = (
                x[:, sum(range_size[:1]) : sum(range_size[:2]), :]
                .reshape(
                    B,
                    self.depth_size[1],
                    self.patch_size[1],
                    self.patch_size[1],
                    C * self.heads[1],
                )
                .permute(0, 4, 1, 2, 3)
            )
            tem2 = (
                x[:, sum(range_size[:2]) : sum(range_size[:3]), :]
                .reshape(
                    B,
                    self.depth_size[2],
                    self.patch_size[2],
                    self.patch_size[2],
                    C * self.heads[2],
                )
                .permute(0, 4, 1, 2, 3)
            )
            tem3 = x[:, sum(range_size[:3]) : sum(range_size[:4]), :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        elif len(self.bridge_reduction_ratios) == 3:
            range_size = [
                self.depth_size[i + 1] * self.patch_size[i + 1] ** 2 * self.heads[i + 1]
                for i in range(len(self.patch_size) - 1)
            ]
            tem0 = (
                x[:, sum(range_size[:0]) : sum(range_size[:1]), :]
                .reshape(
                    B,
                    self.depth_size[1],
                    self.patch_size[1],
                    self.patch_size[1],
                    C * self.heads[1],
                )
                .permute(0, 4, 1, 2, 3)
            )
            tem1 = (
                x[:, sum(range_size[:1]) : sum(range_size[:2]), :]
                .reshape(
                    B,
                    self.depth_size[2],
                    self.patch_size[2],
                    self.patch_size[2],
                    C * self.heads[2],
                )
                .permute(0, 4, 1, 2, 3)
            )
            tem2 = x[:, sum(range_size[:2]) : sum(range_size[:3]), :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        # 这部分是我要拆分上下文桥接模块 把4个尺度的输入分成两块去做
        elif len(self.bridge_reduction_ratios) == 2:
            if self.part == 0:
                # 因为只有一个分界点 就这样写方便了
                interval = self.depth_size[0] * self.patch_size[0] ** 2 * self.heads[0]
                tem0 = (
                    x[:, :interval, :]
                    .reshape(
                        B,
                        self.depth_size[0],
                        self.patch_size[0],
                        self.patch_size[0],
                        C * self.heads[0],
                    )
                    .permute(0, 4, 1, 2, 3)
                )
                tem1 = (
                    x[:, interval:, :]
                    .reshape(
                        B,
                        self.depth_size[1],
                        self.patch_size[1],
                        self.patch_size[1],
                        C * self.heads[1],
                    )
                    .permute(0, 4, 1, 2, 3)
                )

                sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
                sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

                reduce_out = self.norm(torch.cat([sr_0, sr_1], -2))

            elif self.part == 1:
                interval = self.depth_size[2] * self.patch_size[2] ** 2 * self.heads[2]
                tem0 = (
                    x[:, :interval, :]
                    .reshape(
                        B,
                        self.depth_size[2],
                        self.patch_size[2],
                        self.patch_size[2],
                        C * self.heads[2],
                    )
                    .permute(0, 4, 1, 2, 3)
                )
                tem1 = x[:, interval:, :]

                sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)

                reduce_out = self.norm(torch.cat([sr_0, tem1], -2))

        return reduce_out


# Mix高效自注意力
class M_EfficientSelfAtten3D(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        bridge_reduction_ratios,
        d_base_feat_size,
        d_base_depth_size,
        part=None,
    ):
        super().__init__()
        self.heads = heads  # [1, 2, 5, 8]
        self.bridge_reduction_ratios = bridge_reduction_ratios  # [1  2  4  8]
        self.scale = (dim // heads[0]) ** -0.5  # dim = 64  self.scale = 0.125
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if bridge_reduction_ratios is not None:
            self.scale_reduce = Scale_reduce3D(
                dim,
                bridge_reduction_ratios,
                heads,
                d_base_feat_size,
                d_base_depth_size,
                part,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, C).permute(0, 2, 1, 3)

        if self.bridge_reduction_ratios is not None:
            x = self.scale_reduce(
                x
            )  ##[B, N拆开后在每一段用bridge_reduction_radios降低空间分辨率, C]

        kv = self.kv(x).reshape(B, -1, 2, 1, C).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


# Depth-Wise Conv
# 减少参数量
class DWConv3D(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, W, H, D)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN3D(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv3D(c2, 3)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), W, H, D))
        out = self.fc2(ax)
        return out


class MSFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv3D(c2, 3)
        self.dwconv2 = DWConv3D(c2, 5)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        x = self.fc1(x)
        x1 = self.norm1(self.dwconv(x, W, H, D) + self.dwconv2(x, W, H, D) + x)
        ax = self.act(x1)
        out = self.fc2(ax)
        return out


class MixFFN_skip3D(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv3D(c2, 3)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        x = self.fc1(x)
        x1 = self.norm1(self.dwconv(x, W, H, D) + x)
        x2 = self.norm2(x1 + x)
        x3 = self.norm3(x2 + x)
        ax = self.act(x3)
        out = self.fc2(ax)
        return out


class MLP_FFN3D(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class OverlapPatchEmbeddings3D(nn.Module):
    # kernel_size [7, 3, 3, 3]
    # stride [4, 2, 2, 2]
    # padding [3, 1, 1, 1]
    def __init__(self, kernel_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size, stride, padding)
        # (224 - 7 + 2*3 )//4 + 1 = 56
        # (32 -7 + 2*3) //4 + 1 = 8
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, W, H, D = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, W, H, D


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, heads, reduction_ratio=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten3D(dim, heads, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == "mix":
            self.mlp = MixFFN3D(dim, int(dim * 4))
        elif token_mlp == "mix_skip":
            # self.mlp = MixFFN_skip3D(dim, int(dim * 4))
            self.mlp = MSFFN(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN3D(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, W, H, D) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), W, H, D)
        mx = tx + self.mlp(self.norm2(tx), W, H, D)
        return mx


class CrossModalAttention(nn.Module):
    """血管-图像跨模态注意力"""

    def __init__(self, img_dim, gnn_dim, embed_dim=128):
        super().__init__()
        self.query = nn.Linear(img_dim, embed_dim)  # img_dim -> embed_dim
        self.key = nn.Linear(gnn_dim, embed_dim)  # gnn_dim -> embed_dim
        self.value = nn.Linear(gnn_dim, img_dim)  # gnn_dim -> img_dim (最终目标)

    def forward(self, img_feats, gnn_feats, batch_mask):
        B, _, C = img_feats.shape

        # CrossAttention 计算
        Q = self.query(img_feats)  # [B, D*H*W, embed_dim]
        K = self.key(gnn_feats)  # [B, N_max, embed_dim]
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)), dim=-1
        )  # [B, D*H*W, N_max]
        V = self.value(gnn_feats)  # [B, N_max, C]
        fused = torch.bmm(attn, V)  # [B, D*H*W, C]

        # 使用 batch_mask 控制结果
        mask = batch_mask.view(B, 1, 1)
        output = img_feats + fused * mask.float()

        return output


# MISSFormer的Transformer
class MiT3D(nn.Module):
    def __init__(
        self,
        dims,
        layers,
        token_mlp="mix_skip",
        kernel_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        padding_sizes=[3, 1, 1, 1],
        reduction_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        input_dims=1,
        vessel_fusion=False,
        gnn_dim=768,
    ):
        super().__init__()
        self.vessel_fusion = vessel_fusion
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings3D(
            kernel_sizes[0], strides[0], padding_sizes[0], input_dims, dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings3D(
            kernel_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings3D(
            kernel_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings3D(
            kernel_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [
                TransformerBlock3D(dims[0], heads[0], reduction_ratios[0], token_mlp)
                for _ in range(layers[0])
            ]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList(
            [
                TransformerBlock3D(dims[1], heads[1], reduction_ratios[1], token_mlp)
                for _ in range(layers[1])
            ]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList(
            [
                TransformerBlock3D(dims[2], heads[2], reduction_ratios[2], token_mlp)
                for _ in range(layers[2])
            ]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList(
            [
                TransformerBlock3D(dims[3], heads[3], reduction_ratios[3], token_mlp)
                for _ in range(layers[3])
            ]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        if self.vessel_fusion:
            self.fuse_low = CrossModalAttention(dims[0], gnn_dim)
        # self.heads = nn.Linear(dims[3], num_classes)

    def forward(
        self, x: torch.Tensor, vessel_feature=None, batch_mask=None
    ) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, W, H, D = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, W, H, D)
        #  注入先验知识
        if self.vessel_fusion:
            x = self.fuse_low(x, vessel_feature, batch_mask)
        x = self.norm1(x)
        x = x.reshape(B, W, H, D, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        # stage 2
        x, W, H, D = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, W, H, D)
        x = self.norm2(x)
        x = x.reshape(B, W, H, D, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        # stage 3
        x, W, H, D = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, W, H, D)
        x = self.norm3(x)
        x = x.reshape(B, W, H, D, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        # stage 4
        x, W, H, D = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, W, H, D)
        x = self.norm4(x)
        x = x.reshape(B, W, H, D, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        return outs


class PatchExpand3D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = (
            nn.Linear(dim, dim_scale**2 * dim, bias=False)
            if dim_scale == 2
            else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """

        W, H, D = self.input_resolution

        x = self.expand(x)
        B, L, C = x.shape
        assert L == D * H * W, "Input feature has wrong size"

        x = x.view(B, W, H, D, C)
        x = rearrange(
            x,
            "b w h d (p1 p2 p3 c)-> b (w p1) (h p2) (d p3) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            p3=self.dim_scale,
            c=C // (self.dim_scale**3),
        )
        x = x.view(B, -1, C // 8)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4_3D(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale**3 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        W, H, D = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == D * H * W, "Input feature has wrong size"

        x = x.view(B, W, H, D, C)
        x = rearrange(
            x,
            "b w h d (p1 p2 p3 c)-> b (w p1) (h p2) (d p3) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            p3=self.dim_scale,
            c=C // (self.dim_scale**3),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class BridgeLayer_0_1_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size, 0
        )
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip3D(dims * heads[0], dims * heads[0] * 4)
        self.mixffn2 = MixFFN_skip3D(dims * heads[1], dims * heads[1] * 4)
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if type(inputs) == list:
            # print("-----1-----")
            c1, c2 = inputs
            B, C, _, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            inputs = torch.cat([c1f, c2f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        interval = self.depth_size[0] * self.patch_size[0] ** 2 * self.heads[0]
        tem1 = tx[:, :interval, :].reshape(B, -1, C * self.heads[0])
        tem2 = tx[:, interval:, :].reshape(B, -1, C * self.heads[1])

        m1f = self.mixffn1(
            tem1, self.depth_size[0], self.patch_size[0], self.patch_size[0]
        ).reshape(B, -1, C)
        m2f = self.mixffn2(
            tem2, self.depth_size[1], self.patch_size[1], self.patch_size[1]
        ).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f], -2)

        tx2 = tx1 + t1

        return tx2


class BridgeLayer_2_3_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size, 1
        )
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip3D(dims * heads[2], dims * heads[2] * 4)
        self.mixffn2 = MixFFN_skip3D(dims * heads[3], dims * heads[3] * 4)
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if type(inputs) == list:
            # print("-----1-----")
            c3, c4 = inputs
            B, C, _, _, _ = c3.shape
            C = int(C // self.heads[2])
            c3f = c3.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            inputs = torch.cat([c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        interval = self.depth_size[2] * self.patch_size[2] ** 2 * self.heads[2]
        tem1 = tx[:, :interval, :].reshape(B, -1, C * self.heads[2])
        tem2 = tx[:, interval:, :].reshape(B, -1, C * self.heads[3])

        m1f = self.mixffn1(
            tem1, self.depth_size[2], self.patch_size[2], self.patch_size[2]
        ).reshape(B, -1, C)
        m2f = self.mixffn2(
            tem2, self.depth_size[3], self.patch_size[3], self.patch_size[3]
        ).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f], -2)

        tx2 = tx1 + t1

        return tx2


# tokenFi = Reshape(Fi, [B,-1,C])
# mergeToken = Concatenate(tokenFi, dim=1)
# AttenToken = EfficientSelfAtten(LN(mergeToken))
# resToken = LN(mergeToken + AttenToken)
# splitToken = Split(resToken, dim = 1)
# FFNi = EnhanceMixFFN(splitToken)
# output = Concatenate(FFNi, dim=1) + resToken
class BridgeLayer_4_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip3D(dims * heads[0], dims * heads[0] * 4)
        self.mixffn2 = MixFFN_skip3D(dims * heads[1], dims * heads[1] * 4)
        self.mixffn3 = MixFFN_skip3D(dims * heads[2], dims * heads[2] * 4)
        self.mixffn4 = MixFFN_skip3D(dims * heads[3], dims * heads[3] * 4)
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.heads = heads

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if type(inputs) == list:
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c3f = c3.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        range_size = [
            self.patch_size[i] ** 2 * self.depth_size[i] * self.heads[i]
            for i in range(len(self.patch_size))
        ]

        tem1 = tx[:, sum(range_size[:0]) : sum(range_size[:1]), :].reshape(
            B, -1, C * self.heads[0]
        )
        tem2 = tx[:, sum(range_size[:1]) : sum(range_size[:2]), :].reshape(
            B, -1, C * self.heads[1]
        )
        tem3 = tx[:, sum(range_size[:2]) : sum(range_size[:3]), :].reshape(
            B, -1, C * self.heads[2]
        )
        tem4 = tx[:, sum(range_size[:3]) : sum(range_size[:4]), :].reshape(
            B, -1, C * self.heads[3]
        )

        m1f = self.mixffn1(
            tem1, self.depth_size[0], self.patch_size[0], self.patch_size[0]
        ).reshape(B, -1, C)
        m2f = self.mixffn2(
            tem2, self.depth_size[1], self.patch_size[1], self.patch_size[1]
        ).reshape(B, -1, C)
        m3f = self.mixffn3(
            tem3, self.depth_size[2], self.patch_size[2], self.patch_size[2]
        ).reshape(B, -1, C)
        m4f = self.mixffn4(
            tem4, self.depth_size[3], self.patch_size[3], self.patch_size[3]
        ).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2


class BridgeLayer_3_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn2 = MixFFN3D(dims * heads[1], dims * heads[1] * 4)
        self.mixffn3 = MixFFN3D(dims * heads[2], dims * heads[2] * 4)
        self.mixffn4 = MixFFN3D(dims * heads[3], dims * heads[3] * 4)
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.heads = heads

    def forward(self, inputs) -> torch.Tensor:
        B = inputs[0].shape[0]
        C = 64
        if type(inputs) == list:
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c3f = c3.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        range_size = [
            self.depth_size[i + 1] * self.patch_size[i + 1] ** 2 * self.heads[i + 1]
            for i in range(len(self.patch_size) - 1)
        ]
        tem2 = tx[:, sum(range_size[:0]) : sum(range_size[:1]), :].reshape(
            B, -1, C * self.heads[1]
        )
        tem3 = tx[:, sum(range_size[:1]) : sum(range_size[:2]), :].reshape(
            B, -1, C * self.heads[2]
        )
        tem4 = tx[:, sum(range_size[:2]) : sum(range_size[:3]), :].reshape(
            B, -1, C * self.heads[3]
        )

        m2f = self.mixffn2(
            tem2, self.depth_size[1], self.patch_size[1], self.patch_size[1]
        ).reshape(B, -1, C)
        m3f = self.mixffn3(
            tem3, self.depth_size[2], self.patch_size[2], self.patch_size[2]
        ).reshape(B, -1, C)
        m4f = self.mixffn4(
            tem4, self.depth_size[3], self.patch_size[3], self.patch_size[3]
        ).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2


class BridegeBlock_X_3D(nn.Module):
    def __init__(
        self,
        dims,
        heads,
        bridge_reduction_ratios,
        X,
        d_base_feat_size,
        d_base_depth_size,
    ):
        super().__init__()
        # 千万要注意这里的bridge_reduction_ratios顺序   是和步骤反着来的
        self.bridge_layers1 = nn.ModuleList(
            [
                BridgeLayer_0_1_3D(
                    dims,
                    heads,
                    bridge_reduction_ratios[2:],
                    d_base_feat_size,
                    d_base_depth_size,
                )
                for _ in range(X)
            ]
        )
        self.bridge_layers2 = nn.ModuleList(
            [
                BridgeLayer_2_3_3D(
                    dims,
                    heads,
                    bridge_reduction_ratios[:2],
                    d_base_feat_size,
                    d_base_depth_size,
                )
                for _ in range(X)
            ]
        )
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:2], x[2:]
        for layer in self.bridge_layers1:
            x1 = layer(x1)
        for layer in self.bridge_layers2:
            x2 = layer(x2)

        B, _, C = x1.shape
        B1, _, C1 = x2.shape

        assert B == B1 and C == C1, "shape error"

        outs = []
        interval1 = self.depth_size[0] * self.patch_size[0] ** 2 * self.heads[0]
        interval2 = self.depth_size[2] * self.patch_size[2] ** 2 * self.heads[2]
        sk1 = (
            x1[:, :interval1, :]
            .reshape(
                B,
                self.depth_size[0],
                self.patch_size[0],
                self.patch_size[0],
                C * self.heads[0],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk2 = (
            x1[:, interval1:, :]
            .reshape(
                B,
                self.depth_size[1],
                self.patch_size[1],
                self.patch_size[1],
                C * self.heads[1],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk3 = (
            x2[:, :interval2, :]
            .reshape(
                B,
                self.depth_size[2],
                self.patch_size[2],
                self.patch_size[2],
                C * self.heads[2],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk4 = (
            x2[:, interval2:, :]
            .reshape(
                B,
                self.depth_size[3],
                self.patch_size[3],
                self.patch_size[3],
                C * self.heads[3],
            )
            .permute(0, 4, 1, 2, 3)
        )

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class BridegeBlock_4_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer2 = BridgeLayer_4_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer3 = BridgeLayer_4_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer4 = BridgeLayer_4_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape
        outs = []

        range_size = [
            self.depth_size[i] * self.patch_size[i] ** 2 * self.heads[i]
            for i in range(len(self.patch_size))
        ]
        sk1 = (
            bridge4[:, sum(range_size[:0]) : sum(range_size[:1]), :]
            .reshape(
                B,
                self.depth_size[0],
                self.patch_size[0],
                self.patch_size[0],
                C * self.heads[0],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk2 = (
            bridge4[:, sum(range_size[:1]) : sum(range_size[:2]), :]
            .reshape(
                B,
                self.depth_size[1],
                self.patch_size[1],
                self.patch_size[1],
                C * self.heads[1],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk3 = (
            bridge4[:, sum(range_size[:2]) : sum(range_size[:3]), :]
            .reshape(
                B,
                self.depth_size[2],
                self.patch_size[2],
                self.patch_size[2],
                C * self.heads[2],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk4 = (
            bridge4[:, sum(range_size[:3]) : sum(range_size[:4]), :]
            .reshape(
                B,
                self.depth_size[3],
                self.patch_size[3],
                self.patch_size[3],
                C * self.heads[3],
            )
            .permute(0, 4, 1, 2, 3)
        )

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class BridegeBlock_3_3D(nn.Module):
    def __init__(
        self, dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
    ):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer2 = BridgeLayer_3_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer3 = BridgeLayer_3_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.bridge_layer4 = BridgeLayer_3_3D(
            dims, heads, bridge_reduction_ratios, d_base_feat_size, d_base_depth_size
        )
        self.patch_size = [i * d_base_feat_size for i in [8, 4, 2, 1]]
        self.depth_size = [i * d_base_depth_size for i in [8, 4, 2, 1]]
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if type(x) == list:
            # print("-----1-----")
            outs.append(x[0])
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape

        range_size = [
            self.depth_size[i + 1] * self.patch_size[i + 1] ** 2 * self.heads[i + 1]
            for i in range(len(self.patch_size) - 1)
        ]
        sk2 = (
            bridge4[:, sum(range_size[:0]) : sum(range_size[:1]), :]
            .reshape(
                B,
                self.depth_size[1],
                self.patch_size[1],
                self.patch_size[1],
                C * self.heads[1],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk3 = (
            bridge4[:, sum(range_size[:1]) : sum(range_size[:2]), :]
            .reshape(
                B,
                self.depth_size[2],
                self.patch_size[2],
                self.patch_size[2],
                C * self.heads[2],
            )
            .permute(0, 4, 1, 2, 3)
        )
        sk4 = (
            bridge4[:, sum(range_size[:2]) : sum(range_size[:3]), :]
            .reshape(
                B,
                self.depth_size[3],
                self.patch_size[3],
                self.patch_size[3],
                C * self.heads[3],
            )
            .permute(0, 4, 1, 2, 3)
        )

        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer3D(nn.Module):
    def __init__(
        self,
        input_size,
        in_out_chan,
        heads,
        reduction_ratios,
        token_mlp_mode,
        n_class=9,
        norm_layer=nn.LayerNorm,
        is_last=False,
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        # transformer decoder
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand3D(
                input_resolution=input_size,
                dim=out_dim,
                dim_scale=2,
                norm_layer=norm_layer,
            )
            self.last_layer = None
        else:
            # 只有最后一层的时候，才会有4倍缩放比
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4_3D(
                input_resolution=input_size,
                dim=out_dim,
                dim_scale=4,
                norm_layer=norm_layer,
            )
            self.last_layer = nn.Conv3d(out_dim, n_class, kernel_size=1)

        self.layer_former_1 = TransformerBlock3D(
            out_dim, heads, reduction_ratios, token_mlp_mode
        )
        self.layer_former_2 = TransformerBlock3D(
            out_dim, heads, reduction_ratios, token_mlp_mode
        )

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv3d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2):

        b, w, h, d, c = x2.shape
        x2 = x2.view(b, -1, c)
        cat_x = torch.cat([x1, x2], dim=-1)
        cat_linear_x = self.concat_linear(cat_x)
        tran_layer_1 = self.layer_former_1(cat_linear_x, w, h, d)
        tran_layer_2 = self.layer_former_2(tran_layer_1, w, h, d)
        logits = self.layer_up(tran_layer_2)
        if self.last_layer is not None:
            out = self.last_layer(
                logits.view(b, 4 * w, 4 * h, 4 * d, -1).permute(0, 4, 1, 2, 3)
            )
            return out, logits.view(b, 4 * w, 4 * h, 4 * d, -1).permute(0, 4, 1, 2, 3)
        return logits


class MISSFormer3D(nn.Module):
    def __init__(
        self,
        input_dims=1,
        num_classes=9,
        token_mlp_mode="mix_skip",
        heads=[1, 2, 5, 8],
        reduction_ratios=[8, 4, 2, 1],
        bridge_reduction_ratios=[1, 2, 4, 8],
        in_out_chan=[[32, 64], [144, 128], [288, 320], [512, 512]],
        dims=[64, 128, 320, 512],
        layers=[2, 2, 2, 2],
        d_base_feat_size=3,
        d_base_depth_size=1,
        kernel_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        padding_sizes=[3, 1, 1, 1],
        encoder_pretrained=True,
        vessel_fusion=False,
        gnn_dim=768,
    ):
        super().__init__()

        self.backbone = MiT3D(
            dims,
            layers,
            token_mlp_mode,
            kernel_sizes,
            strides,
            padding_sizes,
            reduction_ratios,
            heads,
            input_dims=input_dims,
            vessel_fusion=vessel_fusion,
            gnn_dim=gnn_dim,
        )

        # self.bridge = BridegeBlock_4_3D(
        #     dims[0],
        #     heads,
        #     bridge_reduction_ratios,
        #     d_base_feat_size,
        #     d_base_depth_size,
        # )

        self.bridge = BridegeBlock_X_3D(
            dims[0],
            heads,
            bridge_reduction_ratios,
            6,  # 4
            d_base_feat_size,
            d_base_depth_size,
        )

        self.decoder_3 = MyDecoderLayer3D(
            (
                d_base_depth_size * reduction_ratios[3],
                d_base_feat_size * reduction_ratios[3],
                d_base_feat_size * reduction_ratios[3],
            ),
            in_out_chan[3],
            heads[3],
            reduction_ratios[3],
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_2 = MyDecoderLayer3D(
            (
                d_base_depth_size * reduction_ratios[2],
                d_base_feat_size * reduction_ratios[2],
                d_base_feat_size * reduction_ratios[2],
            ),
            in_out_chan[2],
            heads[2],
            reduction_ratios[2],
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer3D(
            (
                d_base_depth_size * reduction_ratios[1],
                d_base_feat_size * reduction_ratios[1],
                d_base_feat_size * reduction_ratios[1],
            ),
            in_out_chan[1],
            heads[1],
            reduction_ratios[1],
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer3D(
            (
                d_base_depth_size * reduction_ratios[0],
                d_base_feat_size * reduction_ratios[0],
                d_base_feat_size * reduction_ratios[0],
            ),
            in_out_chan[0],
            heads[0],
            reduction_ratios[0],
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x, vessel_feature=None, batch_mask=None):
        x = x.permute(0, 1, 4, 2, 3)
        encoder = self.backbone(x, vessel_feature, batch_mask)
        bridge = self.bridge(encoder)
        b, c, d, h, w = bridge[3].shape
        tmp_3 = self.decoder_3(
            bridge[3].permute(0, 2, 3, 4, 1).reshape(b, -1, c),
            bridge[3].permute(0, 2, 3, 4, 1),
        )
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 4, 1))
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 4, 1))
        out, tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 4, 1))
        return torch.cat(
            [tmp_0.permute(0, 1, 3, 4, 2), out.permute(0, 1, 3, 4, 2)], dim=1
        )


class VesselGNN(nn.Module):
    def __init__(
        self, in_channels=4, hidden_dim=64, out_dim=128, num_layers=5, conv_type="GCN"
    ):
        super(VesselGNN, self).__init__()

        if conv_type == "GCN":
            ConvLayer = GCNConv
        elif conv_type == "GraphSAGE":
            ConvLayer = SAGEConv
        elif conv_type == "GAT":
            ConvLayer = GATConv
        else:
            raise ValueError("Unsupported GNN type!")

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(ConvLayer(in_channels, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_dim, hidden_dim))
        self.convs.append(ConvLayer(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        """
        :param x:  [B, N_max, in_channels]  每个 batch 的节点特征
        :param edge_index: [B, 2, edge]  每个 batch 的边连接
        :return: [B, N_max, out_dim]  返回每个节点的特征
        """
        B, N_max, _ = x.shape
        out_x = torch.zeros(B, N_max, self.convs[-1].out_channels, device=x.device)

        for b in range(B):
            x_b = x[b]  # 取出第 b 个 batch 的节点特征 [N_max, in_channels]
            edge_index_b = edge_index[b]  # 取出对应 batch 的边 [2, edge]

            # 逐层传播
            for conv in self.convs:
                x_b = conv(x_b, edge_index_b)
                x_b = F.relu(x_b)

            out_x[b] = x_b  # 存回结果

        return out_x  # [B, N_max, out_dim]


def compute_normals(keypoints_tensor, edt_gradients):
    """
    计算中心线点的法线
    :param keypoints_tensor: 中心线点坐标（N, 3）
    :param edt_gradients: 欧式距离变换（EDT）的梯度（N, 3）
    :return: 法线向量（N, 3）
    """
    # 计算切线方向（使用梯度计算）
    tangents = torch.gradient(keypoints_tensor, dim=0)[0]

    # 归一化切线
    tangents = tangents / torch.norm(tangents, dim=1, keepdim=True)

    # 归一化 EDT 梯度
    edt_gradients = edt_gradients / torch.norm(edt_gradients, dim=1, keepdim=True)

    # 计算法线方向（法线 = 切线 x EDT梯度）
    normals = torch.cross(tangents, edt_gradients)

    # 归一化法线向量
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return normals


"""
k=6: 每个节点连接最近的6个邻居（基于KNN）
dist_threshold=5.0: 边的最大空间距离（单位：mm）

参数选择依据：
- 门静脉分支角度通常<60度，k=6可覆盖主要分支
- 5mm阈值避免连接非相邻血管段（正常门静脉分支间距约3-8mm）

N（节点数）由血管骨架决定：
skeleton = skeletonize(vessel_mask)
N = torch.sum(skeleton > 0)
"""


def enhanced_build_graph(
    vessel_mask: torch.Tensor, k=6, dist_threshold=5.0, N_max=256, with_normal=False
):
    B, D, W, H = vessel_mask.shape

    edge_index_batch = []
    node_features_batch = []
    batch_mask = torch.ones(
        B, dtype=torch.bool, device=vessel_mask.device
    )  # 用于标记空白部分
    Edge_max = 2 * N_max

    # 固定随机种子以确保确定性输出
    torch.manual_seed(42)

    for b in range(B):
        vessel_mask_np = vessel_mask[b].detach().cpu().numpy()
        vessel_mask_np[vessel_mask_np > 1] = 1
        skeleton = skeletonize(vessel_mask_np)
        keypoints = np.argwhere(skeleton > 0)

        keypoints_tensor = torch.tensor(
            keypoints, dtype=torch.float, device=vessel_mask.device
        )
        # 统一 N_max
        N = keypoints_tensor.shape[0]
        if N == 0:
            feature_dim = (
                4 if not with_normal else 7
            )  # 3 (坐标) + 1 (半径)   + 3 (法向量)
            node_features = torch.zeros((N_max, feature_dim), device=vessel_mask.device)
            nodes = torch.arange(Edge_max, device=vessel_mask.device)
            edge_index = torch.stack([nodes, nodes], dim=0)  # (2, Edge_max)
            batch_mask[b] = False  # 标记为空白部分
        else:
            if N > N_max:
                # 确定性采样：取前 N_max 个节点
                sampled_indices = torch.arange(N_max, device=vessel_mask.device)
                keypoints_tensor = keypoints_tensor[sampled_indices]
            elif N < N_max:
                # 确定性填充：重复前 N_max - N 个节点
                extra_indices = torch.arange(N_max - N, device=vessel_mask.device) % N
                keypoints_tensor = torch.cat(
                    [keypoints_tensor, keypoints_tensor[extra_indices]], dim=0
                )

            # 计算半径特征
            dt = torch.tensor(
                distance_transform_edt(vessel_mask_np),
                dtype=torch.float,
                device=vessel_mask.device,
            )
            radius_feature = dt[skeleton > 0].unsqueeze(1)
            if N > N_max:
                radius_feature = radius_feature[sampled_indices]
            elif N < N_max:
                radius_feature = torch.cat(
                    [radius_feature, radius_feature[extra_indices]], dim=0
                )
            node_features = torch.cat([keypoints_tensor, radius_feature], dim=1)

            # 计算法线特征
            if with_normal:
                dt_dx, dt_dy, dt_dz = torch.gradient(
                    dt, dim=(0, 1, 2)
                )  # 计算 EDT 在 3D 方向上的梯度
                edt_gradients = torch.stack(
                    [dt_dx, dt_dy, dt_dz], dim=-1
                )  # (D, H, W, 3)
                edt_gradients = edt_gradients[skeleton > 0]  # (N, 3)
                normals = compute_normals(keypoints_tensor, edt_gradients)
                if N > N_max:
                    normals = normals[sampled_indices]
                elif N < N_max:
                    normals = torch.cat([normals, normals[extra_indices]], dim=0)
                node_features = torch.cat(
                    [keypoints_tensor, radius_feature, normals], dim=1
                )

            # 邻接矩阵
            dist_matrix = torch.cdist(keypoints_tensor, keypoints_tensor)
            adj_matrix = torch.zeros_like(
                dist_matrix, dtype=torch.int, device=vessel_mask.device
            )
            knn_indices = torch.argsort(dist_matrix, dim=1)[:, :k]
            rows = torch.arange(N_max, device=vessel_mask.device).repeat_interleave(k)
            cols = knn_indices.flatten()
            adj_matrix[rows, cols] = 1
            adj_matrix[dist_matrix > dist_threshold] = 0

            edge_index = torch.stack(torch.where(adj_matrix)).long()

            E = edge_index.shape[1]
            if E > Edge_max:
                # 确定性截断：取前 Edge_max 条边
                edge_index = edge_index[:, :Edge_max]
            elif E < Edge_max:
                # 填充不足的边
                padding = torch.zeros(
                    (2, Edge_max - E), dtype=torch.long, device=vessel_mask.device
                )
                edge_index = torch.cat([edge_index, padding], dim=1)

        edge_index_batch.append(edge_index)
        node_features_batch.append(node_features)

    return (
        torch.stack(edge_index_batch, dim=0),
        torch.stack(node_features_batch, dim=0),
        batch_mask,
    )


class VesselEnhancedNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=9,
        gnn_dim=128,
        with_normal=False,
        is_test=False,
    ):
        super().__init__()
        self.net = MISSFormer3D(
            in_channels, num_classes, vessel_fusion=True, gnn_dim=gnn_dim
        )
        self.with_normal = with_normal
        # 血管处理分支
        self.gnn = VesselGNN(
            in_channels=4 if not self.with_normal else 7,
            hidden_dim=gnn_dim // 2,
            out_dim=gnn_dim,
            num_layers=3,
            conv_type="GCN",
        )
        self.is_test = is_test
        self.num_classes = num_classes
        # 多级融合模块

    def forward(self, inputs):
        ct_volume, vessel_mask = inputs[:, 0].unsqueeze(1), inputs[:, 1]
        vessel_mask = vessel_mask.permute(0, 3, 1, 2)  # [B,W,H,D]->[B,D,W,H]
        # 提取血管拓扑特征 # [B, 2, edges], [B,N_max,4] [B]
        edge_index_batch, node_features_batch, batch_mask = enhanced_build_graph(
            vessel_mask, with_normal=self.with_normal
        )
        # [B, gnn_dim]
        gnn_feats = self.gnn(node_features_batch, edge_index_batch)
        if self.is_test:
            return self.net(ct_volume, gnn_feats, batch_mask)[:, -self.num_classes :]
        return self.net(ct_volume, gnn_feats, batch_mask)
