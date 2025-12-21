import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from einops import rearrange
from collections import Counter

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def generate_vascular_bias(vessel_mask, num_heads=1):
    """
    显存优化的血管偏置生成

    Args:
        vessel_mask: (B,1,H,W,D)
        num_heads: 注意力头数
    Returns:
        bias: (B, num_heads, 1, k_len) 可通过广播机制自动扩展
    """
    # 参数对齐
    B, _, H, W, D = vessel_mask.shape
    kernel_size, stride, padding = 7, 4, 3

    # 下采样 (与patch_embed一致)
    down_mask = F.avg_pool3d(
        vessel_mask.float(), kernel_size=kernel_size, stride=stride, padding=padding
    )  # (B,1,H',W',D')

    # 计算距离场
    D_vascular = []
    for b in range(B):
        mask_np = down_mask[b, 0].detach().cpu().numpy()
        distance = distance_transform_edt(1 - mask_np)
        distance_norm = distance / (distance.max() + 1e-6)
        D_vascular.append(torch.from_numpy(distance_norm).float())

    D_vascular = torch.stack(D_vascular, dim=0).to(vessel_mask.device)  # (B,H',W',D')

    # 构造可广播的偏置张量
    bias = D_vascular.view(B, -1, num_heads, 1)  # (B,  q_len, n_heads, 1)

    return -10.0 * bias


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

    def forward(self, x: torch.Tensor, W, H, D, vessel_dis=None) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(
            B, N, self.heads, C // self.heads
        )  # [B, heads, N, C//heads]
        if vessel_dis is not None:
            q += vessel_dis
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

    def forward(self, x: torch.Tensor, W, H, D, vessel_dis=None) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), W, H, D, vessel_dis=vessel_dis)
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
    ):
        super().__init__()
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

        self.first_head = heads[0]
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
        # self.heads = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor, vessel_mask=None) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # 生成血管偏置（仅在训练时启用）
        if vessel_mask is not None and self.training:
            vessel_dis = generate_vascular_bias(vessel_mask, num_heads=self.first_head)
        else:
            vessel_dis = None

        # stage 1
        x, W, H, D = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, W, H, D, vessel_dis=vessel_dis)
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
        if self.last_layer:
            out = self.last_layer(
                self.layer_up(tran_layer_2)
                .view(b, 4 * w, 4 * h, 4 * d, -1)
                .permute(0, 4, 1, 2, 3)
            )
        else:
            out = self.layer_up(tran_layer_2)
        return out


class MISSFormer3D_disEmbed(nn.Module):
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

    def forward(self, x):
        x, vessel_mask = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        x = x.permute(0, 1, 4, 2, 3)
        vessel_mask = vessel_mask.permute(0, 1, 4, 2, 3)
        encoder = self.backbone(x, vessel_mask)
        bridge = self.bridge(encoder)
        b, c, d, h, w = bridge[3].shape
        tmp_3 = self.decoder_3(
            bridge[3].permute(0, 2, 3, 4, 1).reshape(b, -1, c),
            bridge[3].permute(0, 2, 3, 4, 1),
        )
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 4, 1))
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 4, 1))
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 4, 1))
        return tmp_0.permute(0, 1, 3, 4, 2)
