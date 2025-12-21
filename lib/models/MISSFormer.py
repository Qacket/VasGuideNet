import torch
import torch.nn as nn
from .segformer import *
from typing import Tuple
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x


class SegU_decoder(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

        

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class BridgeLayer_0_1(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.heads = head
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, bridge_reduction_ratios, 0)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims*head[0],dims*head[0]*4)
        self.mixffn2 = MixFFN_skip(dims*head[1],dims*head[1]*4)
        self.patch_size=[56,28,14,7]

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)

            inputs = torch.cat([c1f, c2f], -2)
        else:
            B,_,C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        interval = self.patch_size[0]**2 * self.heads[0]
        tem1 = tx[:,:interval,:].reshape(B, -1, C*self.heads[0])
        tem2 = tx[:,interval:,:].reshape(B, -1, C*self.heads[1])

        m1f = self.mixffn1(tem1, self.patch_size[0], self.patch_size[0]).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, self.patch_size[1], self.patch_size[1]).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f], -2)

        tx2 = tx1 + t1

        return tx2

class BridgeLayer_2_3(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.heads = head
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, bridge_reduction_ratios, 1)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims*head[2],dims*head[2]*4)
        self.mixffn2 = MixFFN_skip(dims*head[3],dims*head[3]*4)
        self.patch_size=[56,28,14,7]

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c3, c4 = inputs
            B, C, _, _= c3.shape
            C= int(C //self.heads[2])
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)

            inputs = torch.cat([c3f, c4f], -2)
        else:
            B,_,C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        interval = self.patch_size[2]**2 * self.heads[2]
        tem1 = tx[:,:interval,:].reshape(B, -1, C*self.heads[2])
        tem2 = tx[:,interval:,:].reshape(B, -1, C*self.heads[3])

        m1f = self.mixffn1(tem1, self.patch_size[2], self.patch_size[2]).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, self.patch_size[3], self.patch_size[3]).reshape(B, -1, C)

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
class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.heads = head
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, bridge_reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims*head[0],dims*head[0]*4)
        self.mixffn2 = MixFFN_skip(dims*head[1],dims*head[1]*4)
        self.mixffn3 = MixFFN_skip(dims*head[2],dims*head[2]*4)
        self.mixffn4 = MixFFN_skip(dims*head[3],dims*head[3]*4)
        self.patch_size=[56,28,14,7]

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        range_size= [self.patch_size[i]**2 * self.head[i] for i in range(len(self.patch_size))]

        tem1 = tx[:,sum(range_size[:0]):sum(range_size[:1]),:].reshape(B, -1, C*self.head[0]) 
        tem2 = tx[:,sum(range_size[:1]):sum(range_size[:2]),:].reshape(B, -1, C*self.heads[1])
        tem3 = tx[:,sum(range_size[:2]):sum(range_size[:3]),:].reshape(B, -1, C*self.heads[2])
        tem4 = tx[:,sum(range_size[:3]):sum(range_size[:4]),:].reshape(B, -1, C*self.heads[3])

        m1f = self.mixffn1(tem1, self.patch_size[0], self.patch_size[0]).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, self.patch_size[1], self.patch_size[1]).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, self.patch_size[2], self.patch_size[2]).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, self.patch_size[3], self.patch_size[3]).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2


class BridgeLayer_3(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.heads = head
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, bridge_reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn2 = MixFFN(dims*head[1],dims*head[1]*4)
        self.mixffn3 = MixFFN(dims*head[2],dims*head[2]*4)
        self.mixffn4 = MixFFN(dims*head[3],dims*head[3]*4)
        self.patch_size=[56,28,14,7]
        
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c2f, c3f, c4f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        range_size= [self.patch_size[i+1]**2 * self.head[i+1] for i in range(len(self.patch_size)-1)]
        tem2 = tx[:,sum(range_size[:0]):sum(range_size[:1]),:].reshape(B, -1, C*self.heads[1])
        tem3 = tx[:,sum(range_size[:1]):sum(range_size[:2]),:].reshape(B, -1, C*self.heads[2])
        tem4 = tx[:,sum(range_size[:2]):sum(range_size[:3]),:].reshape(B, -1, C*self.heads[3])  

        m2f = self.mixffn2(tem2, self.patch_size[1], self.patch_size[1]).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, self.patch_size[2], self.patch_size[2]).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, self.patch_size[3], self.patch_size[3]).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f], -2)
        
        tx2 = tx1 + t1


        return tx2

class BridegeBlock_X(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios, X):
        super().__init__()
        # 千万要注意这里的bridge_reduction_ratios顺序   是和步骤反着来的
        self.bridge_layers1 = nn.ModuleList([BridgeLayer_0_1(dims, head, bridge_reduction_ratios[2:]) for _ in range(X)])
        self.bridge_layers2 = nn.ModuleList([BridgeLayer_2_3(dims, head, bridge_reduction_ratios[:2]) for _ in range(X)])
        self.patch_size=[56,28,14,7]
        self.heads = head

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x1, x2 = x[:2], x[2:]
        for layer in self.bridge_layers1:
            x1 = layer(x1)
        for layer in self.bridge_layers2:
            x2 = layer(x2)

        B,_,C = x1.shape
        B1,_,C1 = x2.shape

        assert B == B1 and C == C1, "shape error"

        outs = []
        interval1 = self.patch_size[0]**2 * self.heads[0]
        interval2 = self.patch_size[2]**2 * self.heads[2]
        sk1 = x1[:,:interval1,:].reshape(B, self.patch_size[0], self.patch_size[0], C*self.heads[0]).permute(0,3,1,2) 
        sk2 = x1[:,interval1:,:].reshape(B, self.patch_size[1], self.patch_size[1], C*self.heads[1]).permute(0,3,1,2) 
        sk3 = x2[:,:interval2,:].reshape(B, self.patch_size[2], self.patch_size[2], C*self.heads[2]).permute(0,3,1,2) 
        sk4 = x2[:,interval2:,:].reshape(B, self.patch_size[3], self.patch_size[3], C*self.heads[3]).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs

class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, bridge_reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, bridge_reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, bridge_reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, bridge_reduction_ratios)
        self.patch_size=[56,28,14,7]
        self.heads = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
        outs = []

        range_size= [self.patch_size[i]**2 * self.head[i] for i in range(len(self.patch_size))]
        sk1 = bridge4[:,sum(range_size[:0]):sum(range_size[:1]),:].reshape(B, self.patch_size[0], self.patch_size[0], C*self.heads[0]).permute(0,3,1,2) 
        sk2 = bridge4[:,sum(range_size[:1]):sum(range_size[:2]),:].reshape(B, self.patch_size[1], self.patch_size[1], C*self.heads[1]).permute(0,3,1,2) 
        sk3 = bridge4[:,sum(range_size[:2]):sum(range_size[:3]),:].reshape(B, self.patch_size[2], self.patch_size[2], C*self.heads[2]).permute(0,3,1,2) 
        sk4 = bridge4[:,sum(range_size[:3]):sum(range_size[:4]),:].reshape(B, self.patch_size[3], self.patch_size[3], C*self.heads[3]).permute(0,3,1,2) 

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class BridegeBlock_3(nn.Module):
    def __init__(self, dims, head, bridge_reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_3(dims, head, bridge_reduction_ratios)
        self.bridge_layer2 = BridgeLayer_3(dims, head, bridge_reduction_ratios)
        self.bridge_layer3 = BridgeLayer_3(dims, head, bridge_reduction_ratios)
        self.bridge_layer4 = BridgeLayer_3(dims, head, bridge_reduction_ratios)
        self.patch_size=[56,28,14,7]
        self.heads = head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        if (type(x) == list):
            # print("-----1-----")
            outs.append(x[0])
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B,_,C = bridge4.shape
    
        range_size= [self.patch_size[i+1]**2 * self.head[i+1] for i in range(len(self.patch_size)-1)]
        sk2 = bridge4[:,sum(range_size[:0]):sum(range_size[:1]),:].reshape(B, self.patch_size[1], self.patch_size[1], C*self.heads[1]).permute(0,3,1,2) 
        sk3 = bridge4[:,sum(range_size[:1]):sum(range_size[:2]),:].reshape(B, self.patch_size[2], self.patch_size[2], C*self.heads[2]).permute(0,3,1,2) 
        sk4 = bridge4[:,sum(range_size[:2]):sum(range_size[:3]),:].reshape(B, self.patch_size[3], self.patch_size[3], C*self.heads[3]).permute(0,3,1,2) 

        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios,token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            # 只有最后一层的时候，才会有4倍缩放比
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)      
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class MISSFormer(nn.Module):
    def __init__(self, num_classes=9, token_mlp_mode="mix_skip", encoder_pretrained=True):
        super().__init__()
    
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 7 #16 for 512 inputsize   7for 224
        in_out_chan = [[32, 64],[144, 128],[288, 320],[512, 512]]
        patch_size = [56, 28, 14, 7]
        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        self.backbone = MiT(dims, layers, token_mlp_mode)

        self.bridge_reduction_ratios = [1, 2, 4, 8]
        # self.bridge = BridegeBlock_4(dims[0], heads, self.bridge_reduction_ratios)
        self.bridge = BridegeBlock_X(dims[0], heads, self.bridge_reduction_ratios, 6)

        self.decoder_3= MyDecoderLayer((d_base_feat_size*reduction_ratios[3],d_base_feat_size*reduction_ratios[3]), in_out_chan[3], heads[3], reduction_ratios[3],token_mlp_mode, n_class=num_classes)
        self.decoder_2= MyDecoderLayer((d_base_feat_size*reduction_ratios[2],d_base_feat_size*reduction_ratios[2]), in_out_chan[2], heads[2], reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1= MyDecoderLayer((d_base_feat_size*reduction_ratios[1],d_base_feat_size*reduction_ratios[1]), in_out_chan[1], heads[1], reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0= MyDecoderLayer((d_base_feat_size*reduction_ratios[0],d_base_feat_size*reduction_ratios[0]), in_out_chan[0], heads[0], reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

        
    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder) #list

        b,c,_,_ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        #---------------Decoder-------------------------     
        # print("stage3-----")   
        tmp_3 = self.decoder_3(bridge[3].permute(0,2,3,1).view(b,-1,c))
        # print("stage2-----")   
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0,2,3,1))
        # print("stage1-----")   
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0,2,3,1))
        # print("stage0-----")  
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0,2,3,1))

        return tmp_0

         
