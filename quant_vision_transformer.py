import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import PatchEmbed 
from timm.models.layers.drop import DropPath
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.weight_init import trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import numpy as np
from probe import process_tensor
# from naive_per_tensor_Quant import *
# from _naive_per_tensor_quan_base import *


_logger = logging.getLogger(__name__)
__all__ = ['qt_deit_small_patch16_224']

def round_pass(x, dtype):
    y = x.round()
    y_grad = x
    return y.detach().type(dtype) - y_grad.detach() + y_grad


class Quantizer():
    def __init__(self, N_bits: int, type: str = "per_tensor",  signed: bool = True, symmetric: bool = True):
        super().__init__()
            
        self.N_bits = N_bits
        self.signed = signed
        self.symmetric = symmetric
        self.q_type = type
        # self.eps = torch.iinfo(dtype).eps
        # self.minimum_range = torch.iinfo(dtype).eps
        if self.N_bits is None:
            return 

        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def __call__(self, x):  
        return self.forward(x)

    def forward(self, x): 
        if self.N_bits is None:
            return x, 1

        if self.symmetric:
            if self.q_type == 'per_tensor': 
                max_x = x.abs().max().detach()
            elif self.q_type == 'per_token': 
                max_x = x.abs().amax(dim=-1, keepdim=True).detach()
            elif self.q_type == 'per_channel': 
                max_x = x.abs().amax(dim=0, keepdim=True).detach()

            print(max_x)
            scale = max_x / self.Qp
            x = x / scale 
            x = round_pass(x.clamp_(self.Qn, self.Qp)) 
            
        else: #Asymmetric
            min_x = x.min().detach()
            max_x = x.max().detach()
            range_x = (max_x - min_x).detach().clamp_(min=self.minimum_range)
            scale = range_x / (self.Qp - self.Qn)

            zero_point = torch.round((min_x / scale) - self.Qn)

            x = (x / scale) + zero_point
            x = round_pass(x.clamp_(self.Qn, self.Qp))

        return x, scale

class QuantAct(nn.Module):
    def __init__(self, 
                 N_bits: int, 
                 type: str , 
                 signed: bool = True, 
                 symmetric: bool = True):
        super(QuantAct, self).__init__()
        self.quantizer = Quantizer(N_bits=N_bits, type = type, signed=signed, symmetric=symmetric)

    def forward(self, x):
        q_x, s_qx = self.quantizer(x)
        return q_x, s_qx

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, weight_grad_quantize_module: Quantizer, act_grad_quantize_module: Quantizer,
                 in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.weight_grad_quantize_module = weight_grad_quantize_module
        self.act_grad_quantize_module = act_grad_quantize_module

    def forward(self, input, s_x):
        return _quantize_global.apply(input, s_x, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.weight_grad_quantize_module, self.act_grad_quantize_module)
    

class _quantize_global(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_3D, s_x, w_2D, bias=None, w_qmodule=None, a_qmodule=None, w_g_qmodule=None, a_g_qmodule=None):

        x_2D = x_3D.view(-1, x_3D.size(-1)) #reshape to 2D
        x_2D = x_2D * s_x 


        weight_quant, s_weight_quant = w_qmodule(w_2D)
        input_quant, s_input_quant = a_qmodule(x_2D)
        ctx.reshape_3D_size = x_3D.size()
        ctx.save_for_backward = (x_2D, 1, w_2D, 1)

        ctx.g_qmodule = w_g_qmodule, a_g_qmodule

        output = torch.matmul(input_quant, weight_quant.t())

        ctx.has_bias = bias is not None
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        s_o = s_weight_quant * s_input_quant 

        return output.view(*x_3D.size()[:-1], -1) * s_o
    
    @staticmethod
    def backward(ctx, g_3D):
        g_2D = g_3D.reshape(-1, g_3D.size(-1))
        grad_X = grad_W = grad_bias = None 
        q_x, s_x, q_w, s_w = ctx.save_for_backward
        w_g_qmodule, a_g_qmodule = ctx.g_qmodule

        reshape_3D = ctx.reshape_3D_size

        
        a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)
        grad_X = torch.matmul(a_g_2D_quant, q_w)
        grad_X = grad_X * a_s_g_2D_quant * s_w 
        grad_X = grad_X.view(reshape_3D)

        w_g_2D_quant, w_s_g_2D_quant = w_g_qmodule(g_2D)
        grad_W = torch.matmul(w_g_2D_quant.t(), q_x)
        grad_W = grad_W * w_s_g_2D_quant * s_x

        if ctx.has_bias:
            grad_bias = g_2D.sum(dim=0)
        else:
            grad_bias = None

        return grad_X, None, grad_W, grad_bias, None, None, None, None



class Mlp(nn.Module):
    def __init__(
            self,
            abits, 
            wbits,
            w_gbits, 
            a_gbits,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=False,
            drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_token'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_channel'),
                                in_features=in_features, 
                                out_features=hidden_features, 
                                bias=True
                                )

        self.act = act_layer()

        self.qact1 = QuantAct(abits, 'per_tensor')

        self.fc2 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_token'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_channel'),
                                in_features=hidden_features, 
                                out_features=out_features, 
                                bias=True
                                )

    def forward(self, x, act_scaling_factor, iteration):
        x = self.fc1(x, act_scaling_factor)
        ##########TODO: Probe ##############first layer LLM은 여기서 outlier가 나온다고 햇음 
        if iteration %200 == 0 :
            process_tensor(x, iteration, layer = 'During MLP (fc1)')
        x = self.act(x)
        x, act_scaling_factor = self.qact1(x)
        x = self.fc2(x, act_scaling_factor)
        #########TODO: Probe ################ second layer LLM 기준으로는 여기서 outlier 안나옴 
        if iteration %200 == 0 :
            process_tensor(x, iteration, layer = 'After MLP (fc2)')
        return x



default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        url='',  # FIXME I have weights for this but > 2GB limit for github release binaries
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}




class Attention(nn.Module):
    def __init__(
            self,
            abits, 
            wbits, 
            w_gbits,
            a_gbits,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)

        self.qkv = Quantized_Linear(
                                    weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                    act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                    weight_grad_quantize_module=Quantizer(w_gbits, 'per_token'),
                                    act_grad_quantize_module=Quantizer(a_gbits, 'per_channel'),
                                    in_features=dim, 
                                    out_features=dim * 3, 
                                    bias=qkv_bias
                                    )
        
        # self.qact1 = QuantAct(nbits, qdtype)
        # self.qact_attn1 = QuantAct(nbits, qdtype)
        self.qact2 = QuantAct(abits, 'per_tensor')
        self.proj = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_token'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_channel'),
                                in_features=dim, 
                                out_features=dim, 
                                bias=True
        )
        self.qact3 = QuantAct(abits, 'per_tensor')
        # self.qact_softmax = QuantAct()

        ##not used yet 
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj_drop = nn.Dropout(proj_drop)
        # self.int_softmax = IntSoftmax(16)

        # self.matmul_1 = QuantMatMul()
        # self.matmul_2 = QuantMatMul()

    def forward(self, x, act_scaling_factor):
        B, N, C = x.shape
        x = self.qkv(x, act_scaling_factor) #quantized input, fp output
        
        # x, act_scaling_factor_1 = self.qact1(x)#Quantize output 

        # x = x * act_scaling_factor_1 #dequantize 

        qkv = x.reshape(B, N, 3, self.num_heads, C //
                        self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)

        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.norm_q(q)
        k = self.norm_k(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x, act_scaling_factor = self.qact2(x)
        x = self.proj(x, act_scaling_factor) #quantized input, fp output
        
        # x = self.proj_drop(x)

        return x

class Q_Block(nn.Module):

    def __init__(self, abits, wbits, w_gbits, a_gbits, dim, num_heads, mlp_ratio=4., 
                #  qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        ######
        self.qact1 = QuantAct(abits, 'per_tensor')
        # self.attn = Q_Attention(nbits_w, nbits_a, dim, num_heads=num_heads)
        ######
        self.attn = Attention(
            abits,
            wbits,
            w_gbits,
            a_gbits,
            dim,
            num_heads=num_heads
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.qact3 = QuantAct(abits, 'per_tensor')
        self.mlp = Mlp(
            abits, 
            wbits, 
            w_gbits, 
            a_gbits,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=None,
            act_layer=act_layer,
        )
        

    def forward(self, x, iteration):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        ###########TODO: Probe ##############
        if iteration %200 == 0 :
            process_tensor(x, iteration, layer = 'Transformer Block Input')
        residual_1 = x
        x = self.norm1(x)
        q_x, s_x = self.qact1(x)
        x = self.attn(q_x, s_x)
        ##########TODO: Probe #############after attention projection
        if iteration %200 == 0 :
            process_tensor(x, iteration, layer = 'After Attention (proj)')

        x = residual_1 + x
        residual_2 = x 

        x = self.norm2(x)
        q_x, s_x = self.qact3(x)
        x = self.mlp(q_x, s_x, iteration) 
       



        x = residual_2 + x
        


        return x


class lowbit_VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, abits, wbits, w_gbits, a_gbits, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            nbits: nbits
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) ##nn.LayerNorm
        act_layer = act_layer or nn.GELU ##GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        #QBlock 
        self.blocks = nn.Sequential(*[
            Q_Block(abits, wbits, w_gbits, a_gbits, embed_dim, num_heads)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        # self.head = LinearQ(self.num_features, num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes, bias=False)
        # nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        # if distilled:
        #     self.head_dist = LinearQ(self.embed_dim, self.num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
            # self.head = LinearQ(self.embed_dim, self.num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
            # nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    ####TODO: TODO: TODO: TODO: 
    def forward_features(self, x, iteration):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x, iteration)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, iteration):
        x = self.forward_features(x, iteration)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        #Non-distillation
        else: 
            x = self.head(x)
        return x

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledVisionTransformer if distilled else VisionTransformer
    model = model_cls(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model, num_classes=num_classes, in_chans=kwargs.get('in_chans', 3),
            filter_fn=partial(checkpoint_filter_fn, model=model))
    return model




@register_model
def qt_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        abits = None, wbits = None, w_gbits = None, a_gbits = None,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model

    
##################

@register_model
def fourbits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        nbits_w = 4, nbits_a = 4,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model



@register_model
def threebits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        nbits_w = 3, nbits_a = 3,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model

@register_model
def twobits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        nbits_w = 2, nbits_a = 2,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model
