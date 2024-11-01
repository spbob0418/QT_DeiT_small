import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
from torch.nn.parameter import Parameter
from _naive_per_tensor_quan_base import _Conv2dQ, Qmodes


__all__ = ['Conv2dQ', 'LinearQ', 'ActQ']


class FunQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        #범위가 벗어나는 quantized weight는 0의 마스크를 씌움 
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
            -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x, dtype):
    y = x.round()
    y_grad = x
    return y.detach().type(dtype) - y_grad.detach() + y_grad


class Conv2dQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, mode=Qmodes.layer_wise, **kwargs):
        super(Conv2dQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.
       
        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, scale_dtype, w_dtype, x_dtype, bias=True, nbits_w=8, nbits_a = 8, **kwargs):
        super(LinearQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias)
        self.act = ActQ(scale_dtype, q_dtype = x_dtype, nbits_a = nbits_a)
        self.nbits = nbits_w
        self.eps = torch.finfo(scale_dtype).eps
        self.scale_dtype = scale_dtype
        self.w_dtype = w_dtype

    def forward(self, x):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1

        max_val = self.weight.abs().max()

        w_scale = max_val / (float(Qp - Qn) / 2)
        w_scale = w_scale.type(self.scale_dtype) #of alpha.to(torch.float16)??

        if w_scale == 0:
            w_scale = self.eps

        w_q = round_pass((self.weight / w_scale).clamp_(Qn, Qp), self.w_dtype) 
  
        x_q , x_scale = self.act(x)

        scale = w_scale * x_scale 

        return F.linear(x_q, w_q, self.bias) , scale 


class ActQ(nn.Module):
    def __init__(self, scale_dtype, q_dtype, nbits_a, **kwargs):
        super(ActQ, self).__init__()
        self.eps = torch.finfo(scale_dtype).eps
        self.scale_dtype = scale_dtype
        self.q_dtype = q_dtype
        self.nbits = nbits_a

    def forward(self, x):

        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1

        max_val = x.abs().max()

        scale = max_val / (float(Qp - Qn) / 2)
        scale = scale.type(self.scale_dtype) 

        if scale == 0:
            scale = self.eps
        # alpha.clamp_(self.eps)

        x_q = round_pass((x / scale).clamp_(Qn, Qp),self.q_dtype) 

        return x_q, scale 
