import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SSGCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True, drop_path=0., layer_scale_init_value=1e-6):
        super(SSGCN_unit, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        if residual:
            self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=(7,1), padding=(3,0), groups=in_channels) # depthwise conv Twise
        else:
            self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=(4,1), stride=(4,1))
            in_channels = out_channels
        self.norm1 = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.pwconv3 = nn.Linear(in_channels, 5 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.norm2 = LayerNorm(5 * in_channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.pwconv4 = nn.Linear(5 * in_channels, in_channels)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.beta = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # EC
        self.lepe1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0), groups=in_channels, bias=True)
        self.lepe2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels, bias=True)
        self.proj2 = nn.Linear(in_channels, in_channels)
        # PC
        self.cpe = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=(2, 1), stride=(2, 1)),
        )
        else:
            self.down = None

        self.residual = residual
        self.alpha = nn.Parameter(torch.zeros(1))
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, preA=None):
        # Unified Tem Block
        input = x
        x_lepe1 = x
        x = self.dwconv(x)         # (N,C,T,V)

        # EC1
        x_lepe1 = self.lepe1(x_lepe1)
        x = x + x_lepe1

        x = x.permute(0, 2, 3, 1)  # (N, C, T, V) -> (N, T, V, C)
        x = self.norm1(x)
        x = self.pwconv1(x)        # (N, T, V, 4*C)
        x = self.act1(x)
        x = self.pwconv2(x)

        # layer_scale
        if self.gamma is not None:
           x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Res
        if self.residual:
            y = input + self.drop_path(x)
        else:
            y = self.drop_path(x)

        # Unified Spatial Block
        input2 = y
        y = y.permute(0, 2, 3, 1)   # (N, C, T, V) -> (N, T, V, C)
        y = self.pwconv3(y)         # (N, T, V, 5*C)
        y = self.norm2(y)

        # GCN
        N, T, V, C5 = y.shape
        y = y.reshape(N, T, V, -1, 5)
        y1 = y[:, :, :, :, 0:4]
        y2 = y[:, :, :, :, 4:5]
        if preA!=None:
            A = self.PA
        else:
            A = self.PA
        Ak = torch.unsqueeze(A, dim=0).repeat_interleave(N, dim=0)  # .permute(0, 3, 2, 1).contiguous().view(-1,V,3)
        y1 = torch.einsum('niuk,ntkci->ntuci', Ak, y1)
        y = torch.cat([y1, y2], dim=-1)
        y = y.reshape(N, T, V, C5)
        y = self.act2(y)

        y = self.pwconv4(y)
        y = y.permute(0, 3, 1, 2)    # (N, H, W, C) -> (N, C, H, W)

        # EC2
        y_lepe2 = self.lepe2(input2)
        y = y + y_lepe2

        # layer_scale
        y = y.permute(0, 2, 3, 1)   # (N, C, T, V) -> (N, T, V, C)
        if self.beta is not None:
           y = self.beta * y
        y = y.permute(0, 3, 1, 2)    # (N, H, W, C) -> (N, C, H, W)

        # Res
        if self.residual:
            y = input2 + self.drop_path(y)
        else:
            y = self.drop_path(y)

        # PC
        pe = self.cpe(y)
        y = y + pe

        # down
        if self.down != None:
            y = self.down(y)

        return y, A


def mean_merge(x):
    N, C, T, V = x.size()
    x = x.permute(0, 1, 3, 2)  # (N, C, T, V) -> (N, C, V, T)
    x = x.reshape(N, C, V, -1, 2).mean(-1)
    x = x.permute(0, 1, 3, 2)  # (N, C, V, T) -> (N, C, T, V)
    return x


def copy_expansion(att1, att2):
    n, s, h, w = att2.shape
    out = torch.zeros((n, s, h * 2, w * 2))
    if att2.get_device() == -1:
        out = torch.zeros((n, s, h * 2, w * 2))
    elif att2.get_device() in [0, 1, 2, 3]:
        out = torch.zeros((n, s, h * 2, w * 2)).cuda(att2.get_device())
    out[..., 0::2, 0::2] = att2[..., 0::1, 0::1]
    out[..., 0::2, 1::2] = att2[..., 0::1, 0::1]
    out[..., 1::2, 0::2] = att2[..., 0::1, 0::1]
    out[..., 1::2, 1::2] = att2[..., 0::1, 0::1]
    out[..., 2::2, 0::2] = att2[..., 1::1, 0::1]
    out[..., 2::2, 1::2] = att2[..., 1::1, 0::1]
    out[..., 3::2, 0::2] = att2[..., 1::1, 0::1]
    out[..., 3::2, 1::2] = att2[..., 1::1, 0::1]

    att = att1 + out
    return att


class MultiSacleTemAtt(nn.Module):
    def __init__(self, num_subset=4, in_channels=96):
        super(MultiSacleTemAtt, self).__init__()
        self.num_subset = num_subset
        self.in_nets16 = nn.Conv2d(in_channels, 3 * in_channels, 1)
        self.in_nets8 = nn.Conv2d(in_channels, 3 * in_channels, 1)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, padding=0),
            nn.BatchNorm2d(in_channels))

        self.alpha = nn.Parameter(torch.Tensor([1]).requires_grad_())
        self.beta = nn.Parameter(torch.Tensor([1]).requires_grad_())

        self.gelu = nn.GELU()
        self.tan = nn.Tanh()
        self.down = lambda x: x

    def forward(self, x):
        N, C, T, V = x.size()
        mid_dim = C // self.num_subset
        x16 = x
        x8 = mean_merge(x)

        q16, k16, v16 = torch.chunk(self.in_nets16(x16).view(N, 3 * self.num_subset, mid_dim, T, V), 3, dim=1)
        # nctv -> n num_subset c'tv
        attention16 = self.tan(torch.einsum('nsctu,nscru->nstr', [q16, k16]) / (mid_dim * T))

        q8, k8, v8 = torch.chunk(self.in_nets8(x8).view(N, 3 * self.num_subset, mid_dim, T // 2, V), 3, dim=1)
        attention8 = self.tan(torch.einsum('nsctu,nscru->nstr', [q8, k8]) / (mid_dim * T // 2))

        attention = copy_expansion(self.alpha * attention16, self.beta * attention8)

        y = torch.einsum('nstr,nscrv->nsctv', [attention, v16]).contiguous().view(N, C, T, V)
        y = self.ff_nets(y)
        y = self.gelu(self.down(x) + y)

        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_path=0, layer_scale_init_value=1e-6, adaptive=True, unify=None):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A     # 3,25,25
        if unify == "coco":
            from data.unifyposecode import COCO
            vindex = COCO
            A = A[:,vindex]
            A = A[:,:,vindex]
            num_point = len(vindex)
        elif unify == "ntu":
            from data.unifyposecode import NTU
            vindex = NTU
            A = A[:,vindex]
            A = A[:,:,vindex]
            num_point = len(vindex)

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 48    # ucla, ntu120
        # base_channel = 32  # ntu60
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, 9)]
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels,  base_channel, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),)
        self.norm_act1 = nn.Sequential(
            nn.LayerNorm(base_channel),
            nn.GELU(),)

        self.msta = MultiSacleTemAtt(num_subset=4, in_channels=base_channel*2)

        self.merge = nn.Sequential(
            nn.Conv2d(base_channel, base_channel*2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),)
        self.merge_norm = nn.Sequential(
            nn.LayerNorm(base_channel*2),)

        # 4 3 2
        self.l2 = SSGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, drop_path=dp_rates[0], layer_scale_init_value=layer_scale_init_value)
        self.l3 = SSGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, drop_path=dp_rates[1], layer_scale_init_value=layer_scale_init_value)
        self.l4 = SSGCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive, drop_path=dp_rates[2], layer_scale_init_value=layer_scale_init_value)
        self.l5 = SSGCN_unit(base_channel*2, base_channel*4, A, adaptive=adaptive, drop_path=dp_rates[3], layer_scale_init_value=layer_scale_init_value)
        self.l6 = SSGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, drop_path=dp_rates[4], layer_scale_init_value=layer_scale_init_value)
        self.l7 = SSGCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive, drop_path=dp_rates[5], layer_scale_init_value=layer_scale_init_value)
        self.l8 = SSGCN_unit(base_channel*4, base_channel*8, A, adaptive=adaptive, drop_path=dp_rates[6], layer_scale_init_value=layer_scale_init_value)
        self.l9 = SSGCN_unit(base_channel*8, base_channel*8, A, adaptive=adaptive, drop_path=dp_rates[7], layer_scale_init_value=layer_scale_init_value)
        self.l10 = SSGCN_unit(base_channel*8, base_channel*8, A, adaptive=adaptive, drop_path=dp_rates[8], layer_scale_init_value=layer_scale_init_value)
        
        self.norm = nn.LayerNorm(base_channel*8, eps=1e-6)
        self.fc = nn.Linear(base_channel*8, num_class)

        self.in_channels = in_channels
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        if self.in_channels == 2:
            x = x[:,0:2,:,:,:]
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.stem1(x)
        c1 = x.size(1)
        x = x.view(N * M, c1, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act1(x)
        T = T // 2
        x = x.permute(0, 2, 1).contiguous().view(N * M, c1, T, V)

        x = self.merge(x)
        c4 = x.size(1)
        x = x.view(N * M, c4, -1).permute(0, 2, 1).contiguous()
        x = self.merge_norm(x)
        T = T // 2
        x = x.permute(0, 2, 1).contiguous().view(N * M, c4, T, V)

        x1 = self.msta(x)
        x = x1 + x

        x, A1 = self.l2(x)
        x, A2 = self.l3(x, A1)
        x, A3 = self.l4(x, A2)
        x, A4 = self.l5(x, A3)
        x, A5 = self.l6(x, A4)
        x, A6 = self.l7(x, A5)
        x, A7 = self.l8(x, A6)
        x, A8 = self.l9(x, A7)
        x, A9 = self.l10(x, A8)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.norm(x)
        return self.fc(x), x
