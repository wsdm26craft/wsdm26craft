import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from torch.utils.data import Sampler
import numpy as np
import pandas as pd
import torch.optim as optim
import math
from typing import Optional
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import copy

import qlib
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from sklearn.metrics import accuracy_score, matthews_corrcoef

def calc_acc_mcc(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    pred = (df['pred']>0).astype(int)
    label = (df['label']>0).astype(int)
    acc = accuracy_score(label, pred)
    mcc = matthews_corrcoef(label, pred)
    return acc, mcc

class WarmUpScheduler(_LRScheduler):
    """
    Args:
        optimizer: [torch.optim.Optimizer] only pass if using as astand alone lr_scheduler
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            eta_min: float = 0.0,
            last_epoch=-1,
            max_lr: Optional[float] = 0.1,
            warmup_steps: Optional[int] = 0,
    ):

        if warmup_steps != 0:
            assert warmup_steps >= 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.step_in_cycle = last_epoch
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup

        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                    for base_lr in self.base_lrs]

        else:
            return [base_lr + (self.max_lr - base_lr) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

        else:
            self.step_in_cycle = self.epoch

        self.max_lr = self.base_max_lr
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnealingWarmRestartsWeightDecay(_LRScheduler):
    """
       Helper class for chained scheduler not to used directly. this class is synchronised with
       previous stage i.e.  WarmUpScheduler (max_lr, T_0, T_cur etc) and is responsible for
       CosineAnealingWarmRestarts with weight decay
       """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_0: int,
            T_mul: float = 1.,
            eta_min: float = 0.001,
            last_epoch=-1,
            max_lr: Optional[float] = 0.1,
            gamma: Optional[float] = 1.,
            min_coef: Optional[float] = 1.0
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.gamma = gamma
        self.min_lr_down_coef = 1.0
        self.coef = min_coef
        self.T_cur = last_epoch  # number of epochs since the last restart
        super(CosineAnealingWarmRestartsWeightDecay, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min
            self.base_lrs.append(self.eta_min)

    def get_lr(self):
        return [
            base_lr * self.min_lr_down_coef + (self.max_lr - base_lr * self.min_lr_down_coef) * (
                    1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mul

        # since warmup steps must be < T_0 and if epoch count > T_0 we just apply cycle count for weight decay
        if self.epoch >= self.T_0:
            if self.T_mul == 1.:
                self.T_cur = self.epoch % self.T_0
                self.cycle = self.epoch // self.T_0
            else:
                n = int(math.log((self.epoch / self.T_0 * (self.T_mul - 1) + 1), self.T_mul))
                self.cycle = n
                self.T_cur = self.epoch - int(self.T_0 * (self.T_mul ** n - 1) / (self.T_mul - 1))
                self.T_i = self.T_0 * self.T_mul ** n

        # base condition that applies original implementation for cosine cycles for details visit:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        else:
            self.T_i = self.T_0
            self.T_cur = self.epoch

        # this is where weight decay is applied
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.min_lr_down_coef = self.coef ** self.cycle
        self.last_epoch = math.floor(self.epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class ChainedScheduler(_LRScheduler):
    """
    Driver class
        Args:
        T_0: First cycle step size, Number of iterations for the first restart.
        T_mul: multiplicative factor Default: -1., A factor increases T_i after a restart
        eta_min: Min learning rate. Default: 0.001.
        max_lr: warmup's max learning rate. Default: 0.1. shared between both schedulers
        warmup_steps: Linear warmup step size. Number of iterations to complete the warmup
        gamma: Decrease rate of max learning rate by cycle. Default: 1.0 i.e. no decay
        last_epoch: The index of last epoch. Default: -1

    Usage:

        ChainedScheduler without initial warmup and weight decay:

            scheduler = ChainedScheduler(
                            optimizer,
                            T_0=20,
                            T_mul=2,
                            eta_min = 1e-5,
                            warmup_steps=0,
                            gamma = 1.0
                        )

        ChainedScheduler with weight decay only:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch=-1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 0,
                            gamma: Optional[float] = 0.9
                        )

        ChainedScheduler with initial warm up and weight decay:
            scheduler = ChainedScheduler(
                            self,
                            optimizer: torch.optim.Optimizer,
                            T_0: int,
                            T_mul: float = 1.0,
                            eta_min: float = 0.001,
                            last_epoch = -1,
                            max_lr: Optional[float] = 1.0,
                            warmup_steps: int = 10,
                            gamma: Optional[float] = 0.9
                        )
    Example:
        >>> model = AlexNet(num_classes=2)
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-1)
        >>> scheduler = ChainedScheduler(
        >>>                 optimizer,
        >>>                 T_0 = 20,
        >>>                 T_mul = 1,
        >>>                 eta_min = 0.0,
        >>>                 gamma = 0.9,
        >>>                 max_lr = 1.0,
        >>>                 warmup_steps= 5 ,
        >>>             )
        >>> for epoch in range(100):
        >>>     optimizer.step()
        >>>     scheduler.step()

    Proper Usage:
        https://wandb.ai/wandb_fc/tips/reports/How-to-Properly-Use-PyTorch-s-CosineAnnealingWarmRestarts-Scheduler--VmlldzoyMTA3MjM2

    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_0: int,
            T_mul: float = 1.0,
            eta_min: float = 0.001,
            last_epoch=-1,
            max_lr: Optional[float] = 1.0,
            warmup_steps: Optional[int] = 10,
            gamma: Optional[float] = 0.95,
            coef: Optional[float] = 1.0,
            step_size: Optional[int] = 2,
            cosine_period: Optional[int] = 3
    ):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mul < 1 or not isinstance(T_mul, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mul))
        if warmup_steps != 0:
            assert warmup_steps < T_0
            warmup_steps = warmup_steps + 1  # directly refers to epoch account for 0 off set

        self.T_0 = T_0
        self.T_mul = T_mul
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.T_i = T_0  # number of epochs between two warm restarts
        self.cycle = 0
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps  # warmup
        self.gamma = gamma
        self.T_cur = last_epoch  # number of epochs since the last restart
        self.last_epoch = last_epoch
        self.coef = coef
        self.cosine_period = cosine_period
        self.cosine_total_steps = T_0 * (1 - T_mul ** cosine_period) / (
                1 - T_mul) if T_mul > 1 else T_0 * self.cosine_period
        # 等比数列求和公式,计算cosine要走多少步

        self.cosine_scheduler1 = WarmUpScheduler(
            optimizer,
            eta_min=self.eta_min,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
        )
        self.cosine_scheduler2 = CosineAnealingWarmRestartsWeightDecay(
            optimizer,
            T_0=self.T_0,
            T_mul=self.T_mul,
            eta_min=self.eta_min,
            max_lr=self.max_lr,
            gamma=self.gamma,
            min_coef=coef,
        )

        self.stepDown_scheduler = StepLR(optimizer, step_size=step_size, gamma=self.coef)

    def get_lr(self):
        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                return self.cosine_scheduler1.get_lr()
        if self.warmup_steps <= self.epoch < self.cosine_total_steps + self.warmup_steps -1:
            return self.cosine_scheduler2.get_lr()

        elif self.epoch >= self.cosine_total_steps + self.warmup_steps - 1:
            return self.stepDown_scheduler.get_lr()

    def step(self, epoch=None):
        self.epoch = epoch
        if self.epoch is None:
            self.epoch = self.last_epoch + 1

        if self.warmup_steps != 0:
            if self.epoch < self.warmup_steps:
                self.cosine_scheduler1.step()
                self.last_epoch = self.epoch

        if self.warmup_steps <= self.epoch < self.cosine_total_steps + self.warmup_steps - 1:
            self.cosine_scheduler2.step()
            self.last_epoch = self.epoch

        elif self.epoch >= self.cosine_total_steps + self.warmup_steps - 1:
            self.stepDown_scheduler.step()
            self.last_epoch = self.epoch

def DLinear_Init(module, min_val=-5e-2, max_val=8e-2):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, min_val, max_val)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size=8, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, kernel_size, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, vocab_size, n_embd, rwkv_emb_scale):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Linear):

                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == vocab_size and shape[1] == n_embd:  # final projection?
                    scale = rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == vocab_size and shape[1] == n_embd:  # token emb?
                    scale = rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale, 2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)  # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)


class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_attn, n_head, ctx_len):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad():  # initial time_w curves for better convergence
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)])  # the distance
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h + 1) / (n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)

        # if .rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn()

        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)  # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 5 * n_ffn // 2  # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv


class RWKV_TinyAttn(nn.Module):  # extra tiny attention
    def __init__(self, n_embd, rwkv_tiny_attn, rwkv_tiny_head):
        super().__init__()
        self.d_attn = rwkv_tiny_attn
        self.n_head = rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head

        self.qkv = nn.Linear(n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(self.head_size))  # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        qk = qk.masked_fill(mask == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        return self.out(qkv)


########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MHA_rotary(nn.Module):
    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.size(-1)))  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # causal mask
        att = F.softmax(att, dim=-1)  # softmax

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 3 * n_ffn
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


########################################################################################################
# MHA_pro: with more tricks
########################################################################################################

class MHA_pro(nn.Module):
    def __init__(self, n_head, ctx_len, n_attn, n_embd, rotary_ndims, head_size, layer_id):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))
        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)  # talking heads

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)  # time-shift mixing
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.size(-1)))  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # causal mask
        att = F.softmax(att, dim=-1)  # softmax
        att = att * w  # time-weighting
        att = self.head_mix(att)  # talking heads

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x


########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed


class FixedNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed


########################################################################################################

class Block(nn.Module):
    def __init__(self, n_embd, n_attn, n_head, ctx_len, n_ffn, hidden_sz, model_type="RWKV", layer_id=1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if model_type == 'RWKV':
            # self.ln1 = FixedNorm(.n_embd)
            # self.ln2 = FixedNorm(.n_embd)
            self.attn = RWKV_TimeMix(layer_id, n_embd, n_attn, n_head, ctx_len)
            self.mlp = RWKV_ChannelMix(layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len)

        elif model_type == 'MHA_rotary':
            self.attn = MHA_rotary(n_embd, n_attn, n_head, ctx_len, layer_id)
            self.mlp = GeGLU(layer_id, n_embd, n_ffn, time_shift=True)

        elif model_type == 'MHA_shift':
            self.attn = MHA_rotary(n_embd, n_attn, n_head, ctx_len, layer_id, time_shift=True)
            self.mlp = GeGLU(layer_id, n_embd, n_ffn, time_shift=True)

        elif model_type == 'MHA_pro':
            self.attn = MHA_pro(n_head, ctx_len, n_attn, n_embd, rotary_ndims=-1, head_size=n_attn, layer_id=layer_id)
            self.mlp = RWKV_ChannelMix(layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(
                qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(
                atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Filter(nn.Module):
    def __init__(self, d_input, d_output, seq_len, kernel=5, stride=5):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.seq_len = seq_len

        self.trans = nn.Linear(d_input, d_output)

        self.aggregate = nn.Conv1d(
            d_output, d_output, kernel_size=kernel, stride=stride, groups=d_output)

        # 输入是[N, T, d_feat]
        conv_feat = math.floor((self.seq_len - kernel) / stride + 1)

        self.proj_out = nn.Linear(conv_feat, 1)

    def forward(self, x):
        x = self.trans.forward(x)  # [N, T, d_feat]
        x_trans = x.transpose(-1, -2)  # [N, d_feat, T]
        x_agg = self.aggregate.forward(x_trans)  # [N, d_feat, conv_feat]
        out = self.proj_out.forward(x_agg)  # [N, d_feat, 1]
        return out.transpose(-1, -2)  # [N, 1, d_feat]


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        # [N, 1, T], [N, T, D] --> [N, 1, D]
        output = torch.matmul(lam, z).squeeze(1)
        return output


class MATCC(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
                 seq_len=8, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.n_attn = d_model
        self.n_head = t_nhead

        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index -
                             gate_input_start_index)  # F'
        self.feature_gate = Filter(self.d_gate_input, self.d_feat, seq_len)

        self.rwkv = Block(layer_id=0, n_embd=self.d_model,
                          n_attn=self.n_attn, n_head=self.n_head, ctx_len=300,
                          n_ffn=self.d_model, hidden_sz=self.d_model)
        RWKV_Init(self.rwkv, vocab_size=self.d_model,
                  n_embd=self.d_model, rwkv_emb_scale=1.0)

        self.dlinear = DLinear(seq_len=seq_len, pred_len=seq_len,
                               enc_in=self.d_model, kernel_size=3, individual=False)
        DLinear_Init(self.dlinear, min_val=-5e-2, max_val=8e-2)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            self.dlinear,  # 【N,T,D】
            self.rwkv,  # 【N,T,D】

            SAttention(d_model=d_model, nhead=s_nhead,
                       dropout=S_dropout_rate),  # [T,N,D]

            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x[N, T, [股票自身的, 市场]]
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, :,
                       self.gate_input_start_index:self.gate_input_end_index]
        src = src + self.feature_gate.forward(gate_input)

        output = self.layers(src).squeeze(-1)

        return output
    
class MATCCModel(Model):
    def __init__(
            self,
            n_epoch = 75,
            lr = 3e-4,
            gamma = 1.0,
            coef = 1.0,
            cosine_period = 4,
            T_0 = 15,
            T_mult = 1,
            warmUp_epoch = 10,
            eta_min = 2e-5,
            weight_decay = 0.001,
            seq_len = 8,
            d_feat = 158,
            d_model = 256,
            n_head = 4,
            dropout = 0.5,
            gate_input_start_index = 158,
            gate_input_end_index = 221,
            train_stop_loss_threshold = 0.95,
            GPU = 0,
            market = 'csi800',
            seed = 0,  # 11031.13031,
            save_path = 'model/',
            save_prefix= ''
    ):
        self.n_epoch = n_epoch
        self.lr = lr
        self.gamma = gamma
        self.coef = coef
        self.cosine_period = cosine_period
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmUp_epoch = warmUp_epoch
        self.eta_min = eta_min
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.d_feat = d_feat
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.train_stop_loss_threshold = train_stop_loss_threshold
        self.GPU = GPU
        self.universe = market
        self.seed = seed
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.device = torch.device(
            f"cuda:{self.GPU}" if torch.cuda.is_available() else "cpu")
        
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = MATCC(
            d_model=self.d_model,
            d_feat=self.d_feat,
            seq_len=self.seq_len,
            t_nhead=self.n_head,
            S_dropout_rate=self.dropout
            ).to(self.device)
        self.fitted = False

        self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(
            0.9, 0.999), weight_decay=self.weight_decay)

        self.lr_scheduler = ChainedScheduler(self.train_optimizer, T_0=self.T_0, T_mul=self.T_mult, eta_min=self.eta_min,
                                        last_epoch=-1, max_lr=self.lr, warmup_steps=self.warmUp_epoch,
                                        gamma=self.gamma, coef=self.coef, step_size=3, cosine_period=self.cosine_period)
    
    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.") 

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)


    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(
            data, sampler=sampler, drop_last=drop_last, num_workers=2, pin_memory=True)
        return data_loader


    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())

            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        self.lr_scheduler.step()

        return float(np.mean(losses))


    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())


        return float(np.mean(losses))
    
    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True


    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)
        
        self.fitted = True
        best_valid_loss = np.Inf

        for step in range(self.n_epoch):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))

            if step <= 10:
                continue

            if (step-10) % 15 == 0:
                best_valid_loss = val_loss
                model_param = copy.deepcopy(self.model.state_dict())
                torch.save(model_param,
                        f'{self.save_path}/{self.save_prefix}matcc_{self.seed}.pth')
    
    def predict(self, dataset: DatasetH, use_pretrained = True):
        if use_pretrained:
            self.load_param(f'{self.save_path}/{self.save_prefix}matcc_{self.seed}.pth')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        pred_all = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            pred_all.append(pred.ravel())


        pred_all = pd.DataFrame(np.concatenate(pred_all), index=dl_test.get_index())
        # pred_all = pred_all.loc[self.label_all.index]
        # rec = self.backtest()
        return pred_all

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index(), dtype=np.float64).groupby(
            "datetime").size().values
        # calculate begin index of each batch
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)

