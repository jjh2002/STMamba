import math
from dataclasses import dataclass
# from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan
from utils import get_batch_feed_dict

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison.

- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 64  #  N in paper/comments
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 4
    predL: int = 48
    HisL: int = 144
    num_nodes: int = 325
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  #  use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 64)


class embed(nn.Module):
    def __init__(self,config,
                 input_embedding_dim=8,
                 tod_embedding_dim=8,
                 dow_embedding_dim=8,
                 adaptive_embedding_dim=8,
                 step_per_day=23,
                 day_per_week=6,
                 input_dim=1,
                 batch_size=1,
                 model_dim=8 + 8 + 8 + 8,
                 ):
        super(embed, self).__init__()
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.step_per_day = step_per_day
        self.day_per_week = day_per_week
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + adaptive_embedding_dim
                + dow_embedding_dim
        )
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding = nn.Embedding((step_per_day + 1), tod_embedding_dim)
        self.dow_embedding = nn.Embedding((day_per_week + 1), dow_embedding_dim)
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(config.HisL, config.num_nodes, adaptive_embedding_dim))
        )
        self.batch_size = batch_size
        self.input_dim = input_dim

    def forward(self, x):
        tod = x[..., 3]
        dow = x[..., 2]
        x = x[..., : self.input_dim]
        x = self.input_proj(x)
        features = [x]
        tod_emb = self.tod_embedding(
            (tod * self.step_per_day).long()
        )
        features.append(tod_emb)
        dow_emb = self.dow_embedding(
            (dow * self.day_per_week).long()
        )
        features.append(dow_emb)
        adp_emb = self.adaptive_embedding.expand(
            size=(x.shape[0], *self.adaptive_embedding.shape)
        )
        features.append(adp_emb)
        x = torch.cat(features, dim=-1)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class SelfAdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(SelfAdaptiveAdjacencyMatrix, self).__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))  # 源节点嵌入
        self.E2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))  # 目标节点嵌入

    def forward(self):
        # 计算自适应邻接矩阵
        A_adp = torch.matmul(self.E1, self.E2.T)  # (num_nodes, num_nodes)
        A_adp = nn.functional.relu(A_adp)  # 应用ReLU激活函数
        A_adp = nn.functional.softmax(A_adp, dim=1)  # 应用SoftMax归一化
        return A_adp


class AdaptiveConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(AdaptiveConvLayer, self).__init__()
        self.num_nodes = num_nodes
        self.W = nn.Parameter(torch.randn(in_channels, out_channels))  # 可学习的卷积权重
        self.adaptive_adj = SelfAdaptiveAdjacencyMatrix(num_nodes, in_channels)  # 自适应邻接矩阵

    def forward(self, x):
        # x: (B, N, D) -> (B, N, in_channels)
        A_adp = self.adaptive_adj()  # 获取自适应邻接矩阵 (num_nodes, num_nodes)

        # 进行图卷积操作
        # 首先需要对输入数据进行矩阵乘法
        x = torch.matmul(A_adp, x)  # (B, N, D)

        # 进行卷积操作
        x = torch.matmul(x, self.W)  # (B, N, out_channels)
        return x


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.embed = embed(config)
        self.encode = nn.Linear(self.embed.model_dim, config.d_model)
        #         self.autoEncode = SpatioTemporalAutoencoder()
        self.positional_encoding = PositionalEncoding(config.d_model, max_len=config.HisL)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.w_out = nn.Parameter(torch.FloatTensor(config.d_model, 1))
        self.b_out = nn.Parameter(torch.FloatTensor(1))
        self.decoder_cell = nn.LSTMCell(config.d_model, config.d_model, bias=True)  # 44,54
        self.fc = nn.Linear(config.d_model, config.d_model)
        self.linear1 = nn.Linear(config.HisL, 1)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.linear2 = nn.Linear(self.embed.model_dim, config.d_model)
        self.weight_layer = nn.Linear(config.d_model, 1)
        self.gcnlayer = AdaptiveConvLayer(config.d_model, config.d_model, self.config.num_nodes)

    def decoder(self, encoder_output, encoder_state):
        outputs = []
        lastinp = encoder_output
        inp = encoder_state
        inp = inp.transpose(1, 2)
        inp = self.linear1(inp)
        inp = inp.transpose(1, 2)
        inp = self.norm_f(inp)  # x : (B*S, L, D)
        inp = inp[:, -1, :]
        h = inp
        c = inp
        for i in range(self.config.predL):
            h, c = self.decoder_cell(lastinp.float(), (h, c))

            encoder_state = encoder_state[:, 1:, :]
            h1 = h.unsqueeze(1)
            encoder_state = torch.cat((encoder_state, h1), dim=1)
            inp = encoder_state
            inp = inp.transpose(1, 2)
            inp = self.linear1(inp)
            inp = inp.transpose(1, 2)
            weight = torch.sigmoid(self.weight_layer(h))
            inp = weight * h1 + (1 - weight) * inp
            inp = self.norm_f(inp)  # x : (B*S, L, D)
            inp = inp[:, -1, :]
            h = inp
            lastinp = h + lastinp
            lastinp = F.relu(lastinp)
            outputs.append(h)
        return outputs, (h, c)

    def input_transform(self,x):
        local_inputs, labels = x
        local_inputs = torch.chunk(local_inputs, self.config.num_nodes, dim=2)
        local_inputs = torch.cat(local_inputs, dim=0).squeeze(2)
        labels = torch.chunk(labels, self.config.num_nodes, dim=2)
        labels = torch.cat(labels, dim=0).squeeze(2)

        local_inputs = local_inputs.permute(1, 0, 2)  # (144h, batch, features)
        labels = labels.permute(1, 0, 2)  # (48h, batch, features)
        n_input_encoder = local_inputs.data.size(2)  # 1features
        n_output_decoder = labels.data.size(2)
        batch_size = local_inputs.data.size(1)
        _local_inputs = local_inputs.contiguous().view(-1, n_input_encoder)  # (144*batch,features)
        _local_inputs = torch.split(_local_inputs, batch_size, 0)  # 144*（batch, features）
        encoder_inputs = _local_inputs
        # print(encoder_inputs[0].shape,11000)
        _labels = labels.contiguous().view(-1, n_output_decoder)
        _labels = torch.split(_labels, batch_size, 0)  # 48*（batch, 1）
        return encoder_inputs, _labels
    def forward(self, x_batch, y_batch):
        #  x : (B, L, D)

        #  y : (B, L, D)
        x_batch = self.embed(x_batch)
        x = get_batch_feed_dict(0, len(x_batch), x_batch, y_batch)
        x, _labels = self.input_transform(x)  #  x : 144*(B, S, D)            #1 144(B*S,D)
        x = torch.stack(x, dim=1).float()  # 1 (B*S,144,D)
        # print(x.shape, 2222)
        _labels = torch.stack(_labels, dim=1).float()
        # 特征编码层
        encodeFirst = self.encode(x)  # x : (B, L, D) 带噪声的序列数据 #1 (B*S,144,config.d_model)
        encodeFirst = self.positional_encoding(encodeFirst)
        inp = encodeFirst

        Loss = 0
        lastinput = inp[:, -1, :]  # 1 时间最后一步（B*S,1,64)
        # print(inp.shape, 333)
        # inp = IC
        for layer in self.layers:
            #             residual = inp
            inp = layer(inp)
        #             inp = self.ln1(residual+inp)
        BN, L, D = inp.shape[0], inp.shape[1], inp.shape[2]
        B = BN // self.config.num_nodes
        inp = inp.view(B, self.config.num_nodes, L, D)  # (B, N, L, D)
        inp = inp.transpose(1, 2)  # (B, L, N, D)
        inp = inp.contiguous().view(B * L, self.config.num_nodes, D)  # (B * L, N, D)
        inp = self.gcnlayer(inp)
        inp = inp.view(B, L, self.config.num_nodes, D)  # (B, L, N, D)
        inp = inp.transpose(1, 2)  # (B, N, L, D)
        inp = inp.reshape(BN, L, D)
        # inp = inp.transpose(1,2)
        # inp = self.linear1(inp)
        # inp = inp.transpose(1,2)
        # inp = self.norm_f(inp)  # x : (B*S, L, D)
        #         print(inp.shape)
        #         print(lastinput.shape)
        decoder_outputs, states = self.decoder(lastinput, inp)
        preds = [torch.matmul(i, self.w_out) + self.b_out for i in decoder_outputs]
        preds = torch.stack(preds, dim=1)
        #         preds = self.outputlayer(inp)
        # inp = inp[:, -1:, :].repeat(1, self.config.predL, 1) # x : (B, L, S, D)
        # for layer in self.delayers:
        #     y = layer(inp, lastinput)
        # y = self.decode(y)
        # Transformer 解码器
        #         decoder_outputs = self.decoder(inp)
        #         preds = torch.matmul(decoder_outputs, self.w_out) + self.b_out
        return preds, _labels, Loss  # ,encodeFirst,decoded_data

class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mixer = MambaBlock(config)
        self.mixer1 = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.Weight = nn.Linear(2 * config.d_model, config.d_model)
        self.Weight2 = nn.Linear(2 * config.d_model, config.d_model)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(2 * config.d_model, config.d_model),
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)


    def forward(self, x):
        #  x : (B, L, S, D)

        #  output : (B, L, S, D)
        sp_input = torch.split(x, self.config.num_nodes, dim=0)  # 1 x(B*S,144,64) sp_input B*(325,144,64)
        sp_input = torch.stack(sp_input, dim=0)  # 5,325,144,64  #1 (B,325,144,64)
        sp_input = sp_input.permute(0, 2, 1, 3)  # 5,144,325,64  #1 (B,144,325,64)
        sp_input = sp_input.reshape(-1, self.config.num_nodes, self.config.d_model)  # 5*144,325,64  #1 (B*144,325,64)
        st_output = self.mixer1(self.norm(sp_input))  # 5*144,325,64  #1 (B*144,325,64)
        st_output = torch.split(st_output, self.config.HisL, dim=0)  # 1 B*(144,325,64)
        st_output = torch.stack(st_output, dim=0)  # 5,144,325,64  #1 (B,144,325,64)
        st_output = st_output.permute(0, 2, 1, 3)  # 1 (B, 325, 144, 64)
        st_output = st_output.reshape(-1, self.config.HisL, self.config.d_model)  # 1 (B*325,144,64)
        sp_input = torch.split(sp_input, self.config.HisL, dim=0)  # 1 B*(144,325,64)
        sp_input = torch.stack(sp_input, dim=0)  # 5,144,325,64  #1 (B,144,325,64)
        sp_input = sp_input.permute(0, 2, 1, 3)  # 1 (B, 325, 144, 64)
        sp_input = sp_input.reshape(-1, self.config.HisL, self.config.d_model)  # 1 (B*325,144,64)
        residual = sp_input
        output = self.mixer(self.norm(st_output))  # 1 (B*325,144,64)
        out = output
        # embed = torch.cat([output, st_output], dim=2)
        # out = self.Weight(embed)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out


    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class deMambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=2)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))
        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x, lastin):
        #  x : (B, L,S, D)

        # y : (B, L, S,D)

        _, L, _ = x.shape
        out = []
        #  x branch
        xz = self.in_proj(x)  # (B, L, 2*ED)
        ix, z = xz.chunk(2, dim=-1)  #  (B, L, S, ED), (B, L, S, ED)
        ix = ix.transpose(1, 2)  #  (B, ED, L)
        ix = self.conv1d(ix)[:, :, :L]  #  depthwise convolution over time, with a short filter
        ix = ix.transpose(1, 2)  #  (B, L, ED)

        ix = F.silu(ix)
        y = self.ssm(ix, lastin)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)
        return output

    def ssm(self, x, lastin):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        y = self.selective_scan_seq(x, lastin, delta, A, B, C, D)

        return y

    # def selective_scan(self, x, delta, A, B, C, D):
    #     #  x : (B, L, ED)
    #     #  Δ : (B, L, ED)
    #     #  A : (ED, N)
    #     #  B : (B, L, N)
    #     #  C : (B, L, N)
    #     #  D : (ED)
    #
    #     #  y : (B, L, ED)
    #
    #     deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
    #     deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)
    #
    #     BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)
    #
    #     hs = pscan(deltaA, BX)
    #
    #     y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
    #
    #     y = y + D * x
    #
    #     return y

    def selective_scan_seq(self, x, lastin, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)
        batch = 8
        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []
        # lasth = lastin  #batch,1,64

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=2)
        self.adj = nn.Parameter(torch.Tensor(self.config.num_nodes, self.config.num_nodes))
        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))
        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        #  x : (B, L, S, D)

        # y : (B, L, S, D)

        _, L, _ = x.shape
        #  x branch
        xz = self.in_proj(x)  # (B, L, 2*ED)
        ix, z = xz.chunk(2, dim=-1)  #  (B, L, S, ED), (B, L, S, ED)
        ix = ix.transpose(1, 2)  #  (B, ED, L)
        ix = self.conv1d(ix)[:, :, :L]  #  depthwise convolution over time, with a short filter
        ix = ix.transpose(1, 2)  #  (B, L, ED)

        ix = F.silu(ix)
        y = self.ssm(ix)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)
        return output

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    #
    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    # def selective_scan_seq(self, x, delta, A, B, C, D):
    #     #  x : (B, L, ED)
    #     #  Δ : (B, L, ED)
    #     #  A : (ED, N)
    #     #  B : (B, L, N)
    #     #  C : (B, L, N)
    #     #  D : (ED)
    #
    #     #  y : (B, L, ED)
    #
    #     _, L, _ = x.shape
    #
    #     deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
    #     deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)
    #     batch = 8
    #
    #     BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)
    #
    #     h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
    #     hs = []
    #
    #     for t in range(0, L):
    #         extrahList = []
    #         for i in range(batch):
    #             extrah = torch.stack((h[0+i], h[batch+i], h[batch*2+i], h[batch*3+i],h[batch*4+i],h[batch*5+i],h[batch*6+i],h[batch*7+i],h[batch*8+i],h[batch*9+i],h[batch*10+i],h[batch*11+i],h[batch*12+i],h[batch*13+i],h[batch*14+i],h[batch*15+i],h[batch*16+i],h[batch*17+i],h[batch*18+i],h[batch*19+i],h[batch*20+i],h[batch*21+i]
    #                             ,h[batch*22+i],h[batch*23+i],h[batch*24+i],h[batch*25+i],h[batch*26+i],h[batch*27+i],h[batch*28+i],h[batch*29+i],h[batch*30+i],h[batch*31+i],h[batch*32+i],h[batch*33+i],h[batch*34+i]), dim=2)
    #             extrah = torch.matmul(extrah, self.adj).permute(2, 0, 1)
    #             extrahList.append(extrah)
    #         extrah = torch.cat(extrahList, dim=0)
    #         h = deltaA[:, t] * h + extrah + BX[:, t]
    #         hs.append(h)
    #
    #     hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)
    #
    #     y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
    #
    #     y = y + D * x
    #
    #     return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    # def step(self, x, cache):
    #     #  x : (B, D)
    #     #  cache : (h, inputs)
    #     # h : (B, ED, N)
    #     #  inputs : (B, ED, d_conv-1)
    #
    #     #  y : (B, D)
    #     #  cache : (h, inputs)
    #
    #     h, inputs = cache
    #
    #     xz = self.in_proj(x)  # (B, 2*ED)
    #     x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)
    #
    #     #  x branch
    #     x_cache = x.unsqueeze(2)
    #     x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)
    #
    #     x = F.silu(x)
    #     y, h = self.ssm_step(x, h)
    #
    #     #  z branch
    #     z = F.silu(z)
    #
    #     output = y * z
    #     output = self.out_proj(output)  #  (B, D)
    #
    #     # prepare cache for next call
    #     inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
    #     cache = (h, inputs)
    #
    #     return output, cache
    #
    # def ssm_step(self, x, h):
    #     #  x : (B, ED)
    #     #  h : (B, ED, N)
    #
    #     #  y : (B, ED)
    #     #  h : (B, ED, N)
    #
    #     A = -torch.exp(
    #         self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
    #     D = self.D.float()
    #     #  TODO remove .float()
    #
    #     deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)
    #
    #     delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
    #                               dim=-1)  #  (B, dt_rank), (B, N), (B, N)
    #     delta = F.softplus(self.dt_proj(delta))  #  (B, ED)
    #
    #     deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
    #     deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)
    #
    #     BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)
    #
    #     if h is None:
    #         h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
    #
    #     h = deltaA * h + BX  #  (B, ED, N)
    #
    #     y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)
    #
    #     y = y + D * x
    #
    #     #  todo : pq h.squeeze(1) ??
    #     return y, h.squeeze(1)


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
