from torch.nn import Module as TorchModule
from models.models import *
import math
import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions import Normal


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalEncoding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * - math.log(10000.0) / d_model)
        self.div_term = torch.unsqueeze(self.div_term, 0)
        self.pe = None

    def forward(self, x, y):
        x = x.view(-1, 1)
        #print(f"In Temporal Encoding, before norm, x={x}")
        if torch.max(x) != torch.min(x):
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) * 200
        else:
            x = torch.zeros_like(x).to(y.device)
        #print(f"In Temporal Encoding, after norm, x={x}")
        pe = torch.zeros(x.size(0), self.d_model)
        pe[:, 0::2] = torch.sin(torch.matmul(x, self.div_term.to(x.device)))
        pe[:, 1::2] = torch.cos(torch.matmul(x, self.div_term.to(x.device)))
        pe = pe.to(y.device)
        #pe = pe.unsqueeze(0).transpose(0, 1).to(y.device)
        y = y + pe
        y = self.dropout(y)
        return y


class Attention(nn.Module):
    def __init__(self, d_input, d_model, heads, adaptive=False):
        super(Attention, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.heads = heads
        self.d_heads = d_model // heads
        # self.adaptive = adaptive

        self.q_linear = nn.Linear(self.d_input, self.d_model)
        self.v_linear = nn.Linear(self.d_input, self.d_model)
        self.k_linear = nn.Linear(self.d_input, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

        self.temporal_encoding = TemporalEncoding(d_input)
        # self.temporal_encoding = PositionalEncoding()

    def forward(self, x, y, Wq_e=None, mask=None):
        bs = x.size(0)
        # k = self.k_linear(x).view(bs, -1, self.heads, self.d_head)
        q = self.q_linear(y).view(bs, -1, self.heads, self.d_heads)
        # v = self.v_linear(x).view(bs, -1, self.heads, self.d_head)
        # if self.adaptive:
        #    q = q + torch.matmul(Wq_e, x)
        #y_hat = self.temporal_encoding(x, y)
        y_hat = y
        #print(f"In Attention, y_hat={y_hat}")
        k = self.k_linear(y_hat).view(bs, -1, self.heads, self.d_heads)
        v = self.v_linear(y_hat).view(bs, -1, self.heads, self.d_heads)
        if Wq_e is not None:
           q = q + torch.unsqueeze(torch.bmm(torch.unsqueeze(y, 1), Wq_e), 1)
           #q = torch.unsqueeze(torch.bmm(torch.unsqueeze(y, 1), Wq_e), 1)
           #q = torch.unsqueeze(torch.bmm(torch.unsqueeze(y, 1), Wq_e), 1)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, mask)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        output = torch.squeeze(output)

        return output

    def attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_heads)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        return output


class CoY0ContextNDPEncoder(nn.Module):
    def __init__(self, y0_encoder, context_encoder):
        super(CoY0ContextNDPEncoder, self).__init__()
        self.y0_encoder = y0_encoder
        self.context_encoder = context_encoder

    def forward(self, Wq_e, x, y, y0):
        L_output = self.y0_encoder(y0)
        D_output = self.context_encoder(Wq_e, x, y)
        return L_output, D_output


class AttentionEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, r_dim, n_layers):
        super(AttentionEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.adaptive_attention = Attention(d_input=y_dim, d_model=h_dim, heads=1)

        attention_list = []
        for _ in range(n_layers - 1):
            attention_list.append(Attention(d_input=h_dim, d_model=h_dim, heads=1))
        self.attention_layers = nn.ModuleList(attention_list)

    def forward(self, Wq_e, x, y):
        h = self.adaptive_attention(x, y, Wq_e)
        for layer in self.attention_layers:
            h = layer(x, h)
        #h = self.attention_layers(x, h)
        return h


class HyperNet(nn.Module):
    def __init__(self, u_dim, h_dim, para_dim, n_layers):
        super(HyperNet, self).__init__()
        self.u_dim = u_dim
        self.h_dim = h_dim
        self.para_dim = para_dim
        self.n_layers = n_layers

        self.net = self.makenet()

    def makenet(self):
        net = nn.Sequential()
        if self.n_layers == 1:
            net.add_module("linear", nn.Linear(self.u_dim, self.para_dim))
        else:
            net.add_module("linear0", nn.Linear(self.u_dim, self.h_dim))
            net.add_module("relu0", nn.ReLU(True))
            for i in range(1, self.n_layers - 2):
                net.add_module(f"linear{i}", nn.Linear(self.h_dim, self.h_dim))
                net.add_module(f"relu{i}", nn.ReLU(True))
            net.add_module(f"linear{i}", nn.Linear(self.h_dim, self.para_dim))
        #net.add_module("sigmoid", nn.Sigmoid())
        return net

    def forward(self, u):
        return self.net(u)


class ContextTimeNeuralProcess(nn.Module):
    def __init__(self, y_dim, r_dim, z_dim, h_dim, context_net: TorchModule, hypernet: TorchModule,
                 encoder: TorchModule, decoder: TorchModule):
        super(ContextTimeNeuralProcess, self).__init__()
        self.x_dim = 1
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.h_dim = h_dim

        # Initialize networks
        self.context_net = context_net
        self.hypernet = hypernet
        self.xy_to_r = encoder
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = decoder
        self.aggregator = nn.Sequential(
            nn.Linear(r_dim, r_dim),
            nn.Tanh(),
        )
        self.temporal_encoding = TemporalEncoding(d_model=h_dim)

    def aggregate(self, x, r_i):
        batch_size, num_points, _ = x.size()
        a = self.aggregator(torch.mean(r_i, dim=-2))
        return a
        r_i_flat = r_i.view(batch_size * num_points, -1)
        r_hat = self.temporal_encoding(x, r_i_flat)
        a = a.unsqueeze(-1)
        r_hat = r_hat.view(batch_size, num_points, -1)
        w = torch.bmm(r_hat, a).squeeze(-1).unsqueeze(1)
        # r = torch.mean(F.tanh(torch.bmm(w, r_hat)), dim=-2)
        r = F.tanh(torch.bmm(w, r_hat))
        return r

    def xy_to_mu_sigma(self, Wq_e, x, y, y0):
        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        r_i_flat = self.xy_to_r(Wq_e, x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        r = self.aggregate(x, r_i)
        # print(f"In xy_to_mu_sigma, r={r}")
        return self.r_to_mu_sigma(r)

    def forward(self, e, x_context, y_context, x_target, y_target=None, y0=None):
        _, num_target, _ = x_target.size()
        #_, _, y_dim = y_context.size()

        if self.training:
            q_e = self.context_net(e)
            #print(f"In ContextTimeNeuralProcess, q_e size={q_e.size()}")
            Wq_e = self.hypernet(q_e)
            mu_target, sigma_target = self.xy_to_mu_sigma(Wq_e, x_target, y_target, y0)
            mu_context, sigma_context = self.xy_to_mu_sigma(Wq_e, x_context, y_context, y0)
            q_target = Normal(mu_target, sigma_target)
            #print("mu_context=", mu_context)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            p_y_pred = self.xz_to_y(x_target, z_sample)
            return p_y_pred, q_target, q_context
        else:
            #print(f"While testing, e={e}")
            q_e = self.context_net(e)
            Wq_e = self.hypernet(q_e)
            mu_context, sigma_context = self.xy_to_mu_sigma(Wq_e, x_context, y_context, y0)
            #print(f"In forward, sigma_context={sigma_context}")
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            p_y_pred = self.xz_to_y(x_target, z_sample)
            return p_y_pred


class ContextNeuralODEProcess(ContextTimeNeuralProcess):
    def __init__(self, y_dim, r_dim, h_dim, L_dim, D_dim, context_net: TorchModule, hypernet: TorchModule, encoder: TorchModule,
                 decoder: TorchModule):
        z_dim = L_dim + D_dim
        super(ContextNeuralODEProcess, self).__init__(y_dim, r_dim, z_dim, h_dim, context_net, hypernet, encoder, decoder)

        #self.context_net = context_net
        self.r_to_mu_sigma = None
        self.L_r_to_mu_sigma = MuSigmaEncoder(r_dim, L_dim)
        self.D_r_to_mu_sigma = MuSigmaEncoder(r_dim, D_dim)
        #self.D_r_to_mu_sigma = nn.Linear(r_dim, D_dim)
        #self.temporal_encoding = TemporalEncoding(d_model=D_dim)

    def xy_to_mu_sigma(self, Wq_e, x, y, y0):
        batch_size, num_points, _ = x.size()
        Wq_e = torch.unsqueeze(Wq_e, 1).repeat(1, num_points, 1)
        Wq_e_flat = Wq_e.view(batch_size * num_points, self.y_dim, self.h_dim)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        L_r_i_flat, D_r_i_flat = self.xy_to_r(Wq_e_flat, x_flat, y_flat, y0)
        #print(f"In ContextNeuralODEProcess, D_r_i_flat={D_r_i_flat}")
        if L_r_i_flat.size(0) == batch_size:
            L_r_i = L_r_i_flat.view(batch_size, 1, self.r_dim)
        else:
            L_r_i = L_r_i_flat.view(batch_size, num_points, self.r_dim)
        D_r_i = D_r_i_flat.view(batch_size, num_points, self.r_dim)
        # L_r = self.aggregate(L_r_i)
        L_r = L_r_i
        D_r = self.aggregate(x, D_r_i)
        #print(f"In ContextNeuralODEProcess, D_r={D_r}")
        L_mu, L_sigma = self.L_r_to_mu_sigma(L_r)
        D_mu, D_sigma = self.D_r_to_mu_sigma(D_r)
        L_mu = L_mu.squeeze(1)
        L_sigma = L_sigma.squeeze(1)
        mu = torch.cat([L_mu, D_mu], dim=-1)
        sigma = torch.cat([L_sigma, D_sigma], dim=-1)
        return mu, sigma


def makecoprocess(dataset, initial_t, n_env):
    x_dim = 1
    if dataset in ["LV"]:
        y_dim_dict = {}
        y_dim_dict["LV"] = 2
        y_dim = y_dim_dict[dataset]
        u_dim = 4
        r_dim = 100
        z_dim = 50
        h_dim = 100
        L_dim = 25
        D_dim = z_dim - L_dim
        n_layers = 4
        context_net = ContextNet(n_env=n_env, context_dim=u_dim)
        hypernet = HyperNet(u_dim=u_dim, h_dim=h_dim, para_dim=h_dim * y_dim, n_layers=n_layers)
        context_encoder = AttentionEncoder(x_dim=x_dim, y_dim=y_dim, h_dim=h_dim, r_dim=r_dim, n_layers=4)
        y0_encoder = MlpY0Encoder(y_dim=y_dim, h_dim=h_dim, r_dim=r_dim)
        encoder = CoY0ContextNDPEncoder(y0_encoder=y0_encoder, context_encoder=context_encoder)
        decoder = MlpNormalODEDecoder(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, h_dim=h_dim, L_dim=L_dim,
                                      initial_t=initial_t, exclude_time=True)
        return ContextNeuralODEProcess(y_dim, r_dim, h_dim, L_dim, D_dim, context_net, hypernet, encoder, decoder)