import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from torchdiffeq import odeint
from torch.nn import functional as F
from datetime import datetime
import sys


class GradMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask


# class ContextNet(nn.Module):


class ContextNet(nn.Module):
    def __init__(self, n_env, context_dim, requires_grad=True, context=None):
        super(ContextNet, self).__init__()
        self.context = None
        self.requires_grad = requires_grad
        self.context_dim = context_dim
        if not requires_grad:
            if context:
                assert (context_dim == context.shape[-1], "Context_dim must fit the shape of context!")
                self.context = torch.nn.Parameter(context, requires_grad=False)
            else:
                raise Exception("Requires value of context")
        else:
            # self.context = torch.nn.Parameter(torch.randn(n_env, context_dim), requires_grad=False)
            self.context = torch.nn.Parameter(torch.randn(n_env, context_dim), requires_grad=True)

    # def forward(self, index):
    #    return torch.index_select(self.context, 0, index)

    def forward(self, index, mask=None):
        if mask is None:
            # print(self.context.device)
            # print(index.device)
            # print(index.shape)
            # print(self.context.size())
            # index = index.squeeze()
            # return torch.mm(index, self.context)
            # print(index)
            index = index.squeeze()
            return torch.index_select(self.context, 0, index)
        if mask.shape[-1] < self.context_dim:
            pad_shape = mask.shape
            pad_shape[-1] = self.context_dim - mask.shape[-1]
            pad = torch.ones(*pad_shape)
            mask = torch.cat(mask, pad, dim=-1)
        elif mask.shape[-1] != self.context_dim:
            raise Exception("Too large mask for context vectors")
        return self.grad_mask(torch.index_select(self.context, 0, index), mask)

    def grad_mask(self, x, mask):
        return GradMask.apply(x, mask)

    def upgrade(self, n_env, device, context=None):
        if context is not None:
            self.context = torch.nn.Parameter(context, requires_grad=False)
        else:
            self.context = torch.nn.Parameter(torch.randn(n_env, self.context_dim), requires_grad=True)

        # self.context = self.context.to(device)

    def set_requires_grad(self, requires_grad=False):
        if self.requires_grad:
            self.context.requires_grad = requires_grad
        else:
            self.context.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, y_dim, r_dim, h_dim, _build_layers):
        super(Encoder, self).__init__()
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.input_to_hidden = _build_layers()

    def forward(self, x, u, y):
        input_pairs = torch.cat((x, y), dim=-1)
        return self.input_to_hidden(input_pairs)


class DAEncoder(nn.Module):
    def __init__(self, y_dim, u_dim, r_dim, h_dim, _build_layers):
        super(DAEncoder, self).__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.input_to_hidden = _build_layers()
        # print(self.input_to_hidden)

    def forward(self, x, u, y):
        batch_size, u_dim = u.shape
        time_points = int(x.shape[0] / batch_size)
        u = u.unsqueeze(1).repeat(1, time_points, 1).view(batch_size * time_points, -1)
        input_pairs = torch.cat((x, u, y), dim=-1)
        return self.input_to_hidden(input_pairs)


class DAConvEncoder(nn.Module):
    def __init__(self, y_dim, u_dim, r_dim, h_dim, img_width, img_height, _build_conv_layers, _build_ctx_layers):
        super(DAConvEncoder, self).__init__()
        assert (img_height * img_width == y_dim, "figure size must remain the same")
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.img_width = img_width
        self.img_height = img_height
        self.conv_layers = _build_conv_layers()
        self.fc1 = _build_ctx_layers()

    def forward(self, x, u, y):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        u : torch.Tensor
            Shape (batch_size, u_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        """
        y = y.view(y.shape[0], self.y_dim, self.img_height, self.img_width)

        u_stack = u.unsqueeze(-1).unsqueeze(-1)
        u_stack = u_stack.repeat(int(y.shape[0] / u.shape[0]), 1, self.img_height, self.img_width)
        # print(u_stack.shape)
        # sys.exit()
        y = torch.cat([y, u_stack], dim=1)

        y = self.conv_layers(y)
        y = y.view(y.size()[0], -1)  # Flatten

        batch_size = u.shape[0]
        num_points = int(x.shape[0] / batch_size)
        u = torch.unsqueeze(u, dim=1)
        # u = u.repeat((1, num_points, 1))
        u = u.repeat(1, num_points, 1)
        u = u.view(x.shape[0], -1)
        xu = torch.cat((x, u), dim=-1)
        input = torch.cat((xu.repeat(1, 128), y), dim=1)
        output = self.fc1(input)
        return output


class ConvEncoder(nn.Module):
    def __init__(self, y_dim, r_dim, img_width, img_height, _build_conv_layers, _build_ctx_layers):
        super(ConvEncoder, self).__init__()
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.r_dim = r_dim
        self.img_height = img_height
        self.img_width = img_width

        self.conv_layers = _build_layers()
        self.fc1 = _build_ctx_layers()

    def forward(self, x, y):
        y = y.view(y.shape[0], 1, self.img_height, self.img_width)
        y = self.conv_layers(y)
        y = y.view(y.size()[0], -1)

        input = torch.cat((x.repeat(1, 128), y), dim=1)
        output = self.fc1(input)
        return output

class MlpY0Encoder(nn.Module):
    def __init__(self, y_dim, h_dim, r_dim):
        super(MlpY0Encoder, self).__init__()
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.fc = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, r_dim)
        )

    def forward(self, y0):
        return self.fc(y0)

class Y0ConvEncoder(nn.Module):
    def __init__(self, y_dim, r_dim, img_width, img_height, _build_conv_layers, _build_ctx_layers):
        super(DAY0ConvEncoder, self).__init__()
        n_filt = 8
        self.n_filt = n_filt
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.img_height = img_height
        self.img_width = img_width

        self.conv_layers = _build_layers()
        self.fc1 = _build_ctx_layers()

    def forward(self, u, y0):
        """
        u : torch.Tensor
            Shape (batch_size, u_dim)
        y0 : torch.Tensor
             Shape (batch_size, y_dim)
        """
        y = y0.view(y0.shape[0], 1, self.img_height, self.img_width)
        y = self.conv_layers(y)
        y = y.view(y.size()[0], -1)
        output = self.fc1(y)
        return output


class DAY0ConvEncoder(nn.Module):
    def __init__(self, y_dim, u_dim, r_dim, img_width, img_height, _build_conv_layers, _build_ctx_layers):
        super(DAY0ConvEncoder, self).__init__()
        n_filt = 8
        self.n_filt = n_filt
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.img_height = img_height
        self.img_width = img_width

        self.conv_layers = _build_layers()
        self.fc1 = _build_ctx_layers()

    def forward(self, u, y0):
        """
        u : torch.Tensor
            Shape (batch_size, u_dim)
        y0 : torch.Tensor
             Shape (batch_size, y_dim)
        """
        y = y0.view(y0.shape[0], 1, self.img_height, self.img_width)
        y = self.conv_layers(y)
        y = y.view(y.size()[0], -1)
        input = torch.cat((u.repeat(1, self.n_filt * 8), y), dim=1)
        output = self.fc1(input)
        return output


class DASingleContextNDPEncoder(nn.Module):
    def __init__(self, context_encoder):
        super(DASingleContextNDPEncoder, self).__init__()
        self.context_encoder = context_encoder

    def forward(self, x, u, y, _):
        output = self.context_encoder(x, u, y)
        return output, output


class DAY0ContextNDPEncoder(nn.Module):
    def __init__(self, y0_encoder, context_encoder, adaptive_y0=False, adaptive_encoder=False):
        super(DAY0ContextNDPEncoder, self).__init__()
        self.y0_encoder = y0_encoder
        self.context_encoder = context_encoder
        self.adaptive_y0 = adaptive_y0
        self.adaptive_encoder = adaptive_encoder

    def forward(self, x, u, y, y0):
        """
        x : torch.Tensor
            Shape (batch_size * num_points, x_dim)
        u : torch.Tensor
            Shape (batch_size, u_dim)
        y : torch.Tensor
            Shape (batch_size * num_points, y_dim)
        y0 : torch.Tensor
            Shape (batch_size, y_dim)
        """
        if self.adaptive_y0:
            L_output = self.y0_encoder(u, y0)
        else:
            L_output = self.y0_encoder(y0)
        if self.adaptive_encoder:
            D_output = self.context_encoder(x, u, y)
        else:
            D_output = self.context_encoder(x, y)
        # print("Encode complete")
        return L_output, D_output


class DAConvDecoderNet(nn.Module):
    def __init__(self, y_dim, feat_width, feat_height, _build_conv_trans_layers):
        super(DAConvDecoderNet, self).__init__()
        self.y_dim = y_dim
        self.feat_width = feat_width
        self.feat_height = feat_height
        self.conv_trans_layers = _build_conv_trans_layers()

    def forward(self, u, x):
        x = x.view(x.shape[0], 8, self.feat_height, self.feat_width)

        u_stack = u.unsqueeze(-1).unsqueeze(-1)
        u_stack = u_stack.repeat(int(x.shape[0] / u.shape[0]), 1, self.feat_height, self.feat_width)
        x = torch.cat([x, u_stack], dim=1)

        x = self.conv_trans_layers(x)
        return x


class ConvDecoderNet(nn.Module):
    def __init__(self, y_dim, feat_width, feat_height, _build_conv_trans_layers):
        super(ConvDecoderNet, self).__init__()
        self.y_dim = y_dim
        self.feat_width = feat_width
        self.feat_height = feat_height
        self.conv_trans_layers = _build_conv_trans_layers()

    def forward(self, u, x):
        x = x.view(x.shape[0], 8, self.feat_height, self.feat_width)
        x = self.conv_trans_layers(x)
        return x


class AbstractODEDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(AbstractODEDecoder, self).__init__()
        # The input is always time.
        assert x_dim == 1

        self.exclude_time = exclude_time
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        #print(f"In AbstractODEDecoder, L_dim = {self.L_dim}")
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim
        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]
        # z = [L0, z_] so dim([L, z_, x]) = dim(z)+1
        self.latent_odefunc = nn.Sequential(*ode_layers)

        # self.decode_layers = [nn.Linear(x_dim + z_dim, h_dim),
        #                      nn.ReLU(inplace=True),
        #                      nn.Linear(h_dim, h_dim),
        #                      nn.ReLU(inplace=True),
        #                      nn.Linear(h_dim, h_dim),
        #                      nn.ReLU(inplace=True)]
        self.decode_layers = [nn.Linear(x_dim + z_dim, h_dim),
                              nn.ReLU(),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU()]
        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)

        self.initial_t = initial_t
        self.nfe = 0

    def integrate_ode(self, t, v):  # v = (L(x), z_)
        self.nfe += 1
        z_ = v[..., self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            # print(f"In AbstractODEDecoder, vt shape = {vt.shape}")
            # print(f"In AbstractODEDecoder, time shape = {time.shape}")
            vt = torch.cat((vt, time), dim=1)
        dL = self.latent_odefunc(vt)
        # print("dL shape=",dL.size())
        dz_ = torch.zeros_like(z_)
        return torch.cat((dL, dz_), dim=-1)

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        raise NotImplementedError('The decoding of the latent ODE state is not implemented')

    def forward(self, x, z):
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        # Append the initial time to the set of supplied times.
        x0 = self.initial_t.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        # ind specifies where each element in x ended up in times.
        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)
        # Remove the initial position index since we don't care about it.
        ind = ind[:, 1:, :]

        # Integrate forward from the batch of initial positions z.
        v = odeint(self.integrate_ode, z, times, method='dopri5')
        v = v.squeeze()
        # print("ODE solve complete")

        # Make shape (batch_size, unique_times, z_dim).
        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        # Extract the relevant (latent, time) pairs for each batch.
        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)
        # print("latent size=",latent.size())

        return self.decode_latent(x, z, latent)


class AbstractODEMetaDecoder(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=None):
        super(AbstractODEMetaDecoder, self).__init__()
        assert x_dim == 1

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        self.exclude_time = exclude_time
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim
        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]

        self.latent_odefunc = nn.Sequential(*ode_layers)

        self.decode_layers = [nn.Linear(x_dim + u_dim + z_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim)]

        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)

        ctx_layers = [nn.Linear(u_dim + z_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, z_dim)]

        self.ctx_net = nn.Sequential(*ctx_layers)

        self.initial_t = initial_t
        self.nfe = 0

    def integrate_ode(self, t, v):
        self.nfe += 1
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        dL = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)

        return torch.cat((dL, dz_), dim=1)

    def decode_latent() -> torch.distributions.Distribution:
        raise NotImplementedError

    def forward(self, x, u, z):
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        x0 = self.initial_t.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)

        ind = ind[:, 1:, :]

        # time1 = datetime.now()
        # print(u.shape)
        # print(z.shape)
        # dz = self.ctx_net(torch.cat([z, u], dim=-1))
        # z = z + dz
        z = self.ctx_net(torch.cat([z, u], dim=-1))
        v = odeint(self.integrate_ode, z, times, method="rk4")
        # time2 = datetime.now()
        # print("ODE time:", (time2 - time1).seconds)

        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)

        return self.decode_latent(x, u, z, latent)


class MlpNormalODEDecoder(AbstractODEDecoder):
    def __init__(self, x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(MlpNormalODEDecoder, self).__init__(x_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim, exclude_time)

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        # print(z.shape)
        # z = z[:, self.L_dim:]
        # z = z[:, self.L_dim:]
        z = z[..., self.L_dim:]
        z = z.repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        input_triplets = torch.cat((x_flat, latent_flat, z_flat), dim=-1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=-1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class MlpNormalODEMetaDecoder(AbstractODEMetaDecoder):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(MlpNormalODEMetaDecoder, self).__init__(x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
                                                      exclude_time=exclude_time)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, u, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        u_flat = u.unsqueeze(1).repeat(1, num_points, 1).view(batch_size * num_points, self.u_dim)
        input_triplets = torch.cat((x_flat, u_flat, latent_flat, z_flat), dim=-1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=-1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class ConvODEMetaDecoder(AbstractODEMetaDecoder):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t, conv_decoder_net, feat_width, feat_height,
                 latent_only=False, exclude_time=False, binary=False):
        super(ConvODEMetaDecoder, self).__init__(x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
                                                 exclude_time=exclude_time)
        self.latent_only = latent_only
        # self.feat = 3 if int(np.sqrt(y_dim)) == 28 else 4
        # self.u_dim = u_dim
        # self.u_dim = u_dim
        self.feat_w = feat_width
        self.feat_h = feat_height
        self.binary = binary

        if self.latent_only:
            # self.decode_fc = nn.Linear(L_dim, self.feat * self.feat * 8)
            self.decode_fc = nn.Sequential(nn.Linear(L_dim, h_dim),
                                           nn.ReLU(),

                                           nn.Linear(h_dim, h_dim),
                                           nn.ReLU(),

                                           nn.Linear(h_dim, self.feat_w * self.feat_h * 8))
        else:
            # self.decode_fc = nn.Linear(x_dim + u_dim + z_dim, self.feat * self.feat * 8)
            self.decode_fc = nn.Sequential(nn.Linear(x_dim + u_dim + z_dim, h_dim),
                                           nn.ReLU(),

                                           nn.Linear(h_dim, h_dim),
                                           nn.ReLU(),

                                           nn.Linear(h_dim, self.feat_w * self.feat_h * 8))

        self.conv_decoder = conv_decoder_net

    def decode_latent(self, x, u, z, latent):
        batch_size, num_points, _ = x.size()
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        u_flat = u.unsqueeze(1).repeat(1, num_points, 1).view(batch_size * num_points, self.u_dim)

        # print("x_flat size=",x_flat.size())
        # print("u_flat size=",u_flat.size())

        input = latent if self.latent_only else torch.cat((x_flat, u_flat, latent_flat, z_flat), dim=1)
        input = self.decode_fc(input)

        output = self.conv_decoder(u, input)
        if self.binary:
            mu = output
            mu = mu.view(batch_size, num_points, self.y_dim)
            return Bernoulli(logits=mu)
        else:
            mu, pre_sigma = torch.chunk(output, 2, dim=1)
            mu = mu.view(batch_size, num_points, self.y_dim)
            pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
            sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
            return Normal(mu, sigma)

        # mu_sigma = self.conv_decoder(input)
        # mu,pre_sigma = torch.split(mu_sigma, 1, dim=1)
        # print("mu_sigma shape=",mu_sigma.shape)
        # print("mu shape=",mu.shape)
        # print("sigma shape=",pre_sigma.shape)
        # sys.exit()
        # print(f"batch_size={batch_size}")
        # print(f"num_points={num_points}")
        # print(f"y_dim={self.y_dim}")
        # mu = mu.view(batch_size, num_points, self.y_dim)
        # mu = F.sigmoid(mu)
        # mu = F.sigmoid((mu - torch.mean(mu, dim=-1, keepdim=True))/torch.mean(mu, dim=-1, keepdim=True))
        # mu = F.sigmoid((mu - torch.mean(mu, dim=-1, keepdim=True)))
        # mu = torch.exp(mu)
        # mu = mu / torch.norm(mu, p=float("inf"), dim=-1, keepdim=True)
        # mu = F.normalize(mu, p=inf, )
        # return Bernoulli(logits=mu)


class AbstractODEDADecoder(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=None):
        super(AbstractODEDADecoder, self).__init__()
        assert x_dim == 1

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim
        self.exclude_time = exclude_time
        if L_out_dim is None:
            L_out_dim = L_dim

        inp_dim = z_dim if exclude_time else z_dim + x_dim
        ode_layers = [nn.Linear(inp_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, h_dim),
                      nn.Tanh(),
                      nn.Linear(h_dim, L_out_dim)]

        self.latent_odefunc = nn.Sequential(*ode_layers)

        self.decode_layers = [nn.Linear(x_dim + u_dim + z_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim),
                              nn.ReLU(inplace=True),
                              nn.Linear(h_dim, h_dim)]

        self.xlz_to_hidden = nn.Sequential(*self.decode_layers)

        self.initial_t = initial_t
        self.nfe = 0

    def integrate_ode(self, t, v):
        self.nfe += 1
        z_ = v[:, self.L_dim:]
        batch_size = v.size()[0]
        vt = v
        if not self.exclude_time:
            time = t.view(1, 1).repeat(batch_size, 1)
            vt = torch.cat((vt, time), dim=1)

        dL = self.latent_odefunc(vt)
        dz_ = torch.zeros_like(z_)

        return torch.cat((dL, dz_), dim=1)

    def decode_latent() -> torch.distributions.Distribution:
        raise NotImplementedError

    def forward(self, x, u, z):
        self.nfe = 0
        batch_size, num_points, _ = x.size()

        x0 = self.initial_t.repeat(batch_size, 1, 1)
        x_sort = torch.cat((x0, x), dim=1)

        times, ind = torch.unique(x_sort, sorted=True, return_inverse=True)

        ind = ind[:, 1:, :]

        # time1 = datetime.now()
        v = odeint(self.integrate_ode, z, times, method="rk4")
        # time2 = datetime.now()
        # print("ODE time:", (time2 - time1).seconds)

        permuted_v = v.permute(1, 0, 2)
        latent = permuted_v[:, :, :self.L_dim]

        tiled_ind = ind.repeat(1, 1, self.L_dim)
        latent = torch.gather(latent, dim=1, index=tiled_ind)

        return self.decode_latent(x, u, z, latent)


class MlpNormalODEDADecoder(AbstractODEDADecoder):
    def __init__(self, x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t, L_out_dim=None, exclude_time=False):
        super(MlpNormalODEDADecoder, self).__init__(x_dim, u_dim, z_dim, h_dim, y_dim, L_dim, initial_t,
                                                    exclude_time=exclude_time)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.L_dim = L_dim

        self.hidden_to_mu = nn.Linear(h_dim + L_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim + L_dim, y_dim)
        self.initial_t = initial_t
        self.nfe = 0

    def decode_latent(self, x, u, z, latent) -> torch.distributions.Distribution:
        batch_size, num_points, _ = x.size()
        z = z[:, self.L_dim:]
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latent_flat = latent.view(batch_size * num_points, -1)
        z_flat = z.view(batch_size * num_points, self.z_dim - self.L_dim)
        u_flat = u.unsqueeze(1).repeat(1, num_points, 1).view(batch_size * num_points, self.u_dim)
        input_triplets = torch.cat((x_flat, u_flat, latent_flat, z_flat), dim=-1)
        hidden = self.xlz_to_hidden(input_triplets)
        hidden = torch.cat((latent_flat, hidden), dim=-1)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return Normal(mu, sigma)


class MuSigmaEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(pre_sigma)
        #sigma = pre_sigma * pre_sigma
        #sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class MlpDiscriminator(nn.Module):
    def __init__(self, z_dim, h_dim, n_env):
        super(MlpDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh(),

            nn.Linear(h_dim, h_dim),
            nn.Tanh(),

            nn.Linear(h_dim, n_env),
            nn.LogSoftmax()
        )

    def forward(self, z):
        return self.layers(self.grad_reverse(z))

    def grad_reverse(self, x):
        return GradReverse.apply(x)


class SepMlpDiscriminator(nn.Module):
    def __init__(self, z_dim, L_dim, h_dim, n_env):
        super(SepMlpDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim - L_dim, h_dim),
            nn.Tanh(),

            nn.Linear(h_dim, h_dim),
            nn.Tanh(),

            nn.Linear(h_dim, n_env),
            nn.LogSoftmax()
        )

        self.L_dim = L_dim

    def forward(self, z):
        d = z[..., self.L_dim:]
        return self.layers(self.grad_reverse(d))

    def grad_reverse(self, x):
        return GradReverse.apply(x)