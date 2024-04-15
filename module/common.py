"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(input_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, latent_dim) # mean
        self.enc_layer2_logvar = nn.Linear(hidden_dim, latent_dim) # variance

        # Define decoding layers
        self.dec_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # h1 = F.relu(self.enc_layer1(x))
        # return self.enc_layer2_mu(h1), self.enc_layer2_logvar(h1)
        h1 = F.relu(self.enc_layer1(x))
        # mu = F.relu(self.enc_layer2_mu(h1))
        # logvar = F.relu(self.enc_layer2_logvar(h1))
        mu = self.enc_layer2_mu(h1)
        logvar = self.enc_layer2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.dec_layer1(z))
        x = torch.sigmoid(self.dec_layer2(h3))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_dim, z_dim, pretrain_category_weight=None):
        super(VQVAE, self).__init__()
        self.z_dim = z_dim
        self.k_dim = k_dim

        # Encoder MLP
        self.encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

        # Embedding Book
        # self.embd = nn.Embedding(self.k_dim, self.z_dim).cuda()
        self.embd = pretrain_category_weight

        # Decoder MLP
        self.decode = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def find_nearest(self, query, target):
        Q = query.unsqueeze(2).repeat(1, 1, target.size(0), 1)
        T = target.unsqueeze(0).repeat(query.size(1), 1, 1).unsqueeze(0).repeat(query.size(0), 1, 1, 1)
        index = (Q - T).pow(2).sum(3).sqrt().min(2)[1]
        return target[index]
    # def find_nearest(self, query, target):
    #     Q = query.unsqueeze(1).repeat(1, target.weight.size(0), 1)
    #     T = target.weight.unsqueeze(0).repeat(query.size(0), 1, 1)
    #     index = (Q - T).pow(2).sum(2).sqrt().min(1)[1]
    #     return target(index)
    
    # def find_nearest(self, query, target):
    #     Q = query.unsqueeze(1).repeat(1, target.size(0), 1)
    #     T = target.unsqueeze(0).repeat(query.size(0), 1, 1)
    #     index = (Q - T).pow(2).sum(2).sqrt().min(1)[1]
    #     return target[index]
    
    def forward(self, X):
        # Z_enc = self.encode(X)
        # Z_emb = self.find_nearest(Z_enc, self.embd.weight)
        Z_enc = X
        Z_emb = self.find_nearest(Z_enc, self.embd)
        # Z_emb.register_hook(self.hook)

        Z_recon = self.decode(Z_emb)
        return Z_recon, Z_enc, Z_emb
    # def forward(self, X):
    #     Z_enc = self.encode(X)
    #     Z_dec = self.find_nearest(Z_enc, self.embd.weight)
    #     Z_dec.register_hook(self.hook)

    #     X_recon = self.decode(Z_dec)
    #     Z_enc_for_embd = self.find_nearest(self.embd.weight, Z_enc)
    #     return X_recon, Z_enc, Z_dec , Z_enc_for_embd

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad

class VQVAE2(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_dim, z_dim):
        super(VQVAE2, self).__init__()
        self.z_dim = z_dim
        self.k_dim = k_dim

        # Encoder MLP
        self.encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

        # Embedding Book
        self.embd = nn.Embedding(self.k_dim, self.z_dim).cuda()

        # Decoder MLP
        self.decode = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def find_nearest(self, query, target):
        Q = query.unsqueeze(2).repeat(1, 1, target.size(0), 1)
        T = target.unsqueeze(0).repeat(query.size(1), 1, 1).unsqueeze(0).repeat(query.size(0), 1, 1, 1)
        index = (Q - T).pow(2).sum(3).sqrt().min(2)[1]
        return target[index]
    # def find_nearest(self, query, target):
    #     Q = query.unsqueeze(1).repeat(1, target.weight.size(0), 1)
    #     T = target.weight.unsqueeze(0).repeat(query.size(0), 1, 1)
    #     index = (Q - T).pow(2).sum(2).sqrt().min(1)[1]
    #     return target(index)
    
    # def find_nearest(self, query, target):
    #     Q = query.unsqueeze(1).repeat(1, target.size(0), 1)
    #     T = target.unsqueeze(0).repeat(query.size(0), 1, 1)
    #     index = (Q - T).pow(2).sum(2).sqrt().min(1)[1]
    #     return target[index]
    
    def forward(self, X):
        Z_enc = self.encode(X)
        Z_emb = self.find_nearest(Z_enc, self.embd.weight)
        # Z_emb.register_hook(self.hook)

        Z_recon = self.decode(Z_emb)
        return Z_recon, Z_enc, Z_emb
    # def forward(self, X):
    #     Z_enc = self.encode(X)
    #     Z_dec = self.find_nearest(Z_enc, self.embd.weight)
    #     Z_dec.register_hook(self.hook)

    #     X_recon = self.decode(Z_dec)
    #     Z_enc_for_embd = self.find_nearest(self.embd.weight, Z_enc)
    #     return X_recon, Z_enc, Z_dec , Z_enc_for_embd

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad

class HyperNetwork_FC(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x
    
class HyperNetwork_FC_VAE(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_VAE, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self.vae = VAE(
            input_dim=model_conf.id_dimension,
            hidden_dim=64,
            latent_dim=32
        )
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        z, _, __ = self.vae(z)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x

class HyperNetwork_FC_VQVAE(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_VQVAE, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self.vqvae = VQVAE2(
            input_dim=model_conf.id_dimension,
            hidden_dim=64,
            k_dim=450, # equal to category_dimension
            z_dim=32,
        )
        # Criterions
        self.MSE_Loss = nn.MSELoss().cuda()
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        z_recon, z_enc, z_emb = self.vqvae(z)
        z_and_emb_loss = self.MSE_Loss(z, z_emb.detach())
        recon_loss = self.MSE_Loss(z_recon, z)
        z_and_sg_embd_loss = self.MSE_Loss(z_enc, z_emb.detach())
        # sg_z_and_embd_loss = self.MSE_Loss(self.vqvae._modules['embd'].weight, Z_enc_for_embd.detach())
        # print("=" * 50, "in HyperNetwork_FC_VQVAE", "=" * 50)
        # print("z_and_emb_loss = ", z_and_emb_loss)
        # print("recon_loss = ", recon_loss)
        # print("z_and_sg_embd_loss = ", z_and_sg_embd_loss)
        total_loss = recon_loss + 0.25 * z_and_sg_embd_loss + 0.25 * z_and_emb_loss

        z = z_recon

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)

        # z_recon, z_enc, z_emb, z_enc_for_embd = self.vqvae(z)
        # recon_loss = self.MSE_Loss(z_recon, z)
        # z_and_sg_embd_loss = self.MSE_Loss(z_enc, z_emb.detach())
        # sg_z_and_embd_loss = self.MSE_Loss(self.vqvae._modules['embd'].weight, z_enc_for_embd.detach())
        # total_loss = recon_loss + sg_z_and_embd_loss + 0.25 * z_and_sg_embd_loss

        # z = z_recon
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x, total_loss

class HyperNetwork_FC_Fusion(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_Fusion, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.w3 = []
        self.w4 = []
        self.b3 = []
        self.b4 = []
        self.output_size = []
        self._item_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._category_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        
        self._item_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._category_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._item_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._item_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_hh_l0)
        
        initializer.default_weight_init(self._category_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._category_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            
            self.w3.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b3.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w4.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b4.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x1, z1, x2, z2, norm=0.01, sample_num=32, trigger_seq_length=30):
        units = self.units

        item_user_state, _ = self._item_gru_cell(z1)
        item_user_state = item_user_state[range(item_user_state.shape[0]), trigger_seq_length, :]
        z1 = self._item_mlp_trans(item_user_state)
        
        category_user_state, _ = self._category_gru_cell(z2)
        category_user_state = category_user_state[range(category_user_state.shape[0]), trigger_seq_length, :]
        z2 = self._category_mlp_trans(category_user_state)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            item_weight = torch.matmul(z1, self.w1[index]) + self.b1[index]
            category_weight = torch.matmul(z2, self.w3[index]) + self.b3[index]

            output_size = units[i]

            if not self.batch:
                item_weight = item_weight.view(input_size, output_size)
                category_weight = category_weight.view(input_size, output_size)
            else:
                item_weight = item_weight.view(sample_num, input_size, output_size)
                category_weight = category_weight.view(sample_num, input_size, output_size)
            item_bias = torch.matmul(z1, self.w2[index]) + self.b2[index]
            category_bias = torch.matmul(z2, self.w4[index]) + self.b4[index]

            weight = torch.clip(item_weight, min=-norm, max=norm) + category_weight
            bias = torch.clip(item_bias, min=-norm, max=norm) + category_bias
            # weight, bias = category_weight, category_bias

            x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1) + bias
            x1 = self.modules[index](x1) if index < len(self.modules) else x1

        return x1

class HyperNetwork_FC_Fusion_VAE(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_Fusion_VAE, self).__init__()
        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.w3 = []
        self.w4 = []
        self.b3 = []
        self.b4 = []
        self.output_size = []
        self.vae = VAE(
            input_dim=model_conf.id_dimension,
            hidden_dim=32,
            latent_dim=32
        )
        self._item_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._category_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._item_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._category_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._item_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._item_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_hh_l0)
        
        initializer.default_weight_init(self._category_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._category_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            
            self.w3.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b3.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w4.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b4.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x1, z1, x2, z2, norm=0.01, sample_num=32, trigger_seq_length=30):
        units = self.units

        item_user_state, _ = self._item_gru_cell(z1)
        item_user_state = item_user_state[range(item_user_state.shape[0]), trigger_seq_length, :]
        z1 = self._item_mlp_trans(item_user_state)
        
        category_user_state, _ = self._category_gru_cell(z2)
        category_user_state = category_user_state[range(category_user_state.shape[0]), trigger_seq_length, :]
        z2 = self._category_mlp_trans(category_user_state)
        z2, _, __ = self.vae(z2)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            item_weight = torch.matmul(z1, self.w1[index]) + self.b1[index]
            category_weight = torch.matmul(z2, self.w3[index]) + self.b3[index]

            output_size = units[i]

            if not self.batch:
                item_weight = item_weight.view(input_size, output_size)
                category_weight = category_weight.view(input_size, output_size)
            else:
                item_weight = item_weight.view(sample_num, input_size, output_size)
                category_weight = category_weight.view(sample_num, input_size, output_size)
            item_bias = torch.matmul(z1, self.w2[index]) + self.b2[index]
            category_bias = torch.matmul(z2, self.w4[index]) + self.b4[index]

            weight = torch.clip(item_weight, min=-norm, max=norm) + category_weight
            bias = torch.clip(item_bias, min=-norm, max=norm) + category_bias

            x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1) + bias
            x1 = self.modules[index](x1) if index < len(self.modules) else x1

        return x1
    
class HyperNetwork_FC_Fusion_VQVAE(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, pretrain_category_weight=None):
        super(HyperNetwork_FC_Fusion_VQVAE, self).__init__()
        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.w3 = []
        self.w4 = []
        self.b3 = []
        self.b4 = []
        self.output_size = []
        self.MSE_Loss = nn.MSELoss().cuda()
        self.vqvae = VQVAE(
            input_dim=model_conf.id_dimension,
            hidden_dim=64,
            k_dim=model_conf.category_id_vocab,
            z_dim=32,
            pretrain_category_weight=pretrain_category_weight
        )
        self._item_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._category_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._item_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._category_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._item_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._item_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_hh_l0)
        
        initializer.default_weight_init(self._category_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._category_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            
            self.w3.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b3.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w4.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b4.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x1, z1, x2, z2, norm=0.01, sample_num=32, trigger_seq_length=30):
        units = self.units

        z_recon, z_enc, z_emb = self.vqvae(z1)
        # z1_and_emb_loss = self.MSE_Loss(z1, z_emb.detach())
        # recon_loss = self.MSE_Loss(z_recon, z1)
        # z_and_sg_embd_loss = self.MSE_Loss(z_enc, z_emb.detach())

        # z2_and_emb_loss = self.MSE_Loss(z2, z_emb.detach())
        # recon_loss = self.MSE_Loss(z_recon, z2)
        # z2_and_sg_embd_loss = self.MSE_Loss(z_enc, z_emb.detach())

        # z2_recon_loss = self.MSE_Loss(z2, z_recon)
        #z2_and_zenc_loss = self.MSE_Loss(z2, z_enc)
        z2_and_emb_loss = self.MSE_Loss(z2, z_emb.detach())
        
        # total_loss = recon_loss + 0.5 * z_and_sg_embd_loss + 0.5 * z1_and_emb_loss # 0.25 * z_and_sg_embd_loss + 0.25 * z_and_emb_loss
        total_loss = z2_and_emb_loss # z2_and_zenc_loss + z2_and_emb_loss
        z1 = z_recon

        item_user_state, _ = self._item_gru_cell(z1)
        item_user_state = item_user_state[range(item_user_state.shape[0]), trigger_seq_length, :]
        z1 = self._item_mlp_trans(item_user_state)

        # z_recon, z_enc, z_emb, z_enc_for_embd = self.vqvae(z1)
        # recon_loss = self.MSE_Loss(z_recon, z1)
        # z_and_sg_embd_loss = self.MSE_Loss(z_enc, z_emb.detach())
        # sg_z_and_embd_loss = self.MSE_Loss(self.vqvae._modules['embd'].weight, z_enc_for_embd.detach())
        # total_loss = recon_loss + sg_z_and_embd_loss + 0.25 * z_and_sg_embd_loss
        # z1 = z_recon
        
        category_user_state, _ = self._category_gru_cell(z2)
        category_user_state = category_user_state[range(category_user_state.shape[0]), trigger_seq_length, :]
        z2 = self._category_mlp_trans(category_user_state)
        # z2, _, _, _ = self.vqvae(z2)

        # total_loss = total_loss + self.MSE_Loss(z2, z_emb.detach())

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            item_weight = torch.matmul(z1, self.w1[index]) + self.b1[index]
            category_weight = torch.matmul(z2, self.w3[index]) + self.b3[index]

            output_size = units[i]

            if not self.batch:
                item_weight = item_weight.view(input_size, output_size)
                category_weight = category_weight.view(input_size, output_size)
            else:
                item_weight = item_weight.view(sample_num, input_size, output_size)
                category_weight = category_weight.view(sample_num, input_size, output_size)
            item_bias = torch.matmul(z1, self.w2[index]) + self.b2[index]
            category_bias = torch.matmul(z2, self.w4[index]) + self.b4[index]

            weight = torch.clip(item_weight, min=-norm, max=norm) + category_weight
            bias = torch.clip(item_bias, min=-norm, max=norm) + category_bias

            x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1) + bias
            x1 = self.modules[index](x1) if index < len(self.modules) else x1

        return x1, total_loss
    
class HyperNetwork_FC_apg(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, N=64, M=32, K=16):
        super(HyperNetwork_FC_apg, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand

        modules_in = []
        modules_out = []
        modules = []
        # self.net = torch.nn.Sequential(*modules)
        # print("*" * 50)
        # print(units)
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]
            # print("=" * 50)
            linear_in = torch.nn.Linear(input_size, K, bias=True)
            initializer.default_weight_init(linear_in.weight)
            initializer.default_bias_init(linear_in.bias)
            modules_in.append(linear_in)
            # print(linear_in.weight.size())

            linear_out = torch.nn.Linear(K, output_size, bias=True)
            initializer.default_weight_init(linear_out.weight)
            initializer.default_bias_init(linear_out.bias)
            modules_out.append(linear_out)
            # print(linear_out.weight.size())
            # if activation_fns[i - 1] is not None:
            #     modules.append(activation_fns[i - 1]())

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, in_dimension * K * K).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(in_dimension * K * K).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())
            else:
                modules.append(None)

        self.K = K
        # self.modules_in = modules_in
        self.modules_in = torch.nn.Sequential(*modules_in)
        # self.modules_out = modules_out
        self.modules_out = torch.nn.Sequential(*modules_out)
        self.modules = modules

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                # weight = weight.view(input_size, output_size)
                weight = weight.view(self.K, self.K)
            else:
                # weight = weight.view(sample_num, input_size, output_size)
                weight = weight.view(sample_num, self.K, self.K)
            # bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            # print(x.device)
            # print(self.modules_in[index].device)
            # print("-" * 50)
            # print(x.size())
            # print(self.modules_in[index].weight.size())
            x = self.modules_in[index](x)
            # print(x.size())
            # x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1)
            # print(x.size())
            x = self.modules_out[index](x)
            # print(x.size())

            # if self.modules[index] is not None:
            #     x = self.modules[index](x)

        return x

class HyperNetwork_FC_APG_Fusion(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, N=64, M=32, K=16):
        super(HyperNetwork_FC_APG_Fusion, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._item_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._category_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._item_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._category_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._item_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._item_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_hh_l0)

        initializer.default_weight_init(self._category_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._category_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_hh_l0)
        self.expand = expand

        modules_in = []
        modules_out = []
        modules = []
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]
            # print("=" * 50)
            linear_in = torch.nn.Linear(input_size, K, bias=True)
            initializer.default_weight_init(linear_in.weight)
            initializer.default_bias_init(linear_in.bias)
            modules_in.append(linear_in)
            # print(linear_in.weight.size())

            linear_out = torch.nn.Linear(K, output_size, bias=True)
            initializer.default_weight_init(linear_out.weight)
            initializer.default_bias_init(linear_out.bias)
            modules_out.append(linear_out)

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())
            else:
                modules.append(None)

        self.K = K
        # self.modules_in = modules_in
        self.modules_in = torch.nn.Sequential(*modules_in)
        # self.modules_out = modules_out
        self.modules_out = torch.nn.Sequential(*modules_out)
        self.modules = modules

    def forward(self, x1, z1, x2, z2, norm=0.01, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z1 = z1.view(sample_num, -1)

        user_state, _ = self._item_gru_cell(z1)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z1 = self._item_mlp_trans(user_state)

        user_state, _ = self._category_gru_cell(z2)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z2 = self._category_mlp_trans(user_state)

        # print(z1.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            item_weight = torch.matmul(z1, self.w1[index]) + self.b1[index]
            category_weight = torch.matmul(z2, self.w2[index]) + self.b2[index]

            output_size = units[i]
            if not self.batch:
                # weight = weight.view(input_size, output_size)
                item_weight = item_weight.view(self.K, self.K)
                category_weight = category_weight.view(self.K, self.K)
            else:
                # weight = weight.view(sample_num, input_size, output_size)
                item_weight = item_weight.view(sample_num, self.K, self.K)
                category_weight = category_weight.view(sample_num, self.K, self.K)
            # bias = torch.matmul(z1, self.w2[index]) + self.b2[index]
            # print(x1.device)
            # print(self.modules_in[index].device)
            # print("-" * 50)
            # print(x1.size())
            # print(self.modules_in[index].weight.size())
            weight = torch.clip(item_weight, min=-norm, max=norm) + category_weight    
            
            x1 = self.modules_in[index](x1)
            # print(x1.size())
            # x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1) + bias
            x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1)
            # print(x1.size())
            x1 = self.modules_out[index](x1)
            # print(x1.size())

            # if self.modules[index] is not None:
            #     x1 = self.modules[index](x1)

        return x1
    
class HyperNetwork_FC_APG_Fusion_VQVAE(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, N=64, M=32, K=16, pretrain_category_weight=None):
        super(HyperNetwork_FC_APG_Fusion_VQVAE, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self.MSE_Loss = nn.MSELoss().cuda()
        self.vqvae = VQVAE(
            input_dim=model_conf.id_dimension,
            hidden_dim=64,
            k_dim=model_conf.category_id_vocab,
            z_dim=32,
            pretrain_category_weight=pretrain_category_weight
        )
        self._item_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._category_gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._item_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._category_mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._item_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._item_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._item_gru_cell.bias_hh_l0)

        initializer.default_weight_init(self._category_gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._category_gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._category_gru_cell.bias_hh_l0)
        self.expand = expand

        modules_in = []
        modules_out = []
        modules = []
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]
            # print("=" * 50)
            linear_in = torch.nn.Linear(input_size, K, bias=True)
            initializer.default_weight_init(linear_in.weight)
            initializer.default_bias_init(linear_in.bias)
            modules_in.append(linear_in)
            # print(linear_in.weight.size())

            linear_out = torch.nn.Linear(K, output_size, bias=True)
            initializer.default_weight_init(linear_out.weight)
            initializer.default_bias_init(linear_out.bias)
            modules_out.append(linear_out)

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())
            else:
                modules.append(None)

        self.K = K
        # self.modules_in = modules_in
        self.modules_in = torch.nn.Sequential(*modules_in)
        # self.modules_out = modules_out
        self.modules_out = torch.nn.Sequential(*modules_out)
        self.modules = modules

    def forward(self, x1, z1, x2, z2, norm=0.01, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z1 = z1.view(sample_num, -1)
        z_recon, z_enc, z_emb = self.vqvae(z1)
        z2_and_emb_loss = self.MSE_Loss(z2, z_emb.detach())
        total_loss = z2_and_emb_loss
        z1 = z_recon

        user_state, _ = self._item_gru_cell(z1)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z1 = self._item_mlp_trans(user_state)

        user_state, _ = self._category_gru_cell(z2)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z2 = self._category_mlp_trans(user_state)

        # print(z1.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            item_weight = torch.matmul(z1, self.w1[index]) + self.b1[index]
            category_weight = torch.matmul(z2, self.w2[index]) + self.b2[index]

            output_size = units[i]
            if not self.batch:
                # weight = weight.view(input_size, output_size)
                item_weight = item_weight.view(self.K, self.K)
                category_weight = category_weight.view(self.K, self.K)
            else:
                # weight = weight.view(sample_num, input_size, output_size)
                item_weight = item_weight.view(sample_num, self.K, self.K)
                category_weight = category_weight.view(sample_num, self.K, self.K)
            # bias = torch.matmul(z1, self.w2[index]) + self.b2[index]
            # print(x1.device)
            # print(self.modules_in[index].device)
            # print("-" * 50)
            # print(x1.size())
            # print(self.modules_in[index].weight.size())
            weight = torch.clip(item_weight, min=-norm, max=norm) + category_weight    
            
            x1 = self.modules_in[index](x1)
            # print(x1.size())
            # x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1) + bias
            x1 = torch.bmm(x1.unsqueeze(1), weight).squeeze(1)
            # print(x1.size())
            x1 = self.modules_out[index](x1)
            # print(x1.size())

            # if self.modules[index] is not None:
            #     x1 = self.modules[index](x1)

        return x1, total_loss