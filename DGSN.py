import numpy as np
import torch
import torch.nn as nn
from typing import List
from soft_dtw import SoftDTW



class shapeVae(nn.Module):
    def __init__(
            self,
            input_size,
            zd_hus=[128, 128],
            zc_hus=[128, 128],
            zd_dim=16,
            zc_dim=16,
            x_hus=[128, 128],
            zs_hus=[128, 128],
            zs_dim = 64,
            time_neighbour = 25,
            idx_hus=[128,128],
            shape_num = 20,
            seg_num = 20,
            label_size = 1,
            device = None
    ):
        super().__init__()
        self.model = "shapeVae"
        self.device = device
        self.time_neighbour = time_neighbour
        self.pz_d = [0.0, torch.log(torch.tensor(1.0 ** 2)).type(torch.float64)]  # p(zd)
        self.pz_s = [0.0, torch.log(torch.tensor(0.5 ** 2)).type(torch.float64)]  # p(zs)
        self.zd_hus = zd_hus
        self.zc_hus = zc_hus
        self.zs_hus = zs_hus
        self.zd_dim = zd_dim
        self.zc_dim = zc_dim
        self.zs_dim = zs_dim
        self.zs_init = None
        self.x_hus = x_hus
        self.zd_pre_encoder = LatentZdPreEncoder(input_size, self.zd_hus)
        self.zc_pre_encoder = LatentZcPreEncoder(input_size + self.zd_dim, self.zc_hus)
        self.zs_pre_encoder = LatentZsPreEncoder(self.zs_dim, self.zs_hus)


        self.zd_gauss_layer = GaussianLayer(self.zd_hus[1], self.zd_dim)
        self.zc_gauss_layer = GaussianLayer(self.zc_hus[1], self.zc_dim)
        self.zs_gauss_layer = GaussianLayer(self.zs_hus[1], self.zs_dim)
        self.class_model = classModel(shape_num,128,label_size)
        self.pre_decoder = PreDecoder(self.zd_dim + self.zc_dim, self.x_hus)
        self.zc_decoder = ZcPreDecoder(self.zd_dim + self.zs_dim, self.zc_hus)
        self.zc_dec_gauss_layer = GaussianLayer(self.zc_hus[1], zc_dim)
        self.dec_gauss_layer = GaussianLayer(self.x_hus[1], input_size)
        self.sdtw_loss = SoftDTW(use_cuda=True, gamma=0.1)
        self.shape_loss = SoftDTW(use_cuda=True, gamma=0.1)
        self.shape_decoder= shapeDecoder(seg_num,shape_num,input_size)
        self.idx_decoder = IdxDecoder(self.zd_hus[1], idx_hus)
        self.loss = nn.CrossEntropyLoss()

    def forward(
            self, x: torch.Tensor, mu_idx: torch.Tensor, num_seqs: int, num_segs: int
    ):
        self.zs_init = torch.empty([num_seqs, 1, self.zs_dim]).normal_(mean=0, std=0.5).to(self.device)
        self.zs_init.requires_grad = True
        zs = self.zs_init[mu_idx,:,:]
        zd_pre_out = self.zd_pre_encoder(x)
        zd_mu, zd_logvar, zd_sample = self.zd_gauss_layer(zd_pre_out)
        zd_mu = zd_mu.view(x.shape[0], x.shape[1], -1)
        zd_logvar = zd_logvar.view(x.shape[0], x.shape[1], -1)
        zd_sample = zd_sample.view(x.shape[0], x.shape[1], -1)
        qzd_x = [zd_mu, zd_logvar]
        zc_pre_out = self.zc_pre_encoder(x,zd_sample)
        zc_mu, zc_logvar, zc_sample = self.zc_gauss_layer(zc_pre_out)
        zc_mu = zc_mu.view(x.shape[0], x.shape[1], -1)
        zc_logvar = zc_logvar.view(x.shape[0], x.shape[1], -1)
        zc_sample = zc_sample.view(x.shape[0], x.shape[1], -1)
        qzc_x = [zc_mu, zc_logvar]
        zs_pre_out = self.zs_pre_encoder(zs)
        zs_mu, zs_logvar, zs_sample = self.zs_gauss_layer(zs_pre_out)
        qz_s = [zs_mu, zs_logvar]
        zc_dec = self.zc_decoder(zs_sample,zd_sample)
        zc_ds_mu, zc_ds_logvar, zc_ds_sample = self.zc_dec_gauss_layer(zc_dec)
        pz_c = [zc_ds_mu,zc_ds_logvar]

        x_pre_out = self.pre_decoder(zd_sample, zc_sample)
        x_mu, x_logvar, x_sample = self.dec_gauss_layer(x_pre_out)
        x_mu = x_mu.view(x.shape[0], x.shape[1],-1)
        x_logvar = x_logvar.view(x.shape[0], x.shape[1],-1)
        x_sample = x_sample.view(x.shape[0], x.shape[1],-1)
        px_z = [x_mu, x_logvar]
        shapelet = self.shape_decoder(zd_sample,zc_sample)
        shapelet_repeat = shapelet.repeat(1,1,num_segs,1)
        distance = torch.sqrt(torch.sum(torch.square(shapelet_repeat-x.unsqueeze(1)),dim=3))
        distance,distance_index = torch.min(distance,dim=2)
        label_out = self.class_model(distance)
        idx_out = self.idx_decoder(zd_pre_out)
        neg_kld_zs = -1 * torch.sum(
            self.kld(qz_s[0], qz_s[1], self.pz_s[0], self.pz_s[1]), dim=1
        )
        neg_kld_zs = torch.mean(neg_kld_zs)
        neg_kld_zc = -1 * torch.sum(
            self.kld(qzc_x[0], qzc_x[1], pz_c[0], pz_c[1]), dim=(1, 2)
        )
        neg_kld_zd = -1 * torch.sum(
            self.kld(qzd_x[0], qzd_x[1], self.pz_d[0], self.pz_d[1]), dim=(1, 2)
        )
        log_px_z = torch.sum(
            self.log_gauss(x, px_z[0].detach(), px_z[1]), dim=(1, 2)
        )
        lower_bound = log_px_z + neg_kld_zd + neg_kld_zc + neg_kld_zs
        pair_loss = self.pair_loss1(x, zc_sample)
        dtw_loss = self.sdtw_loss(x, x_mu)
        return lower_bound, log_px_z, neg_kld_zd, neg_kld_zc, neg_kld_zs, \
               pair_loss,dtw_loss,idx_out,x_sample,label_out,distance,shapelet,distance_index,zd_sample,zc_sample


    def dtw_loss(self,x,x_dec):
        b, t, w = x.shape
        x = x.view(b * t,w,1)
        x_dec = x_dec.view(b * t,w,1)
        loss = self.sdtw_loss(x,x_dec)
        return loss.mean()

    def pair_loss1(self,x,zc,delta=1):
        b, t, w = x.shape
        zc = zc.view(b, t, -1)
        b,t,h = zc.shape
        x_padding = torch.nn.functional.pad(x, pad=(0, 0, self.time_neighbour, self.time_neighbour, 0, 0),
                                            mode='constant', value=0)
        zc_padding = torch.nn.functional.pad(zc, pad=(0, 0, self.time_neighbour, self.time_neighbour, 0, 0),
                                             mode='constant', value=0)
        x_compare = torch.nn.functional.unfold(x_padding.unsqueeze(1),(self.time_neighbour*2+1,w))
        x_compare = x_compare.permute(0,2,1).view(b,t,self.time_neighbour*2+1,w)
        zc_compare = torch.nn.functional.unfold(zc_padding.unsqueeze(1), (self.time_neighbour * 2 + 1, h))
        zc_compare = zc_compare.permute(0, 2, 1).view(b, t, self.time_neighbour * 2 + 1, -1)
        GH = torch.sum(x.unsqueeze(2) * x_compare, dim=3)
        GG = torch.sum(x * x, dim=2)
        HH = torch.sum(x_compare * x_compare, dim=3)
        x_dist = GH/torch.pow(torch.mul(GG.unsqueeze(2),HH)+0.000001,0.5)
        zc_dist = torch.exp(-torch.norm(zc.unsqueeze(2)-zc_compare,dim=3))
        pair_loss = torch.square(delta * x_dist - zc_dist)
        pair_loss = torch.sum(pair_loss,dim=(1,2))/torch.square(torch.tensor(t))
        return pair_loss

    def log_gauss(self, x, mu=0.0, logvar=0.0):
        return -0.5 * (
                torch.log(2 * torch.tensor(np.pi)) + logvar + torch.pow(x - mu, 2) / torch.exp(logvar)
        )

    def kld(self, p_mu, p_logvar, q_mu, q_logvar):
        return -0.5 * (
                1
                + p_logvar
                - q_logvar
                - (torch.pow(p_mu - q_mu, 2) + torch.exp(p_logvar)) / torch.exp(q_logvar)
        )


class VariableLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class LatentZcPreEncoder(nn.Module):
    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            self.hus = [1024, 1024]
        else:
            self.hus = hus
        self.fc1 = VariableLinearLayer(input_size, self.hus[0])
        self.fc2 = VariableLinearLayer(self.hus[0], self.hus[1])

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor):
        out = torch.cat([x.view(x.shape[0] * x.shape[1], -1), lat_seq.view(x.shape[0] * x.shape[1], -1)], dim=-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class LatentZsPreEncoder(nn.Module):
    def __init__(self, input_size, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, x):
        out = x.view(x.shape[0] * x.shape[1],-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class LatentZdPreEncoder(nn.Module):

    def __init__(self, input_size, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, x):
        out = x.view(x.shape[0] * x.shape[1],-1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class GaussianLayer(nn.Module):

    def __init__(self, input_size: int, dim: int):
        super().__init__()
        self.mulayer = nn.Linear(input_size, dim)
        self.logvar_layer = nn.Linear(input_size, dim)

    def forward(self, input_layer: torch.Tensor):
        mu = self.mulayer(input_layer)
        logvar = self.logvar_layer(input_layer)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu, logvar, mu + eps * std

class classModel(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,dropout=0.25):
        super(classModel, self).__init__()
        # last rate
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size, bias=True)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

class IdxDecoder(nn.Module):
    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, lat_seg: torch.Tensor):
        out = self.fc1(lat_seg)
        out = self.fc2(out)
        return out

class ZcPreDecoder(nn.Module):

    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor):
        b,t,w = lat_seq.shape
        lat_seg = lat_seg.unsqueeze(1).repeat(1, t, 1)
        out = torch.cat([lat_seg, lat_seq], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class shapeDecoder(nn.Module):
    def __init__(self, seg_num,shape_size,input_size):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=shape_size, kernel_size=(seg_num-input_size+1, 64))

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor):
        out = torch.cat([lat_seg, lat_seq], -1).unsqueeze(1)
        out = self.conv2d(out)
        out = out.permute(0, 1, 3, 2)
        return out

class PreDecoder(nn.Module):
    def __init__(self, input_size: int, hus: List[int] = None):
        super().__init__()
        if hus is None:
            hus = [1024, 1024]
        self.fc1 = VariableLinearLayer(input_size, hus[0])
        self.fc2 = VariableLinearLayer(hus[0], hus[1])

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor):
        out = torch.cat([lat_seg, lat_seq], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
