from os import path
from os.path import join as pj
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as Functional
import functions.models as F
import modules.utils as utils
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.wishart import Wishart
from random import uniform
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import betaln, digamma, gammaln
from scipy import linalg, sparse
from scipy.special import logsumexp

class Net_DP(nn.Module):
    def __init__(self, o):
        super(Net_DP, self).__init__()
        self.o = o
        self.scdp = SCDP(o)
        self.loss_calculator_dp = LossCalculator_DP(o)
        self.beta = None 

    def forward(self, inputs):
        x_r_pre, z = self.scdp(inputs)      
        loss = self.loss_calculator_dp(inputs, x_r_pre,z)
        return loss


class SCDP(nn.Module):
    def __init__(self, o):
        super(SCDP, self).__init__()
        self.o = o
        self.sampling = False
        self.predict_label = None
        x_encs_y, x_encs_z = {},{}
        x_mid_enc = MLP(o.dims_enc_x, norm=o.norm, drop=o.drop, out_trans='mish')
        x_y_enc = MLP(o.dims_enc_x+[o.dim_z], hid_norm=o.norm, hid_drop=o.drop)
        for m in o.mods:
            x_indiv_ency = MLP([o.dims_h[m], o.dims_enc_x[0]], out_trans='mish', norm=o.norm,
                              drop=o.drop)
            x_encs_y[m] = nn.Sequential(x_indiv_ency, x_y_enc)
        self.x_enc = nn.ModuleDict(x_encs_y)
            # self.x_enc = MLP([o.dims_h[m]]+o.dims_enc_x+[o.dim_z], norm=o.norm, out_trans='mish',
            #                 hid_drop=o.drop)        
        self.x_dec = MLP([o.dim_z]+o.dims_dec_x+[sum(o.dims_h.values())], hid_norm=o.norm,
                         hid_drop=o.drop)




    def forward(self, inputs):
        o = self.o
        x = inputs["x"]
        e = inputs["e"]

        # Encode x_m
        z_x_mu, z_x_logvar = {}, {}        
        x_pp = {}
        for m in x.keys():
            x_pp[m] = preprocess(x[m], m, o.dims_x[m], o.task) #如果是label：one hot,如果是RNA等, 就要平滑处理
            
            if m in ["rna", "adt"]:  # use mask 
                h = x_pp[m] * e[m] #两个rna数据集，只考虑其中相同的
            else:
                h = x_pp[m]
            # encoding

                # encode for z
            z = self.x_enc[m](h)
        # 
        # Generate x_m activation/probability
        x_r_pre = self.x_dec(z).split(list(o.dims_h.values()), dim=1) ###
        x_r_pre = utils.get_dict(o.mods, x_r_pre)
           
        return x_r_pre, z


class LossCalculator_DP(nn.Module):

    def __init__(self, o):
        super(LossCalculator_DP, self).__init__()
        self.o = o
        self.pois_loss = nn.PoissonNLLLoss(full=True, reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')  # log_softmax + nll
        self.mse_loss = nn.MSELoss(reduction='none')
        self.gauss_loss = nn.GaussianNLLLoss(full=True, reduction='none')
        self.mean_dp = None
        self.weight_concentration_dp = None
        self.precisions_cholesky_dp = None
        self.degrees_of_freedom_dp = None
        self.mean_precision_dp = None

    def forward(self, inputs, x_r_pre, z):
        o = self.o
        x = inputs["x"]
        e = inputs["e"]

        loss_recon = self.calc_recon_loss(x, e, x_r_pre)

        if o.epoch_id > 300:
            loss_dp = self.calcudp_loss(z)
        else:
            loss_dp = 0
        # loss = loss_recon + loss_dp


        if o.debug == 1:
            if o.epoch_id > 300:
                print("recon: %.3f\tlossdp: %.3f" % (loss_recon.item(), loss_dp.item()))
            else:
                print("recon: %.3f" % (loss_recon.item()))
        return loss_recon + loss_dp

    def calc_recon_loss(self, x, e, x_r_pre):
        losses = {}
        # Reconstruciton losses of x^m
        for m in x.keys():
            losses[m] = (self.pois_loss(x_r_pre[m], x[m]) * e[m]).sum()
        return sum(losses.values()) / x[m].size(0)


    def calcudp_loss(self,z):
        mean_dp = self.mean_dp.cuda()
        n_features = z.size(1)
        phi = 3.141592653589793
        weight_concentration_dp = self.weight_concentration_dp.cuda()
        precisions_cholesky_dp = self.precisions_cholesky_dp.cuda()
        degrees_of_freedom_dp = self.degrees_of_freedom_dp.cuda()
        mean_precision_dp = self.mean_precision_dp.cuda()
        digamma_sum_dp = th.special.digamma(weight_concentration_dp[0]+weight_concentration_dp[1])
        digamma_a_dp = th.special.digamma(weight_concentration_dp[0])
        digamma_b_dp = th.special.digamma(weight_concentration_dp[1])
        log_weights_dp_b = th.cat((th.zeros(1).cuda(), th.cumsum((digamma_b_dp - digamma_sum_dp), dim = 0)[:-1]), dim = 0)
        log_weights_dp = digamma_a_dp - digamma_sum_dp + log_weights_dp_b
        log_det_dp = th.sum(precisions_cholesky_dp.log(), dim = 1)
        precisions_dp = precisions_cholesky_dp**2
        log_prob_dp = th.sum((mean_dp**2 * precisions_dp), dim = 1) - 2.0 * th.mm(z, (mean_dp * precisions_dp).T) + th.mm(z**2, precisions_dp.T)
        log_gauss_pre_dp = -0.5 * (n_features * math.log(phi * 2) + log_prob_dp) + log_det_dp 
        log_gauss_dp = log_gauss_pre_dp - 0.5 * n_features * (degrees_of_freedom_dp).log()        
        log_lambda_dp = n_features * math.log(2.0) + th.sum(
            th.special.digamma( 0.5 * (degrees_of_freedom_dp - th.arange(0, n_features).unsqueeze(1).cuda())), 0,)
        log_prob_z_dp = log_gauss_dp + 0.5 * (log_lambda_dp - n_features / mean_precision_dp)
        loss_loglikeli = th.logsumexp(log_prob_z_dp + log_weights_dp, dim=1).mean()  
        output = (th.sum((log_prob_z_dp + log_weights_dp).exp(), dim=1)).mean()
        return -loss_loglikeli


class MLP(nn.Module):
    def __init__(self, features=[], hid_trans='mish', out_trans=False,
                 norm=False, hid_norm=False, drop=False, hid_drop=False):
        super(MLP, self).__init__()
        layer_num = len(features)
        assert layer_num > 1, "MLP should have at least 2 layers!"
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False
        
        layers = []
        for i in range(1, layer_num):
            layers.append(nn.Linear(features[i-1], features[i]))
            if i < layer_num - 1:  # hidden layers (if layer number > 2)
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:                  # output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class Layer1D(nn.Module):
    def __init__(self, dim=False, norm=False, trans=False, drop=False):
        super(Layer1D, self).__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if trans:
            layers.append(func(trans))
        if drop:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def preprocess(x, name, dim, task):
    if name == "label":
        x = nn.functional.one_hot(x.squeeze(1), num_classes=dim).float()
    # elif name == "atac":
    #     x = x.log1p()
    elif name == "rna":
        x = x.log1p()
        # x = x
    elif name == "adt":
        x = x.log1p()
    return x


def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0)  # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)


def exp(x, eps=1e-12):
    return (x < 0) * (x.clamp(max=0)).exp() + (x >= 0) / ((-x.clamp(min=0)).exp() + eps)


def log(x, eps=1e-12):
    return (x + eps).log()


def func(func_name):
    if func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'silu':
        return nn.SiLU()
    elif func_name == 'mish':
        return nn.Mish()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    elif func_name == 'log_softmax':
        return nn.LogSoftmax(dim=1)
    else:
        assert False, "Invalid func_name."


class CheckBP(nn.Module):
    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):
    def forward(self, input):
        return F.Identity.apply(input)
