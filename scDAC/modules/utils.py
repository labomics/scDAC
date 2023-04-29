import os
from os import path
from os.path import join as pj

import shutil
from pandas import Categorical
import torch as th
import cv2 as cv
import json
import toml
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import csv
import copy
import math
import itertools
from torch.distributions.beta import Beta
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from scipy.special import gammaln,digamma
from scipy.linalg import det, solve
# IO

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_toml(data, filename):
    with open(filename, 'w') as f:
        toml.dump(data, f)
        

def load_toml(filename):
    with open(filename, 'r') as f:
        data = toml.load(f)
    return data


def rmdir(directory):
    if path.exists(directory):
        print('Removing directory "%s"' % directory)
        shutil.rmtree(directory)


def mkdir(directory, remove_old=False):
    if remove_old:
        rmdir(directory)
    if not path.exists(directory):
        os.makedirs(directory)


def mkdirs(directories, remove_old=False):
    """
    Make directories recursively
    """
    t = type(directories)
    if t in [tuple, list]:
        for d in directories:
            mkdirs(d, remove_old=remove_old)
    elif t is dict:
        for d in directories.values():
            mkdirs(d, remove_old=remove_old)
    else:
        mkdir(directories, remove_old=remove_old)
        

def get_filenames(directory, extension):
    filenames = glob(pj(directory, "*."+extension))
    filenames = [path.basename(filename) for filename in filenames]
    filenames.sort()
    return filenames


def load_csv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def load_tsv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        data = list(reader)
    return data


def save_list_to_csv(data, filename, delimiter=','):
    """
    Save a 2D list `data` into a `.csv` file named as `filename`
    """
    with open(filename, "w") as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)
        

def save_tensor_to_csv(data, filename, delimiter=','):
    """
    Save a 2D tensor `data` into a `.csv` file named as `filename`
    """
    data_list = convert_tensor_to_list(data)
    save_list_to_csv(data_list, filename, delimiter)


def get_name_fmt(file_num):
    """
    Get the format of the filename with a minimum string lenth when each file \
    are named by its ID
    - `file_num`: the number of files to be named
    """
    return "%0" + str(math.floor(math.log10(file_num))+1) + "d"
    

def gen_data_dirs(base_dir, integ_dir):
    dirs = {
        "base":   pj("data", base_dir),
        "prepr":  pj("data", base_dir, "preprocessed"),
        "integ":  pj("data", base_dir, "preprocessed", integ_dir),
        "mat":    pj("data", base_dir, "preprocessed", integ_dir, "mat"),
        "vec":    pj("data", base_dir, "preprocessed", integ_dir, "vec"),
        "name":   pj("data", base_dir, "preprocessed", integ_dir, "name"),
        "fig":    pj("data", base_dir, "preprocessed", integ_dir, "fig"),
        "seurat": pj("data", base_dir, "preprocessed", "seurat")
    }
    return dirs


# Visualization

def imshow(img, height=None, width=None, name='img', delay=1):
    # img: H * W * D
    if th.is_tensor(img):
        img = img.cpu().numpy()
    h = img.shape[0] if height == None else height
    w = img.shape[1] if width == None else width
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.imshow(name, img)
    cv.waitKey(delay)


def imwrite(img, name='img'):
    # img: H * W * 
    img = (img * 255).byte().cpu().numpy()
    cv.imwrite(name + '.jpg', img)


def imresize(img, height, width):
    # img: H * W * D
    is_torch = False
    if th.is_tensor(img):
        img = img.cpu().numpy()
        is_torch = True
    img_resized = cv.resize(img, (width, height))
    if is_torch:
        img_resized = th.from_numpy(img_resized)
    return img_resized


def heatmap(img, cmap='hot'):
    cm = plt.get_cmap(cmap)
    cimg = cm(img.cpu().numpy())
    cimg = th.from_numpy(cimg[:, :, :3])
    cimg = th.index_select(cimg, 2, th.LongTensor([2, 1, 0])) # convert to BGR for opencv
    return cimg


def plot_figure(handles,
                legend = False,
                xlabel = False, ylabel = False,
                xlim = False, ylim = False,
                title = False,
                save_path = False,
                show = False
               ):
    
    print("Plotting figure: " + (title if title else "(no name)"))

    plt.subplots()

    if legend:
        plt.legend(handles = handles)
        
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()


# Data structure conversion

def copy_dict(src_dict):
    return copy.deepcopy(src_dict)
    
    
def get_num(s):
    # return int(path.splitext(s)[0])
    return int(''.join(filter(str.isdigit, s)))


def convert_tensor_to_list(data):
    """
    Covert a 2D tensor `data` to a 2D list
    """
    return [list(line) for line in list(data.cpu().detach().numpy())]


def get_dict(keys, values):
    """
    Construct a dictionary with a key list `keys` and a corresponding value list `values`
    """
    return dict(zip(keys, values))


def get_sub_dict(src_dict, keys):
    return {k: src_dict[k] for k in keys}


def convert_tensors_to_cuda(x):
    """
    Recursively converts tensors to cuda
    """
    y = {}
    for kw, arg in x.items():
        y[kw] = arg.cuda() if th.is_tensor(arg) else convert_tensors_to_cuda(arg)
    return y


def detatch_tensors(x):
    """
    Recursively detatch tensors
    """
    y = {}
    for kw, arg in x.items():
        y[kw] = arg.detach() if th.is_tensor(arg) else detatch_tensors(arg)
    return y


def list_intersection(x):
    """
    - `x`: lists
    """
    return list(set.intersection(*map(set, x)))


def flat_lists(x):
    """
    - `x`: a list of lists
    """
    return list(itertools.chain.from_iterable(x))


def transpose_list(x):
    """
    - `x`: a 2D list
    """
    # return list(zip(*lists))
    return list(map(list, zip(*x)))


def gen_all_batch_ids(s_joint, combs):
    
    s_joint = flat_lists(s_joint)
    combs = flat_lists(combs)
    s = []

    for subset, comb in enumerate(combs):
        s_subset = {}
        for m in comb+["joint"]:
            s_subset[m] = s_joint[subset]
        s.append(s_subset)

    dims_s = {}
    for m in list(np.unique(flat_lists(combs)))+["joint"]:
        s_m = []
        for s_subset in s:
            if m in s_subset.keys():
                s_m.append(s_subset[m])

        sorted, _ = th.tensor(np.unique(s_m)).sort()
        sorted = sorted.tolist()
        dims_s[m] = len(sorted)

        for s_subset in s:
            if m in s_subset.keys():
                s_subset[m] = sorted.index(s_subset[m])

    return s_joint, combs, s, dims_s


# Debugging

def get_nan_mask(x):
    mask = th.isinf(x) + th.isnan(x)
    is_nan = mask.sum() > 0
    return mask, is_nan


# Math computations

def sample_gaussian(mu, logvar):
    std = (0.5*logvar).exp()
    eps = th.randn_like(std)
    return mu + std*eps

# def sample_gaussian1(mu, logvar, c_mu):
#     std = (0.5*logvar).exp()
#     return mu + std*c_mu

def sample_beta(beta_a, beta_b):
    b = Beta(beta_a, beta_b)
    return b.sample()

def sample_categorical(y_c):
    b = Categorical(y_c)
    return b.sample()



def sample_uniform(a, b):
    c = Uniform(a, b)
    return c.sample()



def kl_beta(alpha, pre_a, pre_b):
    m = Beta(th.tensor([1.]), alpha)
    n = Beta(pre_a, pre_b)
    #m_sample = m.sample()
    #n_sample = n.sample()
    return kl_divergence(m, n)


def kl_categorical(y_c, y_d):
    m = Categorical(y_c)
    n = Categorical(y_d)
    #m_sample = m.sample()
    #n_sample = n.sample()
    return kl_divergence(m, n)


def kl_normal(mu, logvar):
    c = Normal(th.tensor([0.]), th.tensor([1.]))
    d = Normal(mu, logvar)
    c_sample = c.sample()
    d_sample = d.sample()
    return c_sample, d_sample, kl_divergence(c, d)




def extract_tria_values(x):
    """
    Extract, vectorize, and sort matrix values of the upper triangular part.
    Note it contains in-place operations.
    - `x`: the 2D input matrix of size N * N
    - `y`: the 1D output vector of size (N-1)*N/2
    """
    N = x.size(0)
    x_triu = x.triu_(diagonal=1)
    y, _ = x_triu.view(-1).sort(descending=True)
    y = y[:(N-1)*N//2]
    return y


def block_compute(func, block_size, *args):
    """
    - `args`: the args of function `func`
    """
    assert len(args) % 2 == 0, "The number of args must be even!"
    para_num = len(args)//2
    args = [arg.split(block_size, dim=0) for arg in args]
    I = len(args[0])
    J = len(args[para_num])
    z_rows = []
    for i in range(I):
        z_row = []
        for j in range(J):
            x = [args[k][i] for k in range(para_num)]
            y = [args[k+para_num][j] for k in range(para_num)]
            z = func(*(x+y))  # B * B
            z_row.append(z)
        z_row = th.cat(z_row, dim=1)  # B * JB
        z_rows.append(z_row)
    z_rows = th.cat(z_rows, dim=0)  # IB * JB
    return z_rows
    
    
def calc_squared_dist(x, y):
    """
    Squared Euclidian distances between two sets of variables
    - `x`: N1 * D
    - `y`: N2 * D
    """
    return th.cdist(x, y) ** 2
    

def calc_bhat_dist(mu1, logvar1, mu2, logvar2, mem_limit=1e9):
    """
    Bhattacharyya distances between two sets of Gaussian distributions
    - `mu1`, `logvar1`: N1 * D
    - `mu2`, `logvar2`: N2 * D
    - `mem_limit`: the maximal memory allowed for computaion
    """
 
    def calc_bhat_dist_(mu1, logvar1, mu2, logvar2):
        N1, N2 = mu1.size(0), mu2.size(0)
        mu1 = mu1.unsqueeze(1)          # N1 * 1 * D
        logvar1 = logvar1.unsqueeze(1)  # N1 * 1 * D
        mu2 = mu2.unsqueeze(0)          # 1 * N2 * D
        logvar2 = logvar2.unsqueeze(0)  # 1 * N2 * D
        
        var1 = logvar1.exp()  # N1 * 1 * D
        var2 = logvar2.exp()  # 1 * N2 * D
        var = (var1 + var2) / 2  # N1 * N2 * D
        inv_var = 1 / var  # N1 * N2 * D
        inv_covar = inv_var.diag_embed()  # N1 * N2 * D * D
        
        ln_det_covar = var.log().sum(-1)  # N1 * N2
        ln_sqrt_det_covar12 = 0.5*(logvar1.sum(-1) + logvar2.sum(-1))  # N1 * N2
        
        mu_diff = mu1 - mu2  # N1 * N2 * D
        mu_diff_h = mu_diff.unsqueeze(-2)  # N1 * N2 * 1 * D
        mu_diff_v = mu_diff.unsqueeze(-1)  # N1 * N2 * D * 1
        
        dist = 1./8 * mu_diff_h.matmul(inv_covar).matmul(mu_diff_v).reshape(N1, N2) +\
               1./2 * (ln_det_covar - ln_sqrt_det_covar12)  # N1 * N2
        return dist
 
    block_size = int(math.sqrt(mem_limit / (mu1.size(1) * mu2.size(1))))
    return block_compute(calc_bhat_dist_, block_size, mu1, logvar1, mu2, logvar2)


# Evaluation metrics

def calc_foscttm(mu1, logvar1, mu2, logvar2):
    """
    Fraction Of Samples Closer Than the True Match
    - `mu1`, `mu2`, `logvar1`, `logvar2`: N * D
    """
    N = mu1.size(0)
    # dists = calc_bhat_dist(mu1, logvar1, mu2, logvar2)  # N * N
    dists = th.cdist(mu1, mu2)  # N * N
    true_match_dists = dists.diagonal().unsqueeze(-1).expand_as(dists)  # N * N
    foscttms = dists.lt(true_match_dists).sum(-1).float() / (N - 1)  # N
    return foscttms.mean().item()
    
    # fracs = []
    # for n in range(N):
    #     dist = dists[n]  # N
    #     # if n == 0:
    #     #     for i in range(dist.size(0)//50):
    #     #         print(dist[i].item())
    #     true_match_dist = dist[n]
    #     fracs += [dist.lt(true_match_dist).sum().float().item() / (N - 1)]
    #     # if n == 0:
    #     #     exit()
    # foscttm = sum(fracs) / len(fracs)
    # return foscttm


def calc_subset_foscttm(model, data_loader):
    mods = data_loader.dataset.comb
    z_mus = get_dict(mods, [[] for _ in mods])
    z_logvars = get_dict(mods, [[] for _ in mods])
    with th.no_grad():
        for data in data_loader:
            data = convert_tensors_to_cuda(data)
            for m in data["x"].keys():
                input_data = {
                    "x": {m: data["x"][m]},
                    "s": data["s"], 
                    "e": {}
                }
                if m in data["e"].keys():
                    input_data["e"][m] = data["e"][m]
                _, _, z_mu, z_logvar, *_ = model(input_data)  # N * Z
                z_mus[m].append(z_mu)
                z_logvars[m].append(z_logvar)

    for m in mods:
        z_mus[m] = th.cat(z_mus[m], dim=0)  # SN * Z
        z_logvars[m] = th.cat(z_logvars[m], dim=0)  # SN * Z

    foscttm = {}
    for m in mods:
        for m_ in set(mods) - {m}:
            foscttm[m+"_to_"+m_] = calc_foscttm(z_mus[m], z_logvars[m], z_mus[m_], z_logvars[m_])
    return foscttm


def calc_subsets_foscttm(model, data_loaders, foscttm_list, split, epoch_num, epoch_id=0):
    model.eval()
    foscttm_sums, foscttm_nums = [], []
    for subset, data_loader in data_loaders.items():
        if len(data_loader) > 0 and len(data_loader.dataset.comb) > 1:
            foscttm = calc_subset_foscttm(model, data_loader)
            for k, v in foscttm.items():
                print('Epoch: %d/%d, subset: %d, split: %s, %s foscttm: %.4f' %
                (epoch_id+1, epoch_num, subset, split, k, v))
            foscttm_sums.append(sum(foscttm.values()))
            foscttm_nums.append(len(foscttm))
    if len(foscttm_sums) > 0:
        foscttm_avg = sum(foscttm_sums) / sum(foscttm_nums)
        print('Epoch: %d/%d, %s foscttm: %.4f\n' % (epoch_id+1, epoch_num, split, foscttm_avg))
        foscttm_list.append((float(epoch_id), float(foscttm_avg)))

_small_negative_number = -1.0e-10
def lnZ_Wishart(nu,V):
    """
    log normalization constant of Wishart distribution
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
      note <CovMat> = V/nu
    """
    if nu < len(V) + 1:
        raise "dof parameter nu must larger than len(V)"

    D = len(V)
    # lnZ = 0.5 * nu * (D * np.log(2.0) - np.log(det(V))) \
    #     + gammaln(np.arange(nu+1-D,nu+1)*0.5).sum()
    lnZ = 0.5 * nu * (np.log(det(V))) \
        + gammaln(np.arange(nu+1-D,nu+1)*0.5).sum()

    return lnZ


def E_lndetW_Wishart(nu,V):
    """
    mean of log determinant of precision matrix over Wishart <lndet(W)>
    input
      nu [float] : dof parameter of Wichart distribution
      V [ndarray, shape (D x D)] : base matrix of Wishart distribution
    """
    if nu < len(V) + 1:
        raise "dof parameter nu must larger than len(V)"

    D = len(V)
    # E = D*np.log(2.0)- np.log(det(V)) + \
    #     digamma(np.arange(nu+1-D,nu+1)*0.5).sum()
    E = np.log(det(V)) + \
        digamma(np.arange(nu+1-D,nu+1)*0.5).sum()

    return E

def KL_Wishart(nu1,V1,nu2,V2):
    """
    KL-div of Wishart distribution KL[q(nu1,V1)||p(nu2,V2)]
    """
    if nu1 < len(V1) + 1:
        raise "dof parameter nu1 must larger than len(V1)"

    if nu2 < len(V2) + 1:
        raise "dof parameter nu2 must larger than len(V2)"

    if len(V1) != len(V2):
        raise "dimension of two matrix dont match, %d and %d"%(len(V1),len(V2))

    D = len(V1)
    # KL = 0.5 * ( (nu1 -nu2) * E_lndetW_Wishart(nu1,V1) \
    #     + nu1 * (np.trace(solve(V1,V2))- D)) \
    #     - lnZ_Wishart(nu1,V1) + lnZ_Wishart(nu2,V2)
    KL = 0.5 * ( (nu1 -nu2) * E_lndetW_Wishart(nu1,V1) \
        + nu1 * (np.trace(solve(V2,V1))- D)) \
        - lnZ_Wishart(nu1,V1) + lnZ_Wishart(nu2,V2)

    if KL < _small_negative_number :
        print (nu1,nu2,V1,V2)
        raise  "KL must be larger than 0"

    return KL


def KL_GaussWishart(nu1,V1,beta1,m1,nu2,V2,beta2,m2):
    """
    KL-div of Gauss-Wishart distr KL[q(nu1,V1,beta1,m1)||p(nu2,V2,beta2,m2)
    """
    if len(m1) != len(m2):
        raise "dimension of two mean dont match, %d and %d"%(len(m1),len(m2))

    D = len(m1)
    # print("D", D)

    # first assign KL of Wishart
    KL1 = KL_Wishart(nu1,V1,nu2,V2)

    # the rest terms
    KL2 = 0.5 * (D * (np.log(beta1/float(beta2)) + beta2/float(beta1) - 1.0) \
        + beta2 * nu1 * np.dot((m1-m2),solve(V1,(m1-m2))))
    # print("beta1", beta1)
    # print("float(beta2)", float(beta2))
    # print("np.log(beta1/float(beta2))", np.log(beta1/float(beta2)))
    # print("beta2/float(beta1) - 1.0", beta2/float(beta1) - 1.0)
    # print("solve(V1,(m1-m2))",solve(V1,(m1-m2)))
    # print("np.dot((m1-m2),solve(V1,(m1-m2)))",np.dot((m1-m2),solve(V1,(m1-m2))))

    KL = KL1 + KL2

    if KL < _small_negative_number :
        raise "KL must be larger than 0"

    return KL





    
