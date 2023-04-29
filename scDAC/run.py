#%%
from os import path
from os.path import join as pj #os.path 模块主要用于获取文件的属性。join:把目录和文件名合成一个路径
import time
import argparse #argparse是python用于解析命令行参数和选项的标准模块 https://www.cnblogs.com/yibeimingyue/p/13800159.html

from tqdm import tqdm #迭代，进度条
import math
import numpy as np
import torch as th
import pandas as pd
import os
from torch import nn, autograd #autograd：提供了类和函数用来对任意标量函数进行求导
import matplotlib.pyplot as plt
import umap

from modules import models, utils
from modules.datasets import MultimodalDataset
from modules.datasets import MultiDatasetSampler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
parser = argparse.ArgumentParser()
## Task
parser.add_argument('--task', type=str, default='baron_single',
    help="Choose a task")
parser.add_argument('--reference', type=str, default='',
    help="Choose a reference task")
parser.add_argument('--exp', type=str, default='e0',
    help="Choose an experiment")
parser.add_argument('--model', type=str, default='default',
    help="Choose a model configuration")
# parser.add_argument('--data', type=str, default='sup',
#     help="Choose a data configuration")
parser.add_argument('--action', type=str, default='train',
    help="Choose an action to run")
parser.add_argument('--method', type=str, default='scDAC',
    help="Choose an method to benchmark")
parser.add_argument('--init-model', type=str, default='',
    help="Load a saved model")
parser.add_argument('--mods-conditioned', type=str, nargs='+', default=[],
    help="Modalities conditioned for sampling")
parser.add_argument('--data-conditioned', type=str, default='prior.csv',
    help="Data conditioned for sampling")
parser.add_argument('--sample-num', type=int, default=0,
    help='Number of samples to be generated')
parser.add_argument('--input-mods', type=str, nargs='+', default=[],
    help="Input modalities for transformation")
## Training
parser.add_argument('--epoch-num', type=int, default=500,
    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4,
    help='Learning rate')
#parser.add_argument('--dim_logitx', type=int, default=64,
#    help='dim_logitx')
parser.add_argument('--grad-clip', type=float, default=-1,
    help='Gradient clipping value')
parser.add_argument('--s-drop-rate', type=float, default=0.1,
    help="Probility of dropping out subject ID during training")
parser.add_argument('--seed', type=int, default=-1,
    help="Set the random seed to reproduce the results")
parser.add_argument('--use-shm', type=int, default=1,
    help="Use shared memory to accelerate training")
## Debugging
parser.add_argument('--print-iters', type=int, default=-1,
    help="Iterations to print training messages")
parser.add_argument('--log-epochs', type=int, default=100,
    help='Epochs to log the training states')
parser.add_argument('--save-epochs', type=int, default=1,
    help='Epochs to save the latest training states (overwrite previous ones)')
parser.add_argument('--time', type=int, default=0, choices=[0, 1],
    help='Time the forward and backward passes')
parser.add_argument('--debug', type=int, default=1, choices=[0, 1],
    help='Print intermediate variables')
# o, _ = parser.parse_known_args()  # for python interactive
o = parser.parse_args()
# path_label = '/root/asj/asj/2023/0118/sc-transformer-gmvaextoytoz/data/z1/label_seurat/label.csv'
if o.task == 'chen_10':
    path_label = './data/label_chen_10.csv'
else:
    path_label = './data/label_chen_8.csv'
# Initialize global varibles
data_config = None
net = None
#discriminator = None 
optimizer_net = None
#optimizer_disc = None
benchmark = {
    "train_loss": [],
    "test_loss": [],
    "foscttm": [],
    "epoch_id_start": 0
}


def main():
    initialize()
    if o.action == "print_model":
        print_model() ##
    elif o.action == "train":
        train() #
    elif o.action == "test":
        test() ###
    elif o.action == "infer_latent":
        infer_latent(save_input=True)###

    else:
        assert False, "Invalid action!"


def initialize():
    init_seed() ##
    init_dirs() ##
    load_data_config() ##
    load_model_config() ##
    get_gpu_config() ##
    init_model() ##


def init_seed():
    if o.seed >= 0:
        np.random.seed(o.seed) #生成指定随机数
        th.manual_seed(o.seed) #设置CPU生成随机数的种子，方便下次复现实验结果
        th.cuda.manual_seed_all(o.seed)#为当前GPU设置随机种子


def init_dirs():
    if o.use_shm == 1:
        o.data_dir = pj("./data",  o.task)
    else:
        o.data_dir = pj("data", "processed", o.task)
    o.result_dir = pj("result", o.task, o.exp, o.model)
    if o.reference == '': 
        o.train_dir = pj("result", o.task, o.exp, o.model, "train")
    else:
        o.train_dir = pj("result", o.reference, o.exp, o.model, "train")
    o.debug_dir = pj(o.result_dir, "debug")
    utils.mkdirs([o.train_dir, o.debug_dir])
    print("Task: %s\nExperiment: %s\nModel: %s\n" % (o.task, o.exp, o.model))


def load_data_config():
    get_dims_x()
    o.mods = list(o.dims_x.keys())
    o.mod_num = len(o.dims_x)
    global data_config
    data_config = utils.load_toml("configs/data.toml")[o.task]
    for k, v in data_config.items():
        vars(o)[k] = v

    o.s_joint, o.combs, o.s, o.dims_s = utils.gen_all_batch_ids(o.s_joint, o.combs)
    

    
    if o.reference != '':
        data_config_ref = utils.load_toml("configs/data.toml")[o.reference]
        _, _, _, o.dims_s = utils.gen_all_batch_ids(data_config_ref["s_joint"], 
                                                    data_config_ref["combs"])



def load_model_config():
    model_config = utils.load_toml("configs/model.toml")["default"]
    if o.model != "default":
        model_config.update(utils.load_toml("configs/model.toml")[o.model])
    for k, v in model_config.items():
        vars(o)[k] = v
    o.dim_z = o.dim_c
    o.dims_dec_x = o.dims_enc_x[::-1]
    if "dims_enc_chr" in vars(o).keys():
        o.dims_dec_chr = o.dims_enc_chr[::-1]
    o.dims_h = {}
    for m, dim in o.dims_x.items():
        o.dims_h[m] = dim if m != "atac" else o.dims_enc_chr[-1] * 22
    print("dims_h:", o.dims_h)

def get_gpu_config():
    o.G = 1  # th.cuda.device_count()  # get GPU number
    o.N = 512
    assert o.N % o.G == 0, "Please ensure the mini-batch size can be divided " \
        "by the GPU number"
    o.n = o.N // o.G
    print("Total mini-batch size: %d, GPU number: %d, GPU mini-batch size: %d" % (o.N, o.G, o.n))


def init_model():
    """
    Initialize the model, optimizer, and benchmark
    """
    global net, optimizer_net
    net = models.Net_DP(o).cuda()
    optimizer_net = th.optim.AdamW(net.parameters(), lr=o.lr)
    if o.init_model != '':
        fpath = pj(o.train_dir, o.init_model)
        savepoint = th.load(fpath+".pt")
        net.load_state_dict(savepoint['net_states'])
        optimizer_net.load_state_dict(savepoint['optim_net_states'])
        benchmark.update(utils.load_toml(fpath+".toml")['benchmark'])
        print('Model is initialized from ' + fpath + ".pt")
    net_param_num = sum([param.data.numel() for param in net.parameters()])
    print('Parameter number: %.3f M' % (net_param_num / 1e6))


def print_model():
    #global net, discriminator
    global net
    with open(pj(o.result_dir, "model_architecture.txt"), 'w') as f:
        print(net, file=f)


def get_dims_x():
    dims_x = utils.load_csv(pj(o.data_dir, "feat", "feat_dims.csv"))
    dims_x = utils.transpose_list(dims_x)
    o.dims_x = {}
    for i in range(1, len(dims_x)):
        m = dims_x[i][0]
        if m == "atac":
            o.dims_chr = list(map(int, dims_x[i][1:]))
            o.dims_x[m] = sum(o.dims_chr)
        else:
            o.dims_x[m] = int(dims_x[i][1])


    print("Input feature numbers: ", o.dims_x)


def train():
    train_data_loader_cat = get_dataloader_cat("train")
    epoch_id_list = []
    ari_list = []
    nmi_list = []
    sc_list = []

    for epoch_id in range(benchmark['epoch_id_start'], o.epoch_num):
        run_epoch(train_data_loader_cat, "train", epoch_id)


        z = infer_latent_dp(save_input=False)
        net.loss_calculator_dp.mean_dp, net.loss_calculator_dp.weight_concentration_dp,net.loss_calculator_dp.mean_precision_dp,net.loss_calculator_dp.precisions_cholesky_dp, net.loss_calculator_dp.degrees_of_freedom_dp, net.scdp.predict_label = dp(z)
        ari, nmi, sc = cluster_index_calculer(z, net.scdp.predict_label)
        print("ari:", ari)
        print("nmi:", nmi)
        print("sc:", sc)
        epoch_id_list.append(epoch_id)
        ari_list.append(ari)
        nmi_list.append(nmi)
        sc_list.append(sc)
        plt_ari(epoch_id_list, ari_list)
        plt_nmi(epoch_id_list, nmi_list)
        plt_sc(epoch_id_list, sc_list)

        check_to_save(epoch_id)

def dp(z):
    # z_np = (z).cpu().detach().numpy()   
    z_np = z.cpu().detach().numpy()   
    bgm = BayesianGaussianMixture(
        n_components=50, weight_concentration_prior=1e-10,mean_precision_prior = 80,covariance_type='diag',init_params ='kmeans', max_iter=1000, warm_start = True
        ).fit(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label = th.Tensor(np.array(predict_label_array)).unsqueeze(1).cuda()
    mean_dp = th.Tensor(np.array(bgm.means_))
    weight_concentration_dp = th.Tensor(np.array(bgm.weight_concentration_))
    precisions_cholesky_dp = th.Tensor(np.array(bgm.precisions_cholesky_))
    degrees_of_freedom_dp = th.Tensor(np.array(bgm.degrees_of_freedom_))
    mean_precision_dp = th.Tensor(np.array(bgm.mean_precision_))   
    return mean_dp, weight_concentration_dp, mean_precision_dp, precisions_cholesky_dp, degrees_of_freedom_dp, predict_label

def dp_infer(z):
    # z_np = (z).cpu().detach().numpy()   
    z_np = z.cpu().detach().numpy()   
    bgm = BayesianGaussianMixture(
        n_components=50, weight_concentration_prior=1e-10,mean_precision_prior = 10, covariance_type='diag', n_init = 10, init_params ='kmeans', max_iter=1000, warm_start = True
        ).fit(z_np)
    predict_label_array = bgm.predict(z_np)
    predict_label = th.Tensor(np.array(predict_label_array)).unsqueeze(1).cuda()
    mean_dp = th.Tensor(np.array(bgm.means_))
    weight_concentration_dp = th.Tensor(np.array(bgm.weight_concentration_))
    precisions_cholesky_dp = th.Tensor(np.array(bgm.precisions_cholesky_))
    degrees_of_freedom_dp = th.Tensor(np.array(bgm.degrees_of_freedom_))
    mean_precision_dp = th.Tensor(np.array(bgm.mean_precision_))   
    return mean_dp, weight_concentration_dp, mean_precision_dp, precisions_cholesky_dp, degrees_of_freedom_dp, predict_label


def cluster_index_calculer(z_all, predict_label):
    z_all_cpu = z_all.cpu()
    predict_label_cpu = predict_label.cpu()
    label_true = utils.load_csv(path_label)
    label_tlist = utils.transpose_list(label_true)[1][1:]
    label_plist = utils.transpose_list(predict_label_cpu)[0]
    ari = adjusted_rand_score(label_tlist, label_plist) #l1 kpca20
    nmi = normalized_mutual_info_score(label_tlist, label_plist)
    sc = silhouette_score(z_all_cpu, label_plist)
    
    return ari, nmi, sc

def plt_ari(epoch_id_list, ari_list):
    y = ari_list
    x = epoch_id_list
    plt.subplots(figsize = (50, 4))
    plt.bar(x, y, width=0.8)
    plt.xticks(x)  # 绘制x刻度标签
    
    fig_dir = pj(o.result_dir, "represent", "fig")
    utils.mkdirs(fig_dir, remove_old=False)
    plt.savefig(pj(fig_dir, "ari.png"))

def plt_nmi(epoch_id_list, nmi_list):
    y = nmi_list
    x = epoch_id_list
    plt.subplots(figsize = (50, 4))
    plt.bar(x, y, width=0.8)
    plt.xticks(x)  # 绘制x刻度标签
    
    fig_dir = pj(o.result_dir, "represent", "fig")
    plt.savefig(pj(fig_dir, "nmi.png"))


def plt_sc(epoch_id_list, sc_list):
    y = sc_list
    x = epoch_id_list
    plt.subplots(figsize = (50, 4))
    plt.bar(x, y, width=0.8)
    plt.xticks(x)  # 绘制x刻度标签
    
    fig_dir = pj(o.result_dir, "represent", "fig")
    plt.savefig(pj(fig_dir, "sc.png"))

def get_dataloaders(split, train_ratio=None):
    data_loaders = {}
    for subset in range(len(o.s)):
        data_loaders[subset] = get_dataloader(subset, split, train_ratio=train_ratio)
    return data_loaders


def get_dataloader(subset, split, train_ratio=None):
    dataset = MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio)
    shuffle = True if split == "train" else False
    # shuffle = False
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N, shuffle=shuffle,
                                           num_workers=64, pin_memory=True)
    print("Subset: %d, modalities %s: %s size: %d" %
          (subset, str(o.combs[subset]), split, dataset.size))
    return data_loader


def get_dataloader_cat(split, train_ratio=None):
    datasets = []
    for subset in range(len(o.s)):
        datasets.append(MultimodalDataset(o.task, o.data_dir, subset, split, train_ratio=train_ratio))
        print("Subset: %d, modalities %s: %s size: %d" %  (subset, str(o.combs[subset]), split,
            datasets[subset].size))
    dataset_cat = th.utils.data.dataset.ConcatDataset(datasets)
    shuffle = True if split == "train" else False
    # shuffle = False
    sampler = MultiDatasetSampler(dataset_cat, batch_size=o.N, shuffle=shuffle)
    data_loader = th.utils.data.DataLoader(dataset_cat, batch_size=o.N, sampler=sampler, 
        num_workers=64, pin_memory=True)
    return data_loader



def get_eval_dataloader(train_ratio=False):
    data_config_new = utils.copy_dict(data_config)
    data_config_new.update({"combs": [o.mods], "comb_ratios": [1]})
    if train_ratio:
        data_config_new.update({"train_ratio": train_ratio})
    dataset = MultimodalDataset(data_config_new, o.mods, "test")
    data_loader = th.utils.data.DataLoader(dataset, batch_size=o.N,
        shuffle=False, num_workers=64, pin_memory=True)
    print("Eval Dataset %s: test: %d\n" % (str(o.mods), dataset.size))
    return data_loader


def test():
    data_loaders = get_dataloaders()
    run_epoch(data_loaders, "test")


def run_epoch(data_loader, split, epoch_id=0):
    if split == "train":
        net.train()
    elif split == "test":
        net.eval()
    else:
        assert False, "Invalid split: %s" % split
    net.o.epoch_id = epoch_id
    loss_total = 0
    for i, data in enumerate(data_loader):
        loss = run_iter(split, data)
        loss_total += loss
        if o.print_iters > 0 and (i+1) % o.print_iters == 0:
            print('Epoch: %d/%d, Batch: %d/%d, %s loss: %.3f' % (epoch_id+1,
            o.epoch_num, i+1, len(data_loader), split, loss))
    loss_avg = loss_total / len(data_loader)
    print('Epoch: %d/%d, %s loss: %.3f\n' % (epoch_id+1, o.epoch_num, split, loss_avg))
    benchmark[split+'_loss'].append((float(epoch_id), float(loss_avg)))
    return loss_avg



def run_iter(split, inputs):
    inputs = utils.convert_tensors_to_cuda(inputs)
    if split == "train":
        with autograd.set_detect_anomaly(o.debug == 1):
            loss_net = forward_net(inputs)
            loss = loss_net
            update_net(loss) 
            
            
            
    else:
        with th.no_grad():
            loss_net = forward_net(inputs)
            loss = loss_net
    
    return loss.item()


def forward_net(inputs):
    return net(inputs)


def update_net(loss):
    update(loss, net, optimizer_net)

    

def update(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()

    if o.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    optimizer.step()


def check_to_save(epoch_id):
    if (epoch_id+1) % o.log_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_%08d" % epoch_id)
    if (epoch_id+1) % o.save_epochs == 0 or epoch_id+1 == o.epoch_num:
        save_training_states(epoch_id, "sp_latest")


def save_training_states(epoch_id, filename):
    benchmark['epoch_id_start'] = epoch_id
    utils.save_toml({"o": vars(o), "benchmark": benchmark}, pj(o.train_dir, filename+".toml"))
    th.save({"net_states": net.state_dict(),
             "optim_net_states": optimizer_net.state_dict(),
            }, pj(o.train_dir, filename+".pt"))




def infer_latent_dp(save_input=False):
    print("Inferring ...")
    dirs = {}
    base_dir = pj(o.result_dir, "represent", o.init_model)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            dirs[subset_id] = {"z": {}, "x_r_pre": {}, "x": {}}
            dirs[subset_id]["z"]["rna"] = pj(base_dir, "subset_"+str(subset_id), "z", "rna")
            utils.mkdirs(dirs[subset_id]["z"]["rna"], remove_old=False)          
            z_list = []
            if save_input:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["x"][m] = pj(base_dir, "subset_"+str(subset_id), "x", m)
                    utils.mkdirs(dirs[subset_id]["x"][m], remove_old=True)
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"    
            for i, data in enumerate(data_loader):
                data = utils.convert_tensors_to_cuda(data)
                _, z= net.scdp(data)      
                z_list.append(z)
                z_all = th.cat(z_list, dim = 0)
    return(z_all)

def infer_latent(save_input=False):
    print("Inferring ...")
    dirs = {}
    base_dir = pj(o.result_dir, "represent", o.init_model)
    data_loaders = get_dataloaders("test", train_ratio=0)
    net.eval()
    with th.no_grad():
        for subset_id, data_loader in data_loaders.items():
            print("Processing subset %d: %s" % (subset_id, str(o.combs[subset_id])))
            dirs[subset_id] = {"z": {}, "x": {}}
            dirs[subset_id]["z"]["rna"] = pj(base_dir, "subset_"+str(subset_id), "z", "rna")
            utils.mkdirs(dirs[subset_id]["z"]["rna"], remove_old=True)   
            if save_input:
                for m in o.combs[subset_id]:
                    dirs[subset_id]["x"][m] = pj(base_dir, "subset_"+str(subset_id), "x", m)
                    utils.mkdirs(dirs[subset_id]["x"][m], remove_old=True)
            fname_fmt = utils.get_name_fmt(len(data_loader))+".csv"
            
            for i, data in enumerate(tqdm(data_loader)):
                data = utils.convert_tensors_to_cuda(data)
                _, z= net.scdp(data) 
                utils.save_tensor_to_csv(z, pj(dirs[subset_id]["z"]["rna"], fname_fmt) % i)
                if save_input:
                    for m in o.combs[subset_id]:
                        utils.save_tensor_to_csv(data["x"][m], pj(dirs[subset_id]["x"][m], fname_fmt) % i)

                # conditioned on each individual modalities
                if i >0:
                    z_all = th.cat((z_all, z), dim = 0)
                else:
                    z_all = z
            _, _, _, _, _, predict_label = dp_infer(z_all)
            predict_label_list = utils.convert_tensor_to_list(predict_label)
            if o.task == 'chen_10':
                utils.save_list_to_csv(predict_label_list, "./data/chen_10/predict_label.csv")
            else:
                utils.save_list_to_csv(predict_label_list, "./data/chen_8/predict_label.csv")          

            z_all_cpu = z_all.cpu()
            predict_label_cpu = predict_label.cpu()
            label_true = utils.load_csv(path_label)
            label_tlist = utils.transpose_list(label_true)[1][1:]
            label_plist = utils.transpose_list(predict_label_cpu)[0]
            ari = adjusted_rand_score(label_tlist, label_plist) #l1 kpca20
            nmi = normalized_mutual_info_score(label_tlist, label_plist)
            sc = silhouette_score(z_all_cpu, label_plist)
            print("ari:", ari)
            print("nmi:", nmi)
            print("sc:", sc)



main()

# %%
