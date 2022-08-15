# coding=utf-8
import torch
from torch import nn, optim
import hydra
from pathlib import Path
from model.models import ShareNetwork, PrpsyNetwork, Mu1Network, Mu0Network, TauNetwork, ESX
from model.dataset import ESXDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
import logging
import sys, os, shutil
import time
import traceback
import random
from util import wasserstein_torch, mmd2_torch


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# import metrics to evaluate models
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
WORK_DIR = Path().resolve()
torch.autograd.set_detect_anomaly(True)


def hook_for_bkwd(module, input_grad, output_grad):
    # print("input grad {}".format(input_grad))
    # print("output grad {}".format(output_grad))
    for i in input_grad:
        if ~torch.isfinite(i).all():
            print("input grad {}".format(input_grad))
    for j in output_grad:
        if ~torch.isfinite(j).all():
            print("output grad {}".format(output_grad))


def validation_split(yt, val_fraction):
    """ Construct a train/validation split """
    n = len(yt)

    if val_fraction > 0:
        n_valid = int(val_fraction * n)
        n_train = n - n_valid
        I = np.random.permutation(range(0, n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid


''' genarate symt data, just for test '''
def gen_data(n, dim):
    n_data = torch.ones(n, dim)
    x0 = torch.normal(1 * n_data, 1)  # + 0.1*torch.normal(0.01*n_data,1)
    y0 = torch.zeros(n)
    x1 = torch.normal(1.5 * n_data, 1)  # + 0.1*torch.normal(0.01*n_data,1)
    y1 = torch.ones(n)
    data = { 'x': np.expand_dims(torch.cat((x0, x1), 0).cpu().detach().numpy(), axis=2)
        , 'yf': np.expand_dims(torch.cat((y0, y1), 0).cpu().detach().numpy(), axis=1)
             }

    data['e'] = np.ones_like(data["yf"])
    data['t'] = np.ones_like(data["yf"])
    data['I'] = np.ones_like(data["yf"])
    data['ycf'] = 1 - data["yf"]
    _, I_invers = validation_split(data["yf"], 0.2)
    data['e'][I_invers] = 0
    _, I_invers = validation_split(data["yf"], 0.1)
    data['t'][I_invers] = 0
    _, I_invers = validation_split(data["yf"], 0.1)
    data['I'][I_invers] = 0
    # data['ycf'][I_invers] = 0

    # shfful
    I = np.random.permutation(range(0, 2 * n))
    data['x'] = data['x'][I]
    data['yf'] = data['yf'][I]

    data['HAVE_TRUTH'] = not data['ycf'] is None
    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]
    return data


def load_data(fname):
    """ Load data set """
    data_in = np.load(fname)
    data = { 'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf'] }
    try:
        data['ycf'] = data_in['ycf']
        data["mu0"] = data_in['mu0']
        data["mu1"] = data_in['mu1']
    except:
        data['ycf'] = None

    try:
        data['e'] = data_in['e']
        if (len(data['e']) < 1):
            data['e'] = np.zeros_like(data_in['yf'])
    except:
        data['e'] = np.zeros_like(data_in['yf'])

    try:
        data['tau'] = data_in['tau']
        data['IS_SYNT'] = True
    except:
        data['tau'] = np.array([None])
        data['IS_SYNT'] = False
    try:
        data['I'] = data_in['I']
    except:
        data['I'] = np.ones_like(data_in['yf'])

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data


# def get_embedding_dims(train_X: pd.DataFrame, embedding_dim) -> Tuple(List((int, int)), int):
#     """Get embedding layer size and concat_feature_vec_dim"""
#     field_dims = list(train_X.max())
#     field_dims = list(map(lambda x: x+1, field_dims))
#
#     embedding_sizes = []
#     concat_dim = 0
#     for field_dim in field_dims:
#         embedding_sizes.append((field_dim, embedding_dim))
#         concat_dim += embedding_dim
#     return embedding_sizes, concat_dim

def validation_split(x, val_fraction):
    """ Construct a train/validation split """
    n = x.shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction * n)
        n_train = n - n_valid
        I = np.random.permutation(range(0, n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = list(range(n))
        I_valid = []

    return I_train, I_valid


def evalWithData(group_name, model, writer, step_or_epoch, p_t, cfg, x, yf, t, e, eff_tau=None, i_exp=None):
    logging.info("group_name:{}, evalWithData... -----------------------------------".format(group_name))
    writer_flag = not writer is None

    p_t = torch.mean(t).item()
    if cfg.reweight_sample:
        w_t = t / (
                2 * p_t)
        w_c = (1 - t) / (2 * (1 - p_t))
        sample_weight = w_t + w_c
    else:
        sample_weight = torch.ones_like(t)
        p_t = 0.5

    # set loss functions
    sample_weight = sample_weight[~e.bool()]
    loss_w_fn = nn.BCELoss(weight=sample_weight)  # for probability
    loss_fn = nn.BCELoss()  # for probability
    loss_mse = nn.MSELoss()
    loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
    loss_w_with_logit_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

    p_prpsy_logit, p_escvr1, p_escvr0, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = model(
        x)


    # as default: tau = h1-h0
    p_tau = p_h1 - p_h0
    p_yf = p_h1 * t + p_h0 * (1 - t)
    p_ycf = p_h0 * t + p_h1 * (1 - t)


    # just for print log
    iter_name = "epoch"
    if i_exp == 0:
        iter_name = "train_step"

    # group_name=="test_set" and dataset is ramdomized, we can get auuc.
    if group_name in ("test_set", "test_pred_result"):
        auuc_score = qini_auc_score(yf.reshape(-1).cpu().numpy(), p_tau.reshape(-1).cpu().numpy(),
                                    t.reshape(-1).cpu().numpy())
        logging.info("group_name {}, {}, {}, auuc_score: {}".format(group_name, iter_name, step_or_epoch, auuc_score))
        if writer_flag:
            writer.add_scalar("{}/auuc_score".format(group_name), auuc_score, step_or_epoch)

    prpsy_loss = cfg.prpsy_w * loss_w_with_logit_fn(p_prpsy_logit[~e.bool()], t[~e.bool()])
    estr_loss = cfg.escvr1_w * loss_w_fn(p_escvr1[~e.bool()], (yf * t)[~e.bool()])
    escr_loss = cfg.escvr0_w * loss_w_fn(p_escvr0[~e.bool()], (yf * (1 - t))[~e.bool()])

    h1_loss = cfg.h1_w * loss_fn(p_h1[t.bool()], yf[t.bool()])  # * (1 / (2 * p_t))
    h0_loss = cfg.h0_w * loss_fn(p_h0[~t.bool()], yf[~t.bool()])  # * (1 / (2 * (1 - p_t)))

    # h1_loss = cfg.mu1_w * loss_with_logit_fn(p_mu1_logit[t.bool()], yf[t.bool()])  # * (1 / (2 * p_t))
    # h0_loss = cfg.mu0_w * loss_with_logit_fn(p_mu0_logit[~t.bool()], yf[~t.bool()])  # * (1 / (2 * (1 - p_t)))

    # X loss for h1,mu1
    # x1loss = cfg.x1loss_w * loss_mse(p_h1, p_mu1)
    # # X loss for h0,mu0
    # x0loss = cfg.x0loss_w * loss_mse(p_h0, p_mu0)

    # bias loss
    imb_dist = 0

    imb_dist_loss = cfg.imb_dist_w * imb_dist

    cross_tr_loss = cfg.mu1hat_w * loss_fn(torch.sigmoid(p_mu0_logit + p_tau_logit)[t.bool()],
                                         yf[t.bool()])
    cross_cr_loss = cfg.mu0hat_w * loss_fn(torch.sigmoid(p_mu1_logit - p_tau_logit)[~t.bool()],
                                         yf[~t.bool()])


    total_loss = prpsy_loss + estr_loss + escr_loss + \
                 h1_loss + h0_loss + \
                 cross_tr_loss + cross_cr_loss + \
                 imb_dist_loss

    logging.info("group_name {}, {} {}, total_loss {}".format(group_name, iter_name, step_or_epoch, total_loss))
    # the p_tau of treatment group
    logging.info(
        "p_tau {}, {}, {} , mean(p_tau[t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                       torch.mean(p_tau[t.bool()]).item()))
    # the p_tau of control group
    logging.info(
        "p_tau {}, {}, {} , mean(p_tau[~t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                        torch.mean(p_tau[~t.bool()]).item()))

    # For virtual data
    if eff_tau is not None:
        # att = torch.mean(yf[t > 0]) - torch.mean(yf[(1 - t) > 0])
        # att_pred = torch.mean(p_tau[t > 0])
        # bias_att = att_pred - att  # the error of att
        # logging.info("group_name {}, epoch {} , bias_att in observation:{}".format(group_name, epoch, bias_att))

        rmse_tau = weighted_rmse_loss(input=p_tau, target=eff_tau)
        logging.info(
            "mse_tau in true {}, {} {} , torch.mean(rmse_tau) :{}".format(group_name, iter_name, step_or_epoch,
                                                                          rmse_tau))
        rmse_tau_t = weighted_rmse_loss(input=p_tau[t.bool()], target=eff_tau[t.bool()])
        logging.info(
            "mse_tau in true {}, {} {} , torch.mean(rmse_tau[t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                                             rmse_tau_t))
        rmse_tau_c = weighted_rmse_loss(input=p_tau[~t.bool()], target=eff_tau[~t.bool()])
        logging.info(
            "mse_tau in true {}, {} {} , torch.mean(rmse_tau[~t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                                              rmse_tau_c))

        if writer_flag:
            writer.add_scalar("{}/rmse_tau(in true)".format(group_name), rmse_tau, step_or_epoch)
            writer.add_scalar("{}/rmse_tau[t](in true)".format(group_name), rmse_tau_t, step_or_epoch)
            writer.add_scalar("{}/rmse_tau[~t](in true)".format(group_name), rmse_tau_c, step_or_epoch)

    if writer_flag:
        writer.add_scalar("{}/escvr1_loss".format(group_name), estr_loss, step_or_epoch)
        writer.add_scalar("{}/escvr0_loss".format(group_name), escr_loss, step_or_epoch)
        # writer.add_scalar("{}/bias_att".format(group_name), bias_att, epoch)
        writer.add_scalar("{}/total_loss".format(group_name), total_loss, step_or_epoch)
        writer.add_scalar("{}/prpsy_loss".format(group_name), prpsy_loss, step_or_epoch)
        writer.add_scalar("{}/h1_loss".format(group_name), h1_loss, step_or_epoch)
        writer.add_scalar("{}/h0_loss".format(group_name), h0_loss, step_or_epoch)

        writer.add_scalar("{}/mu0hat_loss".format(group_name), cross_cr_loss, step_or_epoch)
        writer.add_scalar("{}/mu1hat_loss".format(group_name), cross_tr_loss, step_or_epoch)

        writer.add_scalar("{}/imb_dist".format(group_name), imb_dist, step_or_epoch)

        writer.add_scalar("{}/tau_logit_mean".format(group_name), torch.mean(p_tau_logit).item(),
                          step_or_epoch)

        writer.add_scalar("{}/tau_logit_mean(t=0)".format(group_name), torch.mean(p_tau_logit[(1 - t).bool()]).item(),
                          step_or_epoch)
        writer.add_scalar("{}/tau_logit_mean(t=1)".format(group_name), torch.mean(p_tau_logit[t.bool()]).item(),
                          step_or_epoch)

        writer.add_scalar("{}/tau".format(group_name), torch.mean(p_tau).item(),
                          step_or_epoch)
        writer.add_scalar("{}/tau(t=0)".format(group_name), torch.mean(p_tau[(1 - t).bool()]).item(),
                          step_or_epoch)
        writer.add_scalar("{}/tau(t=1)".format(group_name), torch.mean(p_tau[t.bool()]).item(),
                          step_or_epoch)

        writer.add_scalar("{}/h1".format(group_name), torch.mean(p_h1).item(),
                          step_or_epoch)
        writer.add_scalar("{}/h0".format(group_name), torch.mean(p_h0).item(),
                          step_or_epoch)

        # for mu1
        fpr, tpr, threshold = metrics.roc_curve(yf[t.bool()].cpu().detach().numpy(),
                                                p_mu1[t.bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC, p_mu1(t=1)".format(group_name), roc_auc, step_or_epoch)

        # for mu0
        fpr, tpr, threshold = metrics.roc_curve(yf[(1 - t).bool()].cpu().detach().numpy(),
                                                p_mu0[(1 - t).bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC  p_mu0(t=0)".format(group_name), roc_auc, step_or_epoch)

        # for propensity
        fpr, tpr, threshold = metrics.roc_curve(t.cpu().detach().numpy(),
                                                p_prpsy.cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC  p_prpsy".format(group_name), roc_auc, step_or_epoch)

        # for ESCVR1
        fpr, tpr, threshold = metrics.roc_curve((t * yf).cpu().detach().numpy(),
                                                p_escvr1.cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC  p_escvr1".format(group_name), roc_auc, step_or_epoch)

        # for ESCVR0
        fpr, tpr, threshold = metrics.roc_curve(((1 - t) * yf).cpu().detach().numpy(),
                                                p_escvr0.cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC  p_escvr0".format(group_name), roc_auc, step_or_epoch)

        # for mu0 hat
        p_mu0_hat = torch.sigmoid(p_mu1_logit - p_tau_logit)
        fpr, tpr, threshold = metrics.roc_curve(yf[(1 - t).bool()].cpu().detach().numpy(),
                                                p_mu0_hat[(1 - t).bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC  p_mu0_hat".format(group_name), roc_auc, step_or_epoch)

        # for mu1 hat
        p_mu1_hat = torch.sigmoid(p_mu0_logit + p_tau_logit)
        fpr, tpr, threshold = metrics.roc_curve(yf[t.bool()].cpu().detach().numpy(),
                                                p_mu1_hat[t.bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC, p_mu1_hat".format(group_name), roc_auc, step_or_epoch)


        writer.flush()

    dict_result = { "loss": [total_loss, prpsy_loss, escr_loss, estr_loss, h1_loss, h0_loss, cross_tr_loss, cross_cr_loss],
                    "p_tau": p_tau, "p_yf": p_yf, "p_ycf": p_ycf, "p_prpsy": p_prpsy }
    return dict_result



def weighted_rmse_loss(input, target, weight=1):
    risk = torch.sqrt(torch.mean(weight * torch.square(input - target)))
    return risk
    # return torch.mean(weight * ((input - target) ** 2) )


def train(data_dict, data_test_dict, device, cfg):
    # configs
    share_dim = cfg.share_dim
    base_dim = cfg.base_dim
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    LOGSTEP = cfg.log_step  # training step
    PREDSTEP = cfg.pred_step  # epoch step

    # for test
    x_test_all_exp = data_test_dict["x"]
    yf_test_all_exp = data_test_dict["yf"]
    t_test_all_exp = data_test_dict["t"]
    tau_test_all_exp = data_test_dict["tau"]
    e_test_all_exp = data_test_dict["e"]
    test_dim = data_test_dict['dim']
    test_samples_num = data_test_dict['n']
    HAVE_TRUTH = data_test_dict["HAVE_TRUTH"]
    IS_SYNT = data_test_dict["IS_SYNT"]

    # for train
    x_all_exp = data_dict["x"]
    yf_all_exp = data_dict["yf"]
    tau_all_exp = data_dict["tau"]
    if HAVE_TRUTH:
        ycf_all_exp = data_dict["ycf"]
        mu1f_all_exp = data_dict["mu1"]
        mu0f_all_exp = data_dict["mu0"]

    t_all_exp = data_dict["t"]
    e_all_exp = data_dict["e"]
    dim = data_dict['dim']
    samples_num = data_dict['n']

    ''' Set up for saving variables '''
    result_dict = { }
    for group in ["train", "valid", "test"]:
        result_dict[group] = { "p_prpsy": [], "p_yf": [], "p_ycf": [], "p_tau": [], "loss": [], "val": [] }

    '''init summary'''
    # summary_path = '/home/admin/fengtong/ESX_Model/runs/{}'.format(cfg.model_name)
    summary_path = '{}/{}'.format(cfg.summary_base_dir, cfg.model_name)
    # summary_path = '/home/admin/dufeng/ESX_Model/runs/{}'.format(cfg.model_name)

    if os.path.exists(summary_path):
        logging.info(" shutil.rmtree({}) ...".format(summary_path))
        shutil.rmtree(summary_path)
        time.sleep(0.5)
    else:
        ''' create summary folder'''
        logging.info(" os.mkdir({}) ...".format(summary_path))
        os.mkdir(summary_path)

    writer = SummaryWriter(summary_path)

    # repeat experiment
    for i_exp in range(0, cfg["n_experiments"]):
        # just for debug
        # if i_exp <1 :
        #     continue

        ''' Set up for saving variables for each repeat experiment'''
        # train result
        iexp_p_prpsy = { "train": [], "valid": [], "test": [] }
        iexp_p_yf = { "train": [], "valid": [], "test": [] }
        iexp_p_ycf = { "train": [], "valid": [], "test": [] }
        iexp_p_tau = { "train": [], "valid": [], "test": [] }
        # iexp_val = {"train": [] }
        iexp_losses = { "train": [], "valid": [], "test": [] }

        '''split to training set and validation set '''
        I_train, I_valid = validation_split(yf_all_exp[:, i_exp], cfg.val_rate)

        # pick exp i and I_train for training set that will use to built for dataloader
        x = x_all_exp[I_train, :, i_exp]
        yf = yf_all_exp[I_train, i_exp]
        # if HAVE_TRUTH:#     ycf = ycf_all_exp[I_train, i_exp]
        #     mu1f = mu1f_all_exp[I_train, i_exp]
        #     mu0f = mu0f_all_exp[I_train, i_exp]
        t = t_all_exp[I_train, i_exp]
        e = e_all_exp[I_train, i_exp]  # torch.from_numpy(e_all_exp[:, i_exp]).float().reshape((-1,1))
        # logging.info("for t=1 in train set. x.shape:{},yf.shape:{},t.shape:{},e.shape:{}".format(x[t == 1].shape, yf[t == 1].shape,
        #                                                                        t[t == 1].shape, e[t == 1].shape))
        # logging.info("for t=0. x.shape:{},yf.shape:{},t.shape:{},e.shape:{}".format(x[t == 0].shape, yf[t == 0].shape,
        #                                                                        t[t == 0].shape, e[t == 0].shape))
        # if "sample_alpha" in cfg.keys() and cfg.sample_alpha > 0:
        #     x, yf, e, t = sample_imb_fn(x, yf, e, t, cfg)
        #     logging.info("after sample_imb_fn. for t=1. x.shape:{},yf.shape:{},t.shape:{},e.shape:{}".format(x[t == 1].shape, yf[t == 1].shape,
        #                                                                            t[t == 1].shape, e[t == 1].shape))
        #     logging.info("after sample_imb_fn. for t=0. x.shape:{},yf.shape:{},t.shape:{},e.shape:{}".format(x[t == 0].shape, yf[t == 0].shape,
        #                                                                            t[t == 0].shape, e[t == 0].shape))


        logging.info("training set: x.shape:{}".format(x.shape))

        # pick exp i and I_valid validation set and convert to tensor dtype.
        x_valid = torch.from_numpy(x_all_exp[I_valid, :, i_exp]).float().to(device)
        yf_valid = torch.from_numpy(yf_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)
        # if HAVE_TRUTH:
        #     ycf_valid = torch.from_numpy(ycf_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)
        #     mu1f_valid = torch.from_numpy(mu1f_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)
        #     mu0f_valid = torch.from_numpy(mu0f_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)
        t_valid = torch.from_numpy(t_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)
        e_valid = torch.from_numpy(e_all_exp[I_valid, i_exp]).float().reshape((-1, 1)).to(device)

        # pick exp i for test set and convert to tensor dtype.
        x_test = torch.from_numpy(x_test_all_exp[:, :, i_exp]).float().to(device)
        yf_test = torch.from_numpy(yf_test_all_exp[:, i_exp]).float().reshape((-1, 1)).to(device)
        if IS_SYNT:
            tau_test = torch.from_numpy(tau_test_all_exp[:, i_exp]).float().reshape((-1, 1)).to(device)

        t_test = torch.unsqueeze(torch.from_numpy(t_test_all_exp[:, i_exp]).float().to(device), 1)
        e_test = torch.unsqueeze(torch.from_numpy(e_test_all_exp[:, i_exp]).float().to(device), 1)

        ''' the whole train set, just use for prediction. '''
        x_ = x_all_exp[:, :, i_exp]
        yf_ = yf_all_exp[:, i_exp]
        t_ = t_all_exp[:, i_exp]
        e_ = e_all_exp[:, i_exp]
        # and convert to tensors
        x_train = torch.from_numpy(x_).float().to(device)
        yf_train = torch.from_numpy(yf_).float().to(device).reshape((-1, 1))
        t_train = torch.from_numpy(t_).float().to(device).reshape((-1, 1))
        e_train = torch.from_numpy(e_).float().to(device).reshape((-1, 1))
        if IS_SYNT:
            tau_train = torch.from_numpy(tau_all_exp[:, i_exp]).float().reshape((-1, 1)).to(device)

        ''' true effect '''
        if IS_SYNT:
            eff_valid = tau_train[I_valid]
            eff_test = tau_test
            eff_train = tau_train
        else:
            eff_valid, eff_test, eff_train = None, None, None

        ''' print the proportion information of training set and test set. '''
        if 0 == i_exp:
            logging.info("exp_{}, Train. x.shape : {}".format(i_exp, x.shape))
            logging.info("exp_{}, Train. mean(t) : {}".format(i_exp, np.mean(t)))
            logging.info("exp_{}, Train. mean(t) when e=1: {}".format(i_exp, np.mean(t[e.astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) : {}".format(i_exp, np.mean(yf)))
            logging.info("exp_{}, Train. mean(yf) when t=1: {}".format(i_exp, np.mean(yf[t.astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) when t=0: {}".format(i_exp, np.mean(yf[(1 - t).astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) when t=0 and e=1: {}".format(i_exp, np.mean(
                yf[(e * (1 - t)).astype(bool)])))
            # if HAVE_TRUTH :
            #     logging.info("exp_{}, Train. mean(tau) : {}".format(i_exp, np.mean(abs(mu1f_ - mu0f_))))
            ''' test set '''
            # print test set
            logging.info("exp_{}, Test. x.shape : {}".format(i_exp, x_test.shape))
            logging.info("exp_{}, Test. mean(t): {}".format(i_exp, torch.mean(t_test.float())))
            logging.info("exp_{}, Test. mean(t) when e=1: {}".format(i_exp, torch.mean(t_test[e_test.bool()].float())))
            logging.info("exp_{}, Test. mean(yf): {}".format(i_exp, torch.mean(yf_test)))
            logging.info("exp_{}, Test. mean(yf) when t=1: {}".format(i_exp, torch.mean(yf_test[t_test.bool()])))
            logging.info("exp_{}, Test. mean(yf) when t=0: {}".format(i_exp, torch.mean(yf_test[~t_test.bool()])))
            logging.info(
                "exp_{}, Test. mean(yf) when t=0 and e=1: {}".format(i_exp, torch.mean(
                    yf_test[((1 - t_test) * e_test).bool()])))
            # if HAVE_TRUTH :
            #     logging.info("exp_{}, Test. mean(tau) : {}".format(i_exp, torch.mean(torch.abs(mu1f_test - mu0f_test))) )

        # tmp = [(5,5),(6,6)]
        # feature_extractor = FeatureExtractor(tmp) # pause

        ''' create graph '''
        shareNetwork = ShareNetwork(input_dim=dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg, device=device)
        prpsy_network = PrpsyNetwork(base_dim, cfg=cfg)
        mu1_network = Mu1Network(base_dim, cfg=cfg)
        mu0_network = Mu0Network(base_dim, cfg=cfg)
        tau_network = TauNetwork(base_dim, cfg=cfg)

        model = ESX(prpsy_network, mu1_network, mu0_network, tau_network, shareNetwork, cfg, device)
        model = model.to(device)

        ''' register for backward_hook '''
        # if i_exp == 0:
        #     print("model :{}".format(model))
        #     model.register_backward_hook(hook_for_bkwd)

        ''' create optimizer '''
        # optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
        if cfg.optim == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.decay_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=cfg.decay_step_size,
                                                       gamma=cfg.decay_rate)

        ''' Build dataloader '''
        dataset = ESXDataset(x, yf, t, e)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start fitting
        model.train()
        if (cfg.verbose):
            logging.info("exp_{} start trainning ...".format(i_exp))

        train_step = 0
        for epoch in range(epochs):
            if ((epoch + 1) % LOGSTEP == 0):
                logging.info("exp_i:{},  epoch:{} ...".format(i_exp, epoch))

            running_total_loss = 0.0
            running_prpsy_loss = 0.0
            running_mu1_loss = 0.0
            running_mu0_loss = 0.0
            running_escvr1_loss = 0.0
            running_escvr0_loss = 0.0

            running_h1_loss = 0.0
            running_h0_loss = 0.0

            # for i, (inputs, t_labels, y_labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            for i, (inputs, t_labels, y_labels, e_labels) in enumerate(train_loader):
                if (y_labels.reshape(-1).shape[0] < batch_size):
                    continue

                model.train()
                # logging.debug("epoch:{} .shape :{}".format(epoch, inputs.shape))
                inputs.to(device)
                t_labels = torch.unsqueeze(t_labels.to(device), 1)
                y_labels = torch.unsqueeze(y_labels.to(device), 1)
                e_labels = torch.unsqueeze(e_labels.to(device), 1) # e_ Labels is used to mark whether it is a random sample

                ''' Compute sample reweighting '''
                p_t = torch.mean(t_labels).item()
                if cfg.reweight_sample:
                    w_t = t_labels / (
                            2 * p_t)
                    w_c = (1 - t_labels) / (2 * (1 - p_t))
                    sample_weight = w_t + w_c
                else:
                    sample_weight = torch.ones_like(t_labels)
                    p_t = 0.5

                # set loss functions
                sample_weight = sample_weight[~e_labels.bool()]
                loss_w_fn = nn.BCELoss(weight=sample_weight)
                loss_fn = nn.BCELoss()
                loss_mse = nn.MSELoss()
                loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
                loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

                # Initialize gradient
                optimizer.zero_grad()
                # caluculate losses
                p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = model(
                    inputs)

                try:
                    # loss for propensity
                    prpsy_loss = cfg.prpsy_w * loss_w_with_logit_fn(p_prpsy_logit[~e_labels.bool()],
                                                                    t_labels[~e_labels.bool()])
                    # loss for ESTR, ESCR
                    estr_loss = cfg.escvr1_w * loss_w_fn(p_estr[~e_labels.bool()],
                                                           (y_labels * t_labels)[~e_labels.bool()])
                    escr_loss = cfg.escvr0_w * loss_w_fn(p_escr[~e_labels.bool()],
                                                           (y_labels * (1 - t_labels))[~e_labels.bool()])

                    #loss for TR, CR
                    tr_loss = cfg.h1_w * loss_fn(p_h1[t_labels.bool()],
                                                 y_labels[t_labels.bool()])  # * (1 / (2 * p_t))
                    cr_loss = cfg.h0_w * loss_fn(p_h0[~t_labels.bool()],
                                                 y_labels[~t_labels.bool()])  # * (1 / (2 * (1 - p_t)))


                    #loss for cross TR: mu1_prime, cross CR: mu0_prime
                    cross_tr_loss = cfg.mu1hat_w * loss_fn(torch.sigmoid(p_mu0_logit + p_tau_logit)[t_labels.bool()],
                                                         y_labels[t_labels.bool()])
                    cross_cr_loss = cfg.mu0hat_w * loss_fn(torch.sigmoid(p_mu1_logit - p_tau_logit)[~t_labels.bool()],
                                                         y_labels[~t_labels.bool()])

                    imb_dist = 0
                    if cfg.imb_dist_w > 0:
                        if cfg.imb_dist == "wass":
                            imb_dist = wasserstein_torch(X=shared_h, t=t_labels)
                        elif cfg.imb_dist == "mmd":
                            imb_dist = mmd2_torch(shared_h, t_labels)
                        else:
                            sys.exit(1)
                    imb_dist_loss = cfg.imb_dist_w * imb_dist

                    total_loss = prpsy_loss + estr_loss + escr_loss \
                                 + tr_loss + cr_loss \
                                 + cross_tr_loss + cross_cr_loss \
                                 + imb_dist_loss



                    # Backpropagation
                    total_loss.backward()
                    # total_loss.backward(retain_graph=True)
                    # Update parameters
                    optimizer.step()

                    running_total_loss += total_loss.item()
                    running_prpsy_loss += prpsy_loss.item()
                    running_mu1_loss += cross_tr_loss.item()
                    running_mu0_loss += cross_cr_loss.item()
                    running_escvr1_loss += estr_loss.item()
                    running_escvr0_loss += escr_loss.item()

                    '''save the graph, just saving for the first experiment.'''
                    if (i_exp == 0 and epoch == 0 and i == 0):
                        writer.add_graph(model, inputs)
                        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        logging.info("model saved. pytorch_total_params is {}".format(pytorch_total_params))

                except Exception as e:
                    logging.info("error message:{}".format(e))
                    logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
                    logging.error("there something wrong when calculating loss.")

                running_total_loss = running_total_loss / (i + 1)
                running_prpsy_loss = running_prpsy_loss / (i + 1)
                running_mu1_loss = running_mu1_loss / (i + 1)
                running_mu0_loss = running_mu0_loss / (i + 1)
                running_escvr1_loss = running_escvr1_loss / (i + 1)
                running_escvr0_loss = running_escvr0_loss / (i + 1)

                # the first experiment only
                if i_exp == 0 and (train_step + 1) % LOGSTEP == 0:
                    if (cfg.verbose):
                        logging.info(f'i_exp:{i_exp},epochs:{epoch}, train_step:{train_step}, imb_dist:{imb_dist}')


                        try:
                            model.eval()
                            with torch.no_grad():
                                # validation
                                evalWithData("valid_set", model, writer, train_step, p_t, cfg, x_valid, yf_valid,
                                             t_valid,
                                             e_valid, eff_valid, i_exp
                                             )
                                # test
                                evalWithData("test_set", model, writer, train_step, p_t, cfg, x_test, yf_test, t_test,
                                             e_test, eff_test, i_exp
                                             )
                                # train
                                evalWithData("train_set ", model, writer, train_step, p_t, cfg, x_train, yf_train,
                                             t_train,
                                             e_train, eff_train, i_exp
                                             )
                        except:
                            logging.error("there are something wrong when eval.")

                ''' end loop for a epoch '''
                train_step = train_step + 1
            # update the learning rate
            lr_scheduler.step()
            # get learning rate
            new_lr = lr_scheduler.get_last_lr()
            logging.info(
                "i_exp:{}, epoch:{} ,lr_scheduler.step() and new learning rate is : {}".format(i_exp, epoch, new_lr))

            ''' end loop for epochs '''

            ''' predict every VALSTEP step and get results for each experiments'''
            if (epoch + 1) % PREDSTEP == 0:
                model.eval()
                # start to evel
                logging.info(f'start to predict ... i_exp:{i_exp},epochs:{epoch}, train_step:{train_step}')
                with torch.no_grad():
                    # for test
                    dict_result = evalWithData("test_pred_result", model, None, epoch, p_t, cfg, x_test, yf_test,
                                               t_test,
                                               e_test, eff_test
                                               )
                    # append to list. for saving.
                    iexp_p_prpsy["test"].append(dict_result["p_prpsy"].cpu().detach().numpy()[:, 0])
                    iexp_p_yf["test"].append(dict_result["p_yf"].cpu().detach().numpy()[:, 0])
                    iexp_p_ycf["test"].append(dict_result["p_ycf"].cpu().detach().numpy()[:, 0])
                    iexp_p_tau["test"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                    iexp_losses["test"].append(dict_result["loss"])

                    if cfg.verbose > 0:
                        # for the whole training set
                        dict_result = evalWithData("train_pred_result", model, None, epoch, p_t, cfg, x_train, yf_train,
                                                   t_train,
                                                   e_train, eff_train
                                                   )
                        # append to list. for saving.
                        iexp_p_prpsy["train"].append(dict_result["p_prpsy"].cpu().detach().numpy()[:, 0])
                        iexp_p_yf["train"].append(dict_result["p_yf"].cpu().detach().numpy()[:, 0])
                        iexp_p_ycf["train"].append(dict_result["p_ycf"].cpu().detach().numpy()[:, 0])
                        iexp_p_tau["train"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                        # iexp_losses["train"].append(dict_result["loss"])
                        train_total_loss = dict_result["loss"]  # loss for the whole training set

                        # for validation
                        dict_result = evalWithData("valid_pred_result", model, None, epoch, p_t, cfg, x_valid, yf_valid,
                                                   t_valid,
                                                   e_valid, eff_valid
                                                   )
                        # # append to list. for saving.
                        # iexp_p_prpsy["valid"].append(dict_result["p_prpsy"].cpu().detach().numpy()[:, 0])
                        # iexp_p_yf["valid"].append(dict_result["p_yf"].cpu().detach().numpy()[:, 0])
                        # iexp_p_ycf["valid"].append(dict_result["p_ycf"].cpu().detach().numpy()[:, 0])
                        # iexp_p_tau["valid"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                        iexp_losses["train"].append(train_total_loss + [dict_result["loss"][0]])  # append validation的loss

        # only save the model for the first experiment.
        if i_exp == 0 and cfg.verbose:
            logging.info("exp_{} model saving...".format(i_exp))
            torch.save(model, "./{}_p{}.pth".format(cfg.model_name, i_exp))
            logging.info("exp_{} model saving...done.".format(i_exp))
            writer.close()

        if cfg.verbose > 0:
            group_list = ["train", "test"]
        else:
            group_list = ["test"]
        ''' save preidctions '''
        for group in group_list:
            # {"p_prpsy":[], "p_yf":[], "p_ycf":[], "p_tau":[], "loss":[]}
            result_dict[group]["p_prpsy"].append(iexp_p_prpsy[group])
            result_dict[group]["p_yf"].append(iexp_p_yf[group])
            result_dict[group]["p_ycf"].append(iexp_p_ycf[group])
            result_dict[group]["p_tau"].append(iexp_p_tau[group])
            result_dict[group]["loss"].append(iexp_losses[group])
            if group == "train":
                result_dict[group]["val"].append(I_valid)
        ''' 格式化["train", "valid", "test"]数据集的预测结果和loss，并保存到本地'''
        for group in group_list:
            '''units, exp_i, outputs'''
            all_p_prpsy = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_prpsy"], 0, 2), 1, 2))
            all_p_yf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_yf"], 0, 2), 1, 2))
            all_p_ycf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_ycf"], 0, 2), 1, 2))
            all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))
            # all_p_mu1 = np.array(np.swapaxes(np.swapaxes(all_p_mu1, 0, 2), 1, 2))
            # all_p_mu0 = np.array(np.swapaxes(np.swapaxes(all_p_mu0, 0, 2), 1, 2))
            if group == "train":
                ''' exp_i, I_valid_set '''
                all_I_valid = np.array(result_dict[group]["val"])
            else:
                all_I_valid = []

            ''' outputs, loss_list, exp_i '''
            all_losses = np.swapaxes(np.swapaxes(result_dict[group]["loss"], 0, 1), 1, 2)

            logging.info("saving predict result as a file...")
            npz_file_path = "{}/{}_{}_result.test".format(cfg.pred_output_dir, cfg.model_name, group)
            np.savez(npz_file_path, p_prpsy=all_p_prpsy, p_yf=all_p_yf, \
                     p_ycf=all_p_ycf, p_tau=all_p_tau, loss=all_losses, val=all_I_valid)
            logging.info("saving predict result as a file: {}...done".format(npz_file_path))

import math

def sample_imb_fn(x_, yf_, e_, t_, cfg):
    # treatment group size
    if "total_size" in cfg.keys():
        total_size=cfg["total_size"]
    else:
        total_size = 360000

    t_size = math.floor(total_size * (1/(cfg.sample_alpha + 1)))
    # control group size
    c_size = math.floor(total_size * (cfg.sample_alpha/(cfg.sample_alpha + 1)))

    t_true_size = np.sum(t_ == 1)
    c_true_size = np.sum(t_ == 0)

    # treatment group
    x = np.empty(shape=(0, x_.shape[1]), dtype=x_.dtype)
    yf = np.empty(shape=( 0 ), dtype=yf_.dtype)
    t = np.empty(shape=( 0 ), dtype=t_.dtype)
    for i in range(int(t_size/t_true_size)+1):
        i_loc = t_size - i*t_true_size
        x = np.concatenate([x, x_[t_ == 1][:i_loc ]])
        yf = np.concatenate([yf, yf_[t_ == 1][:i_loc]])
        t = np.concatenate([t, t_[t_ == 1][:i_loc]])

    for i in range(int(c_size/c_true_size)+1):
        i_loc = c_size - i * c_true_size
        x = np.concatenate([x, x_[t_ == 0][:i_loc]])
        yf = np.concatenate([yf, yf_[t_ == 0][:i_loc]])
        t = np.concatenate([t, t_[t_ == 0][:i_loc]])
    e = np.zeros_like(t)
    return x, yf, e, t


@hydra.main(config_path='./conf', config_name='conf_lzd_real_bin_v0_full.yaml')
# @hydra.main(config_path='./conf', config_name='conf_lzd_real_bin_v0.yaml')
# @hydra.main(config_path='./conf', config_name='conf_acic_speed.yaml')
def main(cfg):
    logging.info("log testing ...")
    logging.info("cfg:{}".format(cfg))
    logging.debug("cfg:{}".format(cfg))
    # Load data
    logging.info("training dataset loading ...")
    data_dict = load_data(cfg.data_train_path)

    # data_dict = gen_data(10000, 50)
    logging.info("training dataset loading ...done.")

    logging.info("test dataset loading ....")
    data_test_dict = load_data(cfg.data_test_path)
    # data_test_dict = gen_data(5000, 50)
    logging.info("test dataset loading ...done.")

    # Encode dataset
    # category_columns = list(cfg.columns.feature_columns)
    # encoder = OrdinalEncoder(cols=category_columns, handle_unknown='impute').fit(df_train)
    # df_train_encoded = encoder.transform(df_train)

    if (not os.path.exists(cfg.pred_output_dir)):
        os.mkdir(cfg.pred_output_dir)
    # Start train

    if torch.cuda.is_available() and cfg.device != 'cpu':
        if cfg.device == 'cuda:0':
            device = torch.device('cuda:0')
        elif cfg.device == 'cuda:1':
            device = torch.device('cuda:1')
        else:
            device = torch.device('cuda')
        logging.info("Use GPU {}.".format(cfg.device))
    else:
        logging.info("Use CPU.")
        device = 'cpu'
    train(data_dict, data_test_dict, device, cfg)


if __name__ == '__main__':
    seed_torch(2)
    main()
