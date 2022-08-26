# coding=utf-8
from typing import List, Tuple
import torch
from torch import nn, optim
import pandas as pd
import hydra
from pathlib import Path
from model.models import  BaseModel4MetaLearner
from model.dataset import ESXDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
import logging
import sys, os, shutil
import time
import random


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


def evalWithData(group_name, models, writer, step_or_epoch, cfg, x, yf, t, e, eff_tau=None, i_exp=None):
    logging.info("group_name:{}, evalWithData... -----------------------------------".format(group_name))
    writer_flag = not writer is None


    # set loss functions
    loss_fn = nn.BCELoss()  # for probability
    loss_mse = nn.MSELoss()
    loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit

    if cfg.use_ps:
        pscore = torch.sigmoid(models["propensity"](x))
    else:
        pscore = 0.5
    dhat_cs = models["tau_c"](x)
    dhat_ts = models["tau_t"](x)
    p_tau = pscore * dhat_cs + (1 - pscore) * dhat_ts

    # just for print log
    iter_name = "epoch"
    if i_exp == 0:
        iter_name = "train_step"
    # the p_tau of treatment group
    logging.info(
        "p_tau {}, {}, {} , mean(p_tau[t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                       torch.mean(p_tau[t.bool()]).item()))
    # the p_tau of control group
    logging.info(
        "p_tau {}, {}, {} , mean(p_tau[~t]) :{}".format(group_name, iter_name, step_or_epoch,
                                                        torch.mean(p_tau[~t.bool()]).item()))

    # order by a list
    loss_list=[]
    for name in models.keys():
        pred_logit = models[name](x)
        if name == "propensity":
            if not cfg.use_ps:
                continue
            loss = loss_with_logit_fn(pred_logit, t)
        elif name == "mu_c":
            loss = loss_with_logit_fn(pred_logit[~t.bool()], yf[~t.bool()])
        elif name == "mu_t":
            loss = loss_with_logit_fn(pred_logit[t.bool()], yf[t.bool()])
        elif name == "tau_c":
            target = torch.sigmoid(models["mu_t"](x)[~t.bool()]) - yf[~t.bool()]
            loss = loss_mse(pred_logit[~t.bool()], target)
        elif name == "tau_t":
            target = yf[t.bool()] - torch.sigmoid(models["mu_c"](x))[t.bool()]
            loss = loss_mse(pred_logit[t.bool()], target)
        else:
            loss = 0
        loss_list.append(loss)
        if writer_flag:
            writer.add_scalar("{}/{}_loss".format(group_name, name), loss, step_or_epoch)

    auuc_score = qini_auc_score(yf.reshape(-1).cpu().numpy(), p_tau.reshape(-1).cpu().numpy(),
                                t.reshape(-1).cpu().numpy())
    logging.info("group_name {}, {}, {}, auuc_score: {}".format(group_name, iter_name, step_or_epoch, auuc_score))
    if writer_flag:
        writer.add_scalar("{}/auuc_score".format(group_name), auuc_score, step_or_epoch)

        # mu_t
        pred_score = torch.sigmoid( models["mu_t"](x) )
        fpr, tpr, threshold = metrics.roc_curve(yf[t.bool()].cpu().detach().numpy(),
                                                pred_score[t.bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC, p_mut".format(group_name), roc_auc, step_or_epoch)

        # mu_c
        pred_score = torch.sigmoid(models["mu_c"](x) )
        fpr, tpr, threshold = metrics.roc_curve(yf[~t.bool()].cpu().detach().numpy(),
                                                pred_score[~t.bool()].cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC, p_muc".format(group_name), roc_auc, step_or_epoch)

        # propensity
        pred_score = torch.sigmoid(models["propensity"](x))
        fpr, tpr, threshold = metrics.roc_curve(t.cpu().detach().numpy(),
                                                pred_score.cpu().detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        writer.add_scalar("{}/:AUC, propensity".format(group_name), roc_auc, step_or_epoch)

        writer.flush()

    dict_result = {"loss": loss_list, "p_tau": p_tau}
    return dict_result



def weighted_rmse_loss(input, target, weight=1):
    risk = torch.sqrt(torch.mean(weight * torch.square(input - target)))
    return risk
    # return torch.mean(weight * ((input - target) ** 2) )


import math
def sample_imb_fn(x_, yf_, e_, t_, cfg):
    # treatment group size
    if "total_size" in cfg.keys():
        total_size = cfg["total_size"]
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

def train(data_dict, data_test_dict, device, cfg):
    # configs
    base_dim = cfg.base_dim
    batch_size = cfg.batch_size
    # 2*cfg.epochs for mu_t, mu_c and tau_t, tau_c
    epochs = 2*cfg.epochs
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

        ''' the whole train set, just use for prediction and convert to tensor dtype.'''
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

        ''' print the proportion information. '''
        if 0 == i_exp:
            logging.info("exp_{}, Train. x.shape : {}".format(i_exp, x_.shape))
            logging.info("exp_{}, Train. mean(t) : {}".format(i_exp, np.mean(t_)))
            logging.info("exp_{}, Train. mean(t) when e=1: {}".format(i_exp, np.mean(t_[e_.astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) : {}".format(i_exp, np.mean(yf_)))
            logging.info("exp_{}, Train. mean(yf) when t=1: {}".format(i_exp, np.mean(yf_[t_.astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) when t=0: {}".format(i_exp, np.mean(yf_[(1 - t_).astype(bool)])))
            logging.info("exp_{}, Train. mean(yf) when t=0 and e=1: {}".format(i_exp, np.mean(
                yf_[(e_ * (1 - t_)).astype(bool)])))
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

        ''' create graph 
        X learner
        https://arxiv.org/abs/1706.03461 
        '''
        model_name = ["propensity", "mu_t", "mu_c", "tau_t", "tau_c"]
        models = {}
        optimizers = {}
        lr_schedulers = {}
        for name in model_name:
            models[name] = BaseModel4MetaLearner(input_dim=dim, base_dim=base_dim, cfg=cfg, device=device)
            if cfg.optim == "SGD":
                optimizers[name] = optim.SGD(models[name].parameters(), lr=cfg.lr, weight_decay=cfg.l2)
            else:
                optimizers[name] = optim.Adam(models[name].parameters(), lr=cfg.lr, weight_decay=cfg.l2)
            lr_schedulers[name] = torch.optim.lr_scheduler.StepLR(optimizer=optimizers[name], step_size=cfg.decay_step_size,
                                                           gamma=cfg.decay_rate)


        ''' Build dataloader '''
        dataset = ESXDataset(x, yf, t, e)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Start fitting
        for name in models.keys():
            models[name].train()

        if (cfg.verbose):
            logging.info("exp_{} start trainning ...".format(i_exp))

        train_step = 0
        for epoch in range(epochs):
            if ((epoch + 1) % LOGSTEP == 0):
                logging.info("exp_i:{},  epoch:{} ...".format(i_exp, epoch))

            # for i, (inputs, t_labels, y_labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            for i, (inputs, t_labels, y_labels, e_labels) in enumerate(train_loader):
                if (y_labels.reshape(-1).shape[0] < batch_size):
                    continue

                # logging.debug("epoch:{} .shape :{}".format(epoch, inputs.shape))
                inputs.to(device)
                t_labels = torch.unsqueeze(t_labels.to(device), 1)
                y_labels = torch.unsqueeze(y_labels.to(device), 1)
                e_labels = torch.unsqueeze(e_labels.to(device), 1) # e_ Labels is used to mark whether it is a random sample

                # set loss functions
                loss_fn = nn.BCELoss()  # for probability
                loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
                loss_mse = nn.MSELoss()

                # The odd number of epoch is used for training mu_c,mu_t,propensity
                # Even epochs are used for training tau_c, tau_t
                if (epoch+1)%2 != 0:
                    train_list = ["mu_c", "mu_t", "propensity"]
                else:
                    train_list = ["tau_c", "tau_t"]
                # order by a list
                for name in train_list:
                    models[name].train()
                    optimizers[name].zero_grad()

                    pred_logit = models[name](inputs)
                    if name == "propensity":
                        if not cfg.use_ps:
                            continue
                        loss = loss_with_logit_fn( pred_logit, t_labels)

                    elif name == "mu_c":
                        loss = loss_with_logit_fn(pred_logit[~t_labels.bool()], y_labels[~t_labels.bool()])
                    elif name == "mu_t":
                        loss = loss_with_logit_fn(pred_logit[t_labels.bool()], y_labels[t_labels.bool()])
                    elif name == "tau_c":
                        target = torch.sigmoid( models["mu_t"](inputs) )[~t_labels.bool()] - y_labels[~t_labels.bool()]
                        loss = loss_mse(pred_logit[~t_labels.bool()], target)
                    elif name == "tau_t":
                        target = y_labels[t_labels.bool()] - torch.sigmoid( models["mu_c"](inputs) )[t_labels.bool()]
                        loss = loss_mse(pred_logit[t_labels.bool()], target)

                    if i_exp == 0 and (train_step + 1) % LOGSTEP == 0 and cfg.verbose:
                        logging.info("epoch:{}, model:{}, loss:{}".format(epoch, name, loss))
                    # Backpropagation
                    loss.backward()
                    # Update parameters
                    optimizers[name].step()

                # the first experiment only
                if i_exp == 0 and (train_step + 1) % LOGSTEP == 0 and cfg.verbose:
                    for name in models.keys():
                        models[name].eval()
                    with torch.no_grad():
                        # validation
                        evalWithData("valid_set", models, writer, train_step, cfg, x_valid, yf_valid,
                                     t_valid,
                                     e_valid, eff_valid, i_exp
                                     )
                        # test
                        evalWithData("test_set", models, writer, train_step, cfg, x_test, yf_test, t_test,
                                     e_test, eff_test, i_exp
                                     )
                        # train
                        evalWithData("train_set ", models, writer, train_step, cfg, x_train, yf_train,
                                     t_train,
                                     e_train, eff_train, i_exp
                                     )

                ''' end loop for a epoch '''
                train_step = train_step + 1
            # update the learning rate
            for name in train_list:
                lr_schedulers[name].step()
                # get learning rate
                new_lr = lr_schedulers[name].get_last_lr()
                logging.info(
                    "i_exp:{}, name:{}, epoch:{}, new learning rate is: {}".format(i_exp, name, epoch, new_lr))

            ''' end loop for epochs '''

            ''' predict every VALSTEP step and get results for each experiments'''
            if ((epoch + 1)/2) % PREDSTEP == 0:
                for name in models.keys():
                    models[name].eval()
                # start to evel
                logging.info(f'start to predict ... i_exp:{i_exp},epochs:{epoch}, train_step:{train_step}')
                with torch.no_grad():
                    # for test
                    dict_result = evalWithData("test_pred_result", models, None, epoch, cfg, x_test, yf_test,
                                               t_test,
                                               e_test, eff_test
                                               )
                    # append to list. for saving.
                    iexp_p_tau["test"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                    iexp_losses["test"].append(dict_result["loss"])

                    if cfg.verbose > 0:
                        # for the whole training set
                        dict_result = evalWithData("train_pred_result", models, None, epoch, cfg, x_train, yf_train,
                                                   t_train,
                                                   e_train, eff_train
                                                   )
                        # append to list. for saving.
                        iexp_p_tau["train"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                        train_total_loss = dict_result["loss"]  # loss for the whole training set

                        # for validation
                        # dict_result = evalWithData("valid_pred_result", models, None, epoch, cfg, x_valid, yf_valid,
                        #                            t_valid,
                        #                            e_valid, eff_valid
                        #                            )
                        # # # append to list. for saving.
                        # # iexp_p_prpsy["valid"].append(dict_result["p_prpsy"].cpu().detach().numpy()[:, 0])
                        # # iexp_p_yf["valid"].append(dict_result["p_yf"].cpu().detach().numpy()[:, 0])
                        # # iexp_p_ycf["valid"].append(dict_result["p_ycf"].cpu().detach().numpy()[:, 0])
                        # # iexp_p_tau["valid"].append(dict_result["p_tau"].cpu().detach().numpy()[:, 0])
                        # iexp_losses["train"].append(train_total_loss + [dict_result["loss"][0]])  # append validation的loss

        # only save the model for the first experiment.
        # if i_exp == 0 and cfg.verbose:
        #     logging.info("exp_{} model saving...".format(i_exp))
        #     torch.save(models, "./{}_p{}.pth".format(cfg.model_name, i_exp))
        #     logging.info("exp_{} model saving...done.".format(i_exp))
        #     writer.close()

        if cfg.verbose > 0:
            group_list = ["train", "test"]
        else:
            group_list = ["test"]
        ''' save preidctions '''
        for group in group_list:
            # {"p_prpsy":[], "p_yf":[], "p_ycf":[], "p_tau":[], "loss":[]}
            result_dict[group]["p_tau"].append(iexp_p_tau[group])
            result_dict[group]["loss"].append(iexp_losses[group])
            if group == "train":
                result_dict[group]["val"].append(I_valid)
        ''' Format the prediction results and loss of ["train", "valid", "test"] data set and save them locally '''
        for group in group_list:
            '''units, exp_i, outputs'''
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
            np.savez(npz_file_path, p_tau=all_p_tau, loss=all_losses, val=all_I_valid)
            logging.info("saving predict result as a file: {}...done".format(npz_file_path))

# ./conf just contain a configuration template as default for configuration, which needs to be overwritten(key-value wise) by the hyper-parameters in ./conf4models
@hydra.main(config_path='./conf', config_name='conf_lzd_real_x_learner.yaml')
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
