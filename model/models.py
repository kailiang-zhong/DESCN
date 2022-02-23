import torch
import torch.nn as nn
import math
import sys


def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y=torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, cfg, device):
        super(ShareNetwork, self).__init__()
        if cfg.BatchNorm1d == 'true':
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(

                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=cfg.do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.cfg.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, cfg, device):
        super(BaseModel4MetaLearner, self).__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=cfg.do_rate),
            # nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            nn.ELU(),
            nn.Dropout(p=cfg.do_rate),
            nn.Linear(base_dim, 1),
            # nn.ELU()
            # nn.BatchNorm1d(base_dim),
        )
        self.DNN.apply(init_weights)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """进行propensity预测的network"""

    def __init__(self, base_dim, cfg):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    """进行 mu0 预测的network"""
    def __init__(self, base_dim, cfg):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    """进行 mu1 预测的network"""

    def __init__(self, base_dim, cfg):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """进行 tau 预测的network"""

    def __init__(self, base_dim, cfg):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit

class ESX(nn.Module):
    """ESX"""
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork, cfg, device):
        super(ESX, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)
        self.cfg = cfg
        self.device = device
        self.to(device)

    def forward(self, inputs):
        """ t: just used for CFR(with wass or mmd distance) """
        # h = self.feature_extractor(inputs)  # encode
        shared_h = self.shareNetwork(inputs)

        # Predict propensity
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)

        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # Predict  mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # Predict tau
        tau_logit = self.tau_network(shared_h)

        #ESX_with_tau_prob, ESX_with_tau_logit, ES_only
        #graph_type: "ESX_with_tau_prob_net"
        # if self.cfg.graph_type == "ESX_with_tau_prob":
        #     p_tau = sigmod2(tau_logit)
        #     # p_tau = torch.tanh(tau_logit)
        #     p_mu1 = sigmod2(mu1_logit)
        #     p_mu0 = sigmod2(mu0_logit)
        #     p_h1 = torch.clip(p_mu0 + p_tau, 0.0001, 0.9999)
        #     p_h0 = torch.clip(p_mu1 - p_tau, 0.0001, 0.9999)
        # elif self.cfg.graph_type == "ESX_with_tau_logit":
        #     p_mu1 = sigmod2(mu1_logit)
        #     p_mu0 = sigmod2(mu0_logit)
        #     p_h1 = sigmod2(mu0_logit + tau_logit)
        #     p_h0 = sigmod2(mu1_logit - tau_logit)
        # elif self.cfg.graph_type == "ES_only":
        #     # as default: tau = h1-h0
        #     p_mu1 = sigmod2(mu1_logit)
        #     p_mu0 = sigmod2(mu0_logit)
        #     p_h1 = p_mu1
        #     p_h0 = p_mu0
        # else:
        #     print("error for self.cfg.graph_type:{}".format(self.cfg.graph_type))


        # as default: tau = h1-h0
        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1
        p_h0 = p_mu0


        # Predict entire space probability
        p_escvr1 = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escvr0 = torch.mul(p_i_prpsy, p_h0)

        return p_prpsy_logit, p_escvr1, p_escvr0, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h

    # def forward(self, inputs):
    #     inputs = self.shareNetwork(inputs)
    #     # Predict  mu1, mu0
    #     mu1_logit = self.mu1_network(inputs)
    #     mu1 = sigmod2(mu1_logit)
    #     return mu1_logit, mu1, mu1, mu1/2, mu1, mu1_logit, mu1, mu1, mu1
