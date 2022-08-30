import sys
import os
import numpy as np
import time
from subprocess import call

def load_config(cfg_file):
    cfg = {}
    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split(':')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    return cfg_sample


def cfg_string(cfg):
    ks = sorted(cfg.keys())
    for kk in ["data_test_path", "data_train_path", "pred_output_dir", "summary_base_dir"]:
        if kk in ks:
            ks.remove(kk)
    cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    return cfg_str.lower()


def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)

    # cfg_str is used to avoid repeated training with the same hyper-parameters. No need to include model_name.
    kv_list = [i for i in cfg_str.strip().split(',') if not i.startswith("model_name")]
    cfg_str = ",".join(kv_list)

    used_cfgs = read_used_cfgs(used_cfg_file)

    return cfg_str in used_cfgs

def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            kv_list = [ i for i in l.strip().split(',') if not i.startswith("model_name") ]
            l = ",".join(kv_list)
            used_cfgs.add( l )

    return used_cfgs

def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)

def run(main_process, eval_process, cfg_file, num_runs, data_train_path=None, data_test_path=None):
    configs = load_config(cfg_file)

    outdir = configs['pred_output_dir'][0]
    used_cfg_file = '%s/used_configs.txt' % outdir

    if not os.path.isfile(used_cfg_file):
        f = open(used_cfg_file, 'w')
        f.close()

    for i in range(num_runs):
        cfg = sample_config(configs)

        if is_used_cfg(cfg, used_cfg_file):
            print('Configuration used, skipping')
            continue

        if not data_train_path is None:
            cfg["data_train_path"] = data_train_path
            cfg["data_test_path"] = data_test_path

        # Get the current time (timestamp)
        now = int(time.time())
        timeArray = time.localtime(now)
        otherStyleTime = time.strftime("%Y%m%d_%H%M%S", timeArray)
        if "share_dim" in cfg.keys():
            cfg["model_name"] = cfg["model_name"] + "{}_{}_{}".format(cfg["share_dim"], cfg["base_dim"], otherStyleTime)
        else:
            cfg["model_name"] = cfg["model_name"] + "{}_{}".format( cfg["base_dim"], otherStyleTime)

        if "sample_alpha" in cfg.keys() :
            cfg["model_name"] = cfg["model_name"] + "_alpha_{}".format(cfg["sample_alpha"])

        save_used_cfg(cfg, used_cfg_file)

        print('------------------------------')
        print('Run %d of %d:' % (i+1, num_runs))
        print('------------------------------')
        print('\n'.join(['%s: %s' % (str(k), str(v)) for k,v in cfg.items() if len(configs[k])>1]))

        flags = ' '.join('%s=%s' % (k,str(v)) for k,v in cfg.items())
        call('python {} {}'.format(main_process, flags), shell=True)

        # python
        # eval4real_data.py / Users / brice / Documents / git_resp / ESX / results / pred / ES_BN / Users / brice / Documents / dataset4uplift / lzd_real_data
        # ES_BN__20220111_00: 41:59
        # bin_prob_dataset_vn_v0 / real_bin_set_full
        # .1.test.npz
        # false

        # pred_output_dir: "/Users/brice/Documents/git_resp/ESX/results/pred/ES_BN"
        # data_train_path: "/Users/brice/Documents/dataset4uplift/lzd_real_data/bin_prob_dataset_vn_v0/real_bin_set_full.1.train.npz"
        # data_test_path: "/Users/brice/Documents/dataset4uplift/lzd_real_data/bin_prob_dataset_vn_v0/real_bin_set_full.1.test.npz"

        if eval_process == "eval4real_data.py":
            call('python {eval_process} {pred_output_dir} {data_test_path} {model_name} {if_early_stop} '.format(eval_process=eval_process, pred_output_dir=cfg["pred_output_dir"]
                                                                                                                 ,data_test_path=cfg["data_test_path"]
                                                                                                                 , model_name=cfg["model_name"]
                                                                                                                 , if_early_stop="false"), shell=True)
        elif eval_process == "eval.py":
            call('python {eval_process} {pred_output_dir} {data_train_path} {data_test_path} {model_name} {if_early_stop}'
                 .format(eval_process=eval_process, pred_output_dir=cfg["pred_output_dir"], data_train_path="NA", data_test_path=cfg['data_test_path'],
                         model_name=cfg["model_name"], if_early_stop="false"), shell=True)
        else:
            return


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python search_params.py <main_file> <eval_file> <config file> <num runs>')

    elif len(sys.argv) < 6:
        run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
