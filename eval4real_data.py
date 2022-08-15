# coding=utf-8
import os
import numpy as np
from sklift.metrics import qini_auc_score


# Load and parse
def load_data(file_path):
    """ Load data set """
    data_in = np.load(file_path)
    data = { 'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf'] }
    try:
        data['ycf'] = data_in['ycf']
        data["mu0"] = data_in['mu0']
        data["mu1"] = data_in['mu1']
    except:
        data['ycf'] = None

    try:
        data['e'] = data_in['e']
    except:
        data['e'] = np.ones_like(data_in['yf'])

    try:
        data['tau'] = data_in['tau']
        data['IS_SYNT'] = True
    except:
        data['tau'] = np.array([None])
        data['IS_SYNT'] = False

    data['dim'] = data['x'].shape[1]  # Feature dimension
    data['n'] = data['x'].shape[0]  # Number of samples

    return data


def save_eval_result(result_str, result_file):
    with open(result_file, 'a') as f:
        f.write('%s\n' % result_str)


def evaluate_bin(t, tau_true, tau_pred):
    pehe = np.sqrt(np.mean(np.square(tau_pred - tau_true)))  # PEHE error

    ate_pred = np.mean(tau_pred)
    atc_pred = np.mean(tau_pred[(1 - t) > 0])
    att_pred = np.mean(tau_pred[t > 0])

    att = np.mean(tau_true[t > 0])
    ate = np.mean(tau_true)

    bias_att = np.abs(att_pred - att)  # the error of att
    bias_ate = np.abs(ate_pred - ate)  # the error of ate

    return { "E_pehe": pehe, "E_att": bias_att, "E_ate": bias_ate }


import sys

if __name__ == "__main__":
    # python
    # eval4real_data.py
    # { pred_output_dir }
    # { data_test_path }
    # { model_name }
    # { if_early_stop }

    pred_output_dir = sys.argv[1]
    data_test_path = sys.argv[2]
    model_name = sys.argv[3]
    # Whether it is necessary to select a prediction result according to the loss of validation to avoid selecting the prediction result of over fitting
    if_early_stop = sys.argv[4]

    # trainset_result = "{}/{}_train_result.test.npz".format(pred_base_dir, model_name)
    testset_result = "{}/{}_test_result.test.npz".format(pred_output_dir, model_name)
    # trainset_result = "{}/{}_train_result.test.npz".format(pred_output_dir, model_name)
    # data_train_path = "{}/{}/real_bin_set.10.train.npz".format(dataset_base_dir, data_name)
    # data_test_path = "{}/{}/real_bin_set.5.test.npz".format(dataset_base_dir, data_name)

    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # dict_train_result = np.load(trainset_result)
    dict_test_result = np.load(testset_result)

    # dict_train = load_data(data_train_path)
    dict_test = load_data(data_test_path)

    # save for predictions
    test_eval_result = { "AUUC": [], "E_att": [] }

    num_outputs = dict_test_result["p_tau"].shape[2]
    num_exps = dict_test_result["p_tau"].shape[1]

    for i_exp in range(num_exps):
        print("i_exp:{}/{}".format(i_exp + 1, num_exps))
        # # train
        # X_train = dict_train["x"][:, :, i_exp]
        # yf_train = dict_train["yf"][:, i_exp]
        # t_train = dict_train["t"][:, i_exp]
        # tau_train = dict_train["tau"][:, i_exp]

        # test
        X_test = dict_test["x"][:, :, i_exp]
        yf_test = dict_test["yf"][:, i_exp]
        t_test = dict_test["t"][:, i_exp]

        # early stop
        ''' shape: [i_output, loss_list, i_exp] '''
        # loss_valid_all = dict_train_result["loss"][:, -1, i_exp]
        # i_sel = np.argmin(loss_valid_all)
        # if not if_early_stop == "true":
        #     i_sel = num_outputs - 1
        i_sel = num_outputs - 1
        print("i_sel: {}".format(i_sel))

        att = np.mean(yf_test.reshape(-1)[t_test.reshape(-1) == 1]) - np.mean(
            yf_test.reshape(-1)[t_test.reshape(-1) == 0])
        print("i_exp:{}, att:{}".format(i_exp, att))

        # i_th output
        tmp_auuc_score = 0
        tmp_Eatt = 1
        for i_output in range(num_outputs):
            # test set
            p_tau = dict_test_result["p_tau"][:, i_exp, i_output]

            auuc_score = qini_auc_score(yf_test.reshape(-1), p_tau.reshape(-1), t_test.reshape(-1))

            pred_att = np.mean(p_tau.reshape(-1)[t_test.reshape(-1) == 1])

            E_att = np.abs(pred_att - att)

            # if i_exp == 2:
            #     print("test set,\tE_pehe:%.4f"%eval_result["E_pehe"] + ",\tE_att:%.4f"%eval_result["E_att"] + ",\tE_ate:%.4f"%eval_result["E_ate"])

            print("i_exp:{}, AUUC:{}".format(i_exp, auuc_score))
            if auuc_score > tmp_auuc_score:
                tmp_auuc_score = auuc_score
                tmp_Eatt = E_att

            # the last prediction only.
        #             if i_output == i_sel:
        #                 test_eval_result["AUUC"].append( auuc_score )
        #                 test_eval_result["E_att"].append(E_att)
        # the last prediction only.
        test_eval_result["AUUC"].append(tmp_auuc_score)
        test_eval_result["E_att"].append(tmp_Eatt)

    print(
        "--------------------------------------------test set. split line --------------------------------------------")
    print(test_eval_result)
    result_str_list = []
    for k in test_eval_result.keys():
        val = np.mean(test_eval_result[k])
        std = np.std(test_eval_result[k]) / np.sqrt(num_exps)
        result_str = k + ": %.6f" % val + " +/- %.6f" % std
        result_str_list.append(result_str)
        print(result_str)

    save_eval_result("{},{}".format(model_name, ",".join(result_str_list).lower()),
                     "{}/eval_result.txt".format(pred_output_dir))

    print("done.")


