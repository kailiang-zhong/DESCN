# coding=utf-8
import os
import numpy as np
from sklift.metrics import qini_auc_score


# 加载并解析
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

    data['dim'] = data['x'].shape[1]  # 特征维度
    data['n'] = data['x'].shape[0]  # 样本数

    return data


# 评估函数
def evaluate_bin(yf, t, tau_true, tau_pred):
    pehe = np.sqrt(np.mean(np.square(tau_pred - tau_true)))  # PEHE error

    ate_pred = np.mean(tau_pred)
    atc_pred = np.mean(tau_pred[(1 - t) > 0])
    att_pred = np.mean(tau_pred[t > 0])

    att = np.mean(tau_true[t > 0])
    ate = np.mean(tau_true)

    bias_att = np.abs(att_pred - att)  # the error of att
    bias_ate = np.abs(ate_pred - ate)  # the error of ate

    auuc_score = qini_auc_score(yf, tau_pred, t)

    return { "E_pehe": pehe, "E_att": bias_att, "E_ate": bias_ate, "AUUC": auuc_score }


def save_eval_result(result_str, result_file):
    with open(result_file, 'a') as f:
        f.write('%s\n' % result_str)


import sys

if __name__ == "__main__":
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    pred_output_dir = sys.argv[1]  # /home/admin/fengtong/ESX_Model/res/epilepsy4new/ESX
    data_train_path = sys.argv[2]
    data_test_path = sys.argv[3]
    model_name = sys.argv[4]

    # 是否需要根据validation 的loss 来选择某次预测结果，避免选择过拟合的预测结果。
    if_early_stop = "false"
    if len(sys.argv) > 4:
        if_early_stop = sys.argv[5]

    # trainset_result = "/home/admin/fengtong/ESX_Model/res/{}/{}/{}_train_result.test.npz".format(data_name,
    #                                                                                              ori_model_name,
    #                                                                                              model_name)
    # testset_result = "/home/admin/fengtong/ESX_Model/res/{}/{}/{}_test_result.test.npz".format(data_name,
    #                                                                                            ori_model_name,
    #                                                                                            model_name)

    trainset_result = "{}/{}_train_result.test.npz".format(pred_output_dir, model_name)
    testset_result = "{}/{}_test_result.test.npz".format(pred_output_dir, model_name)

    if data_train_path != "NA":
        dict_train_result = np.load(trainset_result)
        dict_train = load_data(data_train_path)

    dict_test = load_data(data_test_path)
    dict_test_result = np.load(testset_result)

    # save for predictions
    train_eval_result = { "E_pehe": [], "E_att": [], "E_ate": [], "AUUC": [] }
    test_eval_result = { "E_pehe": [], "E_att": [], "E_ate": [], "AUUC": [] }

    num_outputs = dict_test_result["p_tau"].shape[2]
    num_exps = dict_test_result["p_tau"].shape[1]

    # 模拟数据重复生成了多份数据，每一份数据对应一次重复实验
    for i_exp in range(num_exps):
        print("i_exp:{}/{}".format(i_exp + 1, num_exps))

        if data_train_path != "NA":
            # train
            X_train = dict_train["x"][:, :, i_exp]
            yf_train = dict_train["yf"][:, i_exp]
            t_train = dict_train["t"][:, i_exp]
            tau_train = dict_train["tau"][:, i_exp]

        # test
        X_test = dict_test["x"][:, :, i_exp]
        yf_test = dict_test["yf"][:, i_exp]
        t_test = dict_test["t"][:, i_exp]
        tau_test = dict_test["tau"][:, i_exp]

        # early stop
        ''' shape: [i_output, loss_list, i_exp] '''
        ''' 最后一个loss 是validation 的total loss，以此作为early stop 依据 '''
        # loss_valid_all = dict_train_result["loss"][:, -1, i_exp]
        # i_sel = np.argmin(loss_valid_all)
        # if not if_early_stop == "true":
        #     i_sel = num_outputs - 1
        # print("i_sel: {}".format(i_sel))

        train_res = { "E_pehe": 100, "E_att": 100, "E_ate": 100, "AUUC": 0 }
        test_res = { "E_pehe": 100, "E_att": 100, "E_ate": 100, "AUUC": 0 }
        sel_target = "E_pehe"
        # i_th output
        for i_output in range(num_outputs):

            if data_train_path != "NA":
                # training set
                p_tau = dict_train_result["p_tau"][:, i_exp, i_output]
                eval_result = evaluate_bin(yf_train, t_train, tau_train, p_tau)

                if eval_result[sel_target] < train_res[sel_target]:
                    for k in eval_result.keys():
                        train_res[k] = eval_result[k]

            # # if i_exp == 2:
            # #     print("train set,\tE_pehe:%.4f"%eval_result["E_pehe"] + ",\tE_att:%.4f"%eval_result["E_att"] + ",\tE_ate:%.4f"%eval_result["E_ate"])
            #
            # # the last prediction only.
            # if i_output == i_sel:
            #     train_eval_result["E_pehe"].append(eval_result["E_pehe"])
            #     train_eval_result["E_att"].append(eval_result["E_att"])
            #     train_eval_result["E_ate"].append(eval_result["E_ate"])
            #     train_eval_result["AUUC"].append(eval_result["AUUC"])

            # test set
            p_tau = dict_test_result["p_tau"][:, i_exp, i_output]
            eval_result = evaluate_bin(yf_test, t_test, tau_test, p_tau)
            if eval_result[sel_target] < test_res[sel_target]:
                for k in eval_result.keys():
                    test_res[k] = eval_result[k]

            # if i_exp == 2:
            #     print("test set,\tE_pehe:%.4f"%eval_result["E_pehe"] + ",\tE_att:%.4f"%eval_result["E_att"] + ",\tE_ate:%.4f"%eval_result["E_ate"])

            # the last prediction only.
            # if i_output == i_sel:
            #     test_eval_result["E_pehe"].append(eval_result["E_pehe"])
            #     test_eval_result["E_att"].append(eval_result["E_att"])
            #     test_eval_result["E_ate"].append(eval_result["E_ate"])
            #     test_eval_result["AUUC"].append(eval_result["AUUC"])

        for k in test_res.keys():
            test_eval_result[k].append( test_res[k] )
            train_eval_result[k].append( train_res[k])

    result_str = ""
    if data_train_path != "NA":
        result_str += "\n----train set----\n"
        print("----train set. split line ----")
        for k in train_eval_result.keys():
            val = np.mean(train_eval_result[k])
            std = np.std(train_eval_result[k]) / np.sqrt(num_exps)
            print(k + ": %.6f" % val + " +/- %.6f" % std)
            result_str += str(k) + ": " + str(round(val, 6)) + "+/- " + str(round(std, 6)) + "\n"

    res = ""
    result_str += "----test set----\n"
    print("----test set. split line ----")
    for k in train_eval_result.keys():
        val = np.mean(test_eval_result[k])
        std = np.std(test_eval_result[k]) / np.sqrt(num_exps)
        print(k + ": %.6f" % val + " +/- %.6f" % std)
        result_str += str(k) + ": " + str(round(val, 6)) + "+/- " + str(round(std, 6)) + "\n"
        if k in ["E_pehe", "E_ate"]:
            res += str(k) + " " + str(round(val, 6)) + " "

    save_eval_result("{} {} {}".format(model_name, if_early_stop, result_str),
                     "{}/eval_result.txt".format(pred_output_dir))

    save_eval_result("{} {} {}".format(model_name, if_early_stop, res),
                     "{}/eval_result_summary.txt".format(pred_output_dir))
    print("done.")



