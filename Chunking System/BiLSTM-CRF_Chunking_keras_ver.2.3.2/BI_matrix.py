import os
import pickle
import pandas as pd
import numpy as np

import utils_saveNload
import config

IDX_PATH = config.IDX_PATH
CHR_PATH = config.CHR_PATH


def make_BI_mat():
    chunk_set = utils_saveNload.load_chunk_set()

    chunk_list = list(chunk_set)
    chunk_list.sort()  # B, B, B, ..., I, I, I
    chunk_list = sorted(chunk_list, key=lambda label: label[2:])
    print("chunk_list: ", chunk_list)
    print("number of chunks: ", len(chunk_list))

    len_chk = len(chunk_list)
    BI_array = np.zeros(len(chunk_list)**2).reshape(len_chk, len_chk)  # 없으면 NaN
    # df = pd.DataFrame(index=chunk_list, columns=chunk_list)
    BI_matrix = pd.DataFrame(BI_array, index=chunk_list, columns=chunk_list)

    true_sent = utils_saveNload.load_pickle_chr_ture()
    pred_sent = utils_saveNload.load_pickle_chr_pred()

    for t_sent, p_sent in zip(true_sent, pred_sent):
        for t, p in zip(t_sent, p_sent):
            BI_matrix.loc[t, p] += 1

    print(BI_matrix)
    utils_saveNload.save_BI_matrix(BI_matrix)


def load_BI_mat():
    BI_matrix = utils_saveNload.load_BI_matrix()
    print(BI_matrix)


if __name__ == "__main__":
    make_BI_mat()
    load_BI_mat()

