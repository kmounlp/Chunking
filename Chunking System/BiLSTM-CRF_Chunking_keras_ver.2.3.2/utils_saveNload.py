import os
import pickle
import pandas as pd

import config

CHR_PATH = "prediction_result"
IDX_PATH = "pickle"
MAT_PATH = "BI_matrix"
BI_PATH = "BI_correct"

TEST_SENT = os.path.join(CHR_PATH, "test_sentences.sent")
TRUE_SENT = os.path.join(CHR_PATH, "true_sentences.sent")
PRED_SENT = os.path.join(CHR_PATH, "pred_sentences.sent")
CHR_TEST = os.path.join(IDX_PATH, "ch_test_sent.pickle")
CHR_TRUE = os.path.join(IDX_PATH, "ch_true_sent.pickle")
CHR_PRED = os.path.join(IDX_PATH, "ch_pred_sent.pickle")
IDX_TEST = os.path.join(IDX_PATH, "idx_test_sent.pickle")
IDX_TRUE = os.path.join(IDX_PATH, "idx_true_sent.pickle")
IDX_PRED = os.path.join(IDX_PATH, "idx_pred_sent.pickle")
CHK_SET_PATH = os.path.join(IDX_PATH, "chunk_set.pickle")
BI_MAT_PATH = os.path.join(MAT_PATH, "BI_matrix")

# [ BI_error_corrector_2.py ]
BI_SENT = os.path.join(BI_PATH, 'BI_correct.sent')
BI_PCKL = os.path.join(BI_PATH, 'BI_correct.pickle')
BI_SENT_2 = os.path.join(BI_PATH, 'BI_correct_2.sent')
BI_PCKL_2 = os.path.join(BI_PATH, 'BI_correct_2.pickle')



def save_result(X_test, y_true, y_pred):
    with open(TEST_SENT, 'a', encoding='utf-8') as fopen:
        for sent in X_test:
            fopen.write(str(sent) + '\n')
    with open(TRUE_SENT, 'a', encoding='utf-8') as fopen:
        for sent in y_true:
            fopen.write(str(sent) + '\n')
    with open(PRED_SENT, 'a', encoding='utf-8') as fopen:
        for sent in y_pred:
            fopen.write(str(sent) + '\n')


#########################################################################
def save_as_pickle(X_test, y_true, y_pred):
    with open(CHR_TEST, 'wb') as f:
        pickle.dump(X_test,f)
    with open(CHR_TRUE, 'wb') as f:
        pickle.dump(y_true, f)
    with open(CHR_PRED, 'wb') as f:
        pickle.dump(y_pred, f)


def load_pickle_chr_ture():
    with open(CHR_TRUE, 'rb') as f:
        return pickle.load(f)


def load_pickle_chr_pred():
    with open(CHR_PRED, 'rb') as f:
        return pickle.load(f)


#########################################################################
def save_as_pickle_idx(X_test, y_true, y_pred):
    with open(IDX_TEST, 'wb') as f:
        pickle.dump(X_test, f)
    with open(IDX_TRUE, 'wb') as f:
        pickle.dump(y_true, f)
    with open(IDX_PRED, 'wb') as f:
        pickle.dump(y_pred, f)


#########################################################################
def save_chunk_set(chunk_set):
    with open(CHK_SET_PATH, 'wb') as f:
        pickle.dump(chunk_set, f)


def load_chunk_set():
    with open(CHK_SET_PATH, 'rb') as f:
        return pickle.load(f)


#########################################################################
def save_BI_matrix(BI_matrix):
    BI_matrix.to_csv(BI_MAT_PATH, sep='\t')


def load_BI_matrix():
    return pd.read_csv(BI_MAT_PATH, sep='\t')


#########################################################################
# [ BI_error_corrector_2.py ]
def save_BI_correct(chk_sent):
    with open(BI_SENT, 'a', encoding='utf-8') as f:
        f.write(str(chk_sent)+'\n')


def save_BI_correct_pickle(chk_sent):
    with open(BI_PCKL, 'wb') as f:
        pickle.dump(chk_sent, f)


def save_BI_correct_2(chk_sent):
    with open(BI_SENT_2, 'a', encoding='utf-8') as f:
        f.write(str(chk_sent)+'\n')


def save_BI_correct_2_pickle(chk_sent):
    with open(BI_PCKL_2, 'wb') as f:
        pickle.dump(chk_sent, f)


def load_BI_correct():
    with open(BI_PCKL, 'rb') as f:
        return pickle.load(f)


def load_BI_correct_2():
    with open(BI_PCKL_2, 'rb') as f:
        return pickle.load(f)


