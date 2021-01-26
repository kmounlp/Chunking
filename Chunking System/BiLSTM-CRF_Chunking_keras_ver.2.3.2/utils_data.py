import re
import matplotlib.pyplot as plt

import utils_saveNload
import config

# Chunked corpus in CoNLL form
DATA_PATH = config.CORPUS_PATH


def read_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as fopen:
        return fopen.readlines()


def pre_processing():
    """
    sentences = [[('Morph/POS', 'LBL), (), ..., ()], ..., []]
    :return sentences, word_set, chunk_set:
    """
    data = read_data()

    sentences = []
    sentence = []
    word_set = set()
    chunk_set = set()
    for line in data:
        if re.match(r'#', line):  # excluding sentence information
            continue

        line = line.strip()
        if line:
            line = line.split()
            # find the way to make use of pandas
            morph = line[1]
            pos = line[2]
            chk_lbl = line[4]  # chunk label
            word = morph + '/' + pos

            sentence.append((word, chk_lbl))
            word_set.add(word)
            chunk_set.add(chk_lbl)
        else:  # end of a sentence
            sentences.append(sentence)
            sentence = []

    return sentences, word_set, chunk_set


def show_length(sentences):
    """
    Show the lengths of sentences in corpus in a graph.
    To determine the max length (for padding)
    :param sentences: whole sentences in the data set
        <ex> [[('Morph/POS', 'LBL'), (), ... , ()], ..., []]
    """
    plt.hist([len(sent) for sent in sentences], bins=50)
    plt.xlabel('length of Data')
    plt.ylabel('number of Data')
    plt.show()
    # print(max(len(sent) for sent in sentences))  #  len(the longest one) = 108


# can change into other embedding models
def word2index(word_set):
    wd2idx = {wd: i for i, wd in enumerate(word_set, 2)}
    wd2idx['PAD'] = 0
    wd2idx['OOV'] = 1
    return wd2idx


def chunk2index(chunk_set):
    chk2idx = {chk: i for i, chk in enumerate(chunk_set, 1)}
    chk2idx['PAD'] = 0
    return chk2idx


def index2word(wd2idx):
    idx2wd = {}
    for key, value in wd2idx.items():
        idx2wd[value] = key
    return idx2wd


def index2chunk(chk2idx):
    idx2chk = {}
    for key, value in chk2idx.items():
        idx2chk[value] = key
    return idx2chk


def word_emb(sentences, wd2idx):
    data_X = []
    for sent in sentences:
        temp_X = []
        for wd, lbl in sent:
            temp_X.append(wd2idx.get(wd, 1))  # OOV: 1
        data_X.append(temp_X)
    return data_X

# def _load_fasttext():
#     fname = r"./embeddings/fasttext/morph_ft.model"
#     ft_model = FastText.load(fname)
#
#     return ft_model
#
#
# def word_emb_fasttext(sentences):
#     ft_wd_emb = _load_fasttext()
#
#     data_X = []
#     for sent in sentences:
#         temp_X = []
#         for wd, lbl in sent:
#             try:
#                 temp_X.append(ft_wd_emb.wv[wd])
#             except KeyError:
#                 temp_X.append('1')  # OOV: 1
#         data_X.append(temp_X)
#     return data_X


def label_emb(sentences, chk2idx):
    data_y = []
    for sent in sentences:
        temp_y = []
        for wd, lbl in sent:
            temp_y.append(chk2idx.get(lbl, 1))  # OOV: 1
        data_y.append(temp_y)
    return data_y


def input_data():
    sentences, word_set, chunk_set = pre_processing()

    wd2idx = word2index(word_set)
    idx2wd = index2word(wd2idx)
    n_words = len(wd2idx)  # vocab size

    chk2idx = chunk2index(chunk_set)
    idx2chk = index2chunk(chk2idx)
    n_labels = len(chk2idx)

    data_X = word_emb(sentences, wd2idx)
    data_y = label_emb(sentences, chk2idx)

    utils_saveNload.save_chunk_set(chunk_set)

    return data_X, data_y, wd2idx, chk2idx, n_words, idx2wd, idx2chk, n_labels


if __name__ == "__main__":
    sentences, word_set, chunk_set = pre_processing()
    print(chunk_set)
    print(len(chunk_set))

    # utils_save_load.save_chunk_set(chunk_set)
    # wd2idx = word2index(word_set)
    # chk2idx = chunk2index(chunk_set)