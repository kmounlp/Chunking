"""
BiLSTM-CRF_Chunking_keras_ver.2.3.0
- to make chunk-based dependency corpus
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

from utils_data import input_data
from utils_model import padding, data_split
import _report
import _metrics
import metrics_new
import config
import utils_saveNload
import BI_matrix

RES_PATH = config.CHR_PATH

# :: hyper parameters ::
# [model parameters]
output_dim = 20
max_len = config.max_len

units = 50
recurrent_dropout = 0.1
activation = "relu"

# [compile parameters]
optimizer = "rmsprop"
# optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# [training  parameters]
batch_size = 32
epochs = 8
# valid_split = 0.1


def labeling_model():
    model = Sequential()
    model.add(Embedding(input_dim=n_words, output_dim=output_dim, input_length=max_len, mask_zero=True))
    model.add(Bidirectional(LSTM(units=units, return_sequences=True, recurrent_dropout=recurrent_dropout)))
    model.add(TimeDistributed(Dense(units, activation=activation)))
    print("\nn_labels:", n_labels)
    crf = CRF(n_labels)
    model.add(crf)

    model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    # plot_model(model, to_file="model.png")  # save model structure

    return model


def model_path():
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    # using checkpoint callback
    # checkpoint_path = "checkpoints/train_cp{epoch:02d}.ckpt"
    checkpoint_path = "checkpoints/train_0506.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    return checkpoint_path, checkpoint_dir


def model_train(model, X_train, y_train):  # , X_val, y_val
    # [save checkpoints]
    # ckpt_path, _ = model_path()
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1)
    # [save weights]
    # model.save_weights(ckpt_path)

    # train
    # history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
    #                     validation_split=valid_split, verbose=1)
    # history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
    #                     validation_data=(X_val, y_val))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # [save whole model]
    # if not os.path.isdir("saved_models"):
    #     os.mkdir("saved_models")
    # save in HDF5 format
    # model.save(r'saved_models/labeling_model.h5')

    return history


def model_test(model, X_test, y_test):
    # load latest checkpoint
    # load model

    # evaluate
    print("\n::Evaluate::")
    # print("X_test")
    # print(X_test)
    # print("y_test")
    # print(y_test)
    loss, acc = model.evaluate(X_test, y_test)
    print("\n Test accuracy: %.4f" % acc)


def trueNpred(X_test, y_test2, idx2wd, idx2chk):
    if not os.path.isdir(RES_PATH):
        os.mkdir(RES_PATH)

    # print("y_test2:", y_test2)

    # ver.2.3.0.
    # whole_sentences.sent -> whole_sentences_real.sent
    with open(os.path.join(RES_PATH, "whole_sentences_real.sent"), 'a', encoding='utf-8') as file:
        y_true_idx, y_pred_idx = [], []  # integer
        X_sent_list, y_true_list, y_pred_list = [], [], []  # character
        for idx, (X_test_sent, y_test2_sent) in enumerate(zip(X_test, y_test2), 1):
            y_predicted = model.predict(np.array([X_test_sent]))
            y_predicted = np.argmax(y_predicted, axis=-1)  # one-hot → integer
            pred = y_predicted[0]
            # print("y_test2_sent:", y_test2_sent)
            true = np.argmax(y_test2_sent, axis=-1)  # one-hot → integer

            # integer
            y_true_idx.append(true)
            y_pred_idx.append(pred)

            # in character
            X_sent = []
            for s in X_test_sent:
                if s != 0:  # 0: PAD
                    X_sent.append(idx2wd[s])
            # print("true:",true)
            # print("pred:",pred)
            y_true, y_pred = [], []

            for t, p in zip(true, pred):  # true, pred: one sentence
                if t != 0:  # 0: PAD
                    y_true.append(idx2chk[t])
                    y_pred.append(idx2chk[p])

            # save whole data in characters
            file.write(str(idx) + '\n')
            file.write(str(X_sent) + '\n')
            file.write(str(y_true) + '\n')
            file.write(str(y_pred) + '\n')
            file.write("\n")

            X_sent_list.append(X_sent)
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

    return y_true_idx, y_pred_idx, X_sent_list, y_true_list, y_pred_list


def metric_mat(model, X_test, y_test2, idx2wd, idx2chk):
    i = 13  # data index to check
    # print("*X_test[i]: ", X_test[i])  # 1D, integer array
    # print("np.array([X_test[i]]): ", np.array([X_test[i]]))  # 1D → 2D, integer array
    y_predicted = model.predict(np.array([X_test[i]]))  # integer → represent predicted vector into one-hot
    # print("y_predicted(model.predict): ", y_predicted)
    # print("y_predicted[0][0](one-hot Dimenion): ", y_predicted[0][0])
    # print("len of one-hot: ", len(y_predicted[0][0]))  # 34
    y_predicted = np.argmax(y_predicted, axis=-1)  # one-hot → integer(one-hot의 index)
    # print("y_predicted(np.argmax): ", y_predicted)
    #  np.argmax: axis에 해당하는 값들 중 가장 큰 값의 인덱스를 반환하는 함수
    # print("*y_predicted[0]: ", y_predicted[0])  # 2D → 1D

    true = np.argmax(y_test2[i], -1)  # one-hot → integer
    # print("\ny_test2[i]: ", y_test2[i])
    # print("*true: ", true)

    print("{:10}|{:7}|{:7}|{}".format("Word", "POS", "True", "Pred"))
    print(35 * "-")

    # all variables in 'zip' are 1D
    for wd, lbl, pred in zip(X_test[i], true, y_predicted[0]):
        if wd != 0:
            # word, pos = ft_model.wv.vocab[wd].split('/')
            word, pos = idx2wd[wd].split('/')
            print("{:10}{:7}: {:7} {}".format(word, pos, idx2chk[lbl], idx2chk[pred]))


# chunk metrics
def print_metrics(true, y_pred, chunk_set):
    _metrics.metrics(true, y_pred, chunk_set)


# BIO classification, token-based
def print_report(true, y_pred):
    print(_report.bio_classification_report(true, y_pred))


def show_loss(history):
    epochs = range(1, len(history.history['val_loss']) + 1)
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    # sentences, word_set, chunk_set = pre_processing()

    # assign index
    # wd2idx = word2index(word_set)
    # chk2idx = chunk2index(chunk_set)

    # integer embedding
    # data_X = word_emb(sentences, wd2idx)
    # data_X = word_emb_fasttext(sentences)
    # data_y = label_emb(sentences, chk2idx)

    # vars for models: n_words, n_labels
    # vars for test data: chk2idx, idx2wd, idx2chk
    data_X, data_y, wd2idx, chk2idx, n_words, idx2wd, idx2chk, n_labels = input_data()


    pad_X, pad_y = padding(data_X, data_y)
    # ver.2.3.0.
    X_train, X_test, X_real, y_train, y_test, y_real = data_split(pad_X, pad_y)

    # print(y_test)

    model = labeling_model()  # question? 임베딩 사이즈가 n_words?
    # history = model_train(model, X_train, y_train, X_val, y_val)
    # history = model_train(model, X_train, y_train)
    # show_loss(history)
    model_test(model, X_test, y_test)

    # metric_mat(model, X_test, y_test, idx2wd, idx2chk)
    # # print("X_test: ", X_test)
    # # print("y_test2: ", y_test)

    # 0619에 주석처리 함
    # saving prediction result
    y_true_idx, y_pred_idx, X_sent, y_true, y_pred = trueNpred(X_real, y_real, idx2wd, idx2chk)
    #
    # utils_saveNload.save_result(X_sent, y_true, y_pred)
    # utils_saveNload.save_as_pickle(X_sent, y_true, y_pred)
    # utils_saveNload.save_as_pickle_idx(X_test, y_true_idx, y_pred_idx)
    # print("saving finished")
    #
    # # metrics and report
    # print_metrics(y_true, y_pred, chk2idx.keys())  # chk2idx.keys() : chunk name set # check
    # print_report(y_true, y_pred)
    # metrics_new.metrics_new(y_true, y_pred)
    #
    # BI_matrix.make_BI_mat()
    # BI_matrix.load_BI_mat()

