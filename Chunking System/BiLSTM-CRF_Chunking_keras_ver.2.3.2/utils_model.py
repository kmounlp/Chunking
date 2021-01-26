from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import config

max_len = config.max_len


def padding(data_X, data_y):
    pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
    pad_y = pad_sequences(data_y, padding='post', value=0, maxlen=max_len)  # 0: chk2idx['PAD']

    return pad_X, pad_y


def count_morphemes(X_data):
    """
    count the number of morphemes
    :param X_data:
    :return:
    """
    count = 0
    for sent in X_data:
        for morph in sent:
            if morph != 0:  #? why 0?
                count += 1
    return count


def data_split(pad_X, pad_y):
    # X_train, X_remain, y_train, y_remain = train_test_split(pad_X, pad_y, train_size=.8, random_state=0)
    # X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.2, random_state=777)

    # ver.2.3.0.
    X_train, X_test, y_train, y_test = train_test_split(pad_X[:13113], pad_y[:13113], train_size=.8, random_state=0)
    X_real, y_real = pad_X[13113:], pad_y[13113:]

    # Count the number of sentences and morphemes
    print("[ Train ]")
    print("number of sentences: ", len(X_train))
    count_train = count_morphemes(X_train)
    print("number of morphemes: ", count_train)

    print("[ Test ]")
    print("number of sentences: ", len(X_test))
    count_test = count_morphemes(X_test)
    print("number of morphemes: ", count_test)

    print("[ Real ]")
    print("number of sentences: ", len(X_real))
    count_real = count_morphemes(X_real)
    print("number of morphemes: ", count_real)

    # print("[ Validation ]")
    # print("number of sentences: ", len(X_val))
    # count_test = 0
    # for sent in X_val:
    #     for morph in sent:
    #         if morph != 0:
    #             count_test += 1
    # print("number of morphemes: ", count_test)

    # transform integer to binary class matrix (one-hot)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_real = np_utils.to_categorical(y_real)
    # y_val = np_utils.to_categorical(y_val)

    return X_train, X_test, X_real, y_train, y_test, y_real # X_val, y_val
