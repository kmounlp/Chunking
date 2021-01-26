import pickle
from sklearn.metrics import multilabel_confusion_matrix

SENT_PATH = 'idx_test_sent.pickle'
TRUE_PATH = 'idx_true_sent.pickle'
PRED_PATH = 'idx_pred_sent.pickle'


def load_sentences():
    with open(SENT_PATH, 'rb') as f:
        X_test = pickle.load(f)
    with open(TRUE_PATH, 'rb') as f:
        y_true = pickle.load(f)
    with open(PRED_PATH, 'rb') as f:
        y_pred = pickle.load(f)

    return X_test, y_true, y_pred


def compare_length(X_test, y_true, y_pred):
    count = 0
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if not t == p:
            print(X_test[i])
            print(t)
            print(p)
            count += 1
    print(count)  # error sentence: 683


def confusion_mat(y_true, y_pred):
    print(multilabel_confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    X_test, y_true, y_pred = load_sentences()
    # compare_length(*load_sentences())
    confusion_mat(y_true, y_pred)
