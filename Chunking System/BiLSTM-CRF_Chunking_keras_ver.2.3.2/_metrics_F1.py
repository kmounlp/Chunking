def precision(correct, actual):
    if actual == 0:
        return 0

    return correct / actual


def recall(correct, possible):
    if possible == 0:
        return 0

    return correct / possible


def f1(p, r):
    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)


def show_result(results):
    eval_schema_list = list(results.keys())
    for eval_schema in eval_schema_list:
        correct = results[eval_schema]['correct']
        actual = results[eval_schema]['actual']
        possible = results[eval_schema]['possible']

        PRECISION = precision(correct, actual)
        RECALL = recall(correct, possible)
        F1 = f1(PRECISION, RECALL)

        print("[ eval_schema ] :", eval_schema)
        print("precision: ", PRECISION)
        print("recall: ", RECALL)
        print("F1: ", F1, "\n")
