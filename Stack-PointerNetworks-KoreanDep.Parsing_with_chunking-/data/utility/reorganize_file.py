"""
This file is for reorganize chunk and sejong file which are deleted partially.
2019-11-09
"""
import sys
import re
import pickle

CK_TRN = "../chunk/train.conllc"
CK_DEV = "../chunk/dev.conllc"
CK_TST = "../chunk/test.conllc"

SJ_TRN = "../sejong/train.conllu"
SJ_DEV = "../sejong/dev.conllu"
SJ_TST = "../sejong/test.conllu"

CK_ORG = "../chunk_dep_corpus(10_columns).conllc"
SJ_ORG = "../sejong_dep_corpus(sync).conllu"

ck_trn_sv = open(CK_TRN, 'w', encoding='utf-8')
ck_dev_sv = open(CK_DEV, 'w', encoding='utf-8')
ck_tst_sv = open(CK_TST, 'w', encoding='utf-8')

sj_trn_sv = open(SJ_TRN, 'w', encoding='utf-8')
sj_dev_sv = open(SJ_DEV, 'w', encoding='utf-8')
sj_tst_sv = open(SJ_TST, 'w', encoding='utf-8')

ck_org_file = open(CK_ORG, 'r', encoding='utf-8')
sj_org_file = open(SJ_ORG, 'r', encoding='utf-8')

# ck_data_sv = open("../chunk_data.conllc", 'w', encoding='utf-8')
# sj_data_sv = open("../sejong_data.conllu", 'w', encoding='utf-8')
ck_data_rd = open("../chunk_data.conllc", 'r', encoding='utf-8')
sj_data_rd = open("../sejong_data.conllu", 'r', encoding='utf-8')

ck_input_sv = open("../chunk_input.conllc", 'w', encoding='utf-8')
sj_input_sv = open("../sejong_input.conllu", 'w', encoding='utf-8')
ck_input_rd = open("../chunk_input.conllc", 'r', encoding='utf-8')
sj_input_rd = open("../sejong_input.conllu", 'r', encoding='utf-8')


def count_sents(file_name):
    file_name.seek(0)
    count = 0
    for line in file_name:
        line = line.strip()
        if not line:
            count += 1
    return count


def mk_sents_with_lemma(file_name, idx_lemma):  # lemma: index of lemma
    """
    make sentence with lemma column and return a list of the sentences.
    :param file_name:
    :param idx_lemma:
    :return:
    """
    file_name.seek(0)
    file_sents = []
    sentence = []
    count = 1

    for line in file_name:
        line = line.strip()
        if line:
            line = line.split('\t')
            lemma = line[idx_lemma].split()
            sentence.extend(lemma)
        else:
            count += 1
            file_sents.append(sentence)
            sentence = []

    return file_sents


def mk_sents_with_file(file_name, file_type):
    """
    make list with original sentences.
    :param file_name:
    :param file_type:
    :return original sentence list:
    """
    file_name.seek(0)
    file_sents = []
    if file_type == 'ck':
        split_sbl = '='
        sent_sbl = "# text"
    elif file_type == 'sj':
        split_sbl = ':'
        sent_sbl = "#ORGSENT"
    else:
        split_sbl = None
        sent_sbl = None
        sys.stderr.write("check 'file_type'")

    for line in file_name:
        line = line.strip()
        if line:
            line = line.split(split_sbl, 1)
            if line[0].strip() == sent_sbl:
                file_sents.append(line[1].strip())
    return file_sents


def sents_sync(ck_sents, sj_sents):
    """
    check if two sentences are same.
    if there're differences, return the list to banish.
    :param ck_sents:
    :param sj_sents:
    :return:
    """
    # ck_sents = mk_sents_with_lemma(ck_file, 3)
    # sj_sents = mk_sents_with_lemma(sj_file, 2)

    # ck_sents = mk_sents_with_file(ck_org_file, 'ck')
    # sj_sents = mk_sents_with_file(sj_org_file, 'sj')

    out_list = []
    for i, (ck_s, sj_s) in enumerate(zip(ck_sents, sj_sents), 1):
        if ck_s == sj_s: pass
        else:
            out_list.append(i)
    return out_list

    # ck_out_list = [i for i, sent in enumerate(ck_sents, 1) if sent not in sj_sents]
    # sj_out_list = [i for i, sent in enumerate(sj_sents, 1) if sent not in ck_sents]
    #
    # print("num of ck_out: ", len(ck_out_list))
    # print("num of sj_out: ", len(sj_out_list))
    # print(ck_out_list)
    # print(sj_out_list)
    #
    # return ck_out_list, sj_out_list


def mk_list(file_name):
    file_name.seek(0)
    idx_list = []
    head_list = []
    id_lst = []
    hd_lst = []
    for line in file_name:
        line = line.strip()
        if line:
            if line[0] == "#" or re.match('ID', line):
                continue
            line = line.split('\t')
            idx = line[0].split()
            head = line[7].split()
            id_lst.extend(idx)
            hd_lst.extend(head)
        else:
            # print(id_lst)
            # print(hd_lst)
            idx_list.append(id_lst)
            head_list.append(hd_lst)
            id_lst = []
            hd_lst = []
    return idx_list, head_list


def check_error(ck_id_list, ck_head_list):
    """
    check if head is not in the id list to remove the sentence which involves this case.
    :param ck_id_list:
    :param ck_head_list:
    :return:
    """
    error_sent = []
    for i, (ck_id_sent, ck_head_sent) in enumerate(zip(ck_id_list, ck_head_list), 1):
        if len(ck_id_sent) == 1:  # check short sentence error
            error_sent.append(i)
            continue
        elif len(ck_id_sent) != len(ck_head_sent):  # check length difference error
            input("sentence length different")
            error_sent.append(i)  # there's no error in this type
        else:
            for hd in ck_head_sent:  # check no head error
                if hd == '0': continue
                elif hd not in ck_id_sent:
                    error_sent.append(i)
                    pass
                else: continue
                break
    return error_sent


def purify_input_file(file_name, save_file, error_list_file):
    """
    remove sentences based on error lists.
    remove information part.
    :param file_name:
    :param save_file:
    :param error_list_file:
    :return:
    """
    error_list = read_error_list(error_list_file)
    file_name.seek(0)
    count = 1
    for line in file_name:
        line = line.strip()
        if line:
            if error_list and count == error_list[0]: continue
            if line[0] == "#" or re.match('ID', line): pass
            else:
                print(line, file=save_file)
        else:
            if error_list and count == error_list[0]: del error_list[0]
            else:
                print(line, file=save_file)
            count += 1
    save_file.close()


def save_error_list(error_list, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(error_list, file)


def read_error_list(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def data_split(file_name, train, dev, test):
    file_name.seek(0)
    save_file = train
    count = 1
    for line in file_name:
        line = line.strip()
        print(line, file=save_file)
        if not line:
            if count == 33384:
                save_file = dev
            elif count == 37556:
                save_file = test
            count += 1


if __name__ == "__main__":
    # print(count_sents(CK_TRN))  # 34857
    # print(count_sents(CK_DEV))  # 4784
    # print(count_sents(CK_TST))  # 4455
    #
    # print(count_sents(SJ_TRN))  # 37259
    # print(count_sents(SJ_DEV))  # 4785
    # print(count_sents(SJ_TST))  # 4467
    #
    # sents_sync(CK_TRN, SJ_TRN)
    # sents_sync(CK_DEV, SJ_DEV)
    # sents_sync(CK_TST, SJ_TST)
    ################################################
    # print(count_sents(ck_org_file))  # 49292
    # print(count_sents(sj_org_file))  # 49292

    # ck_sents = mk_sents_with_file(ck_org_file, 'ck')
    # sj_sents = mk_sents_with_file(sj_org_file, 'sj')
    # sents_sync(ck_sents, sj_sents)  # all sentences are same!

    # # find the id of weird sentences and delete with sj simultaneously
    # ck_id_list, ck_head_list = mk_list(ck_org_file)  # for check head error
    # error_list = check_error(ck_id_list, ck_head_list)  # head error: 1564, shortage error:5964, all: 7528
    # save_error_list(error_list, "error_list.txt")
    #
    # purify_input_file(ck_org_file, ck_data_sv, "error_list.txt")
    # purify_input_file(sj_org_file, sj_data_sv, "error_list.txt")

    # print(count_sents(ck_data_rd))  # 41764
    # print(count_sents(sj_data_rd))  # 41764

    # # chunk_data
    # ck_sents = mk_sents_with_lemma(ck_data_rd, 3)
    # sj_sents = mk_sents_with_lemma(sj_data_rd, 2)
    # out_list = sents_sync(ck_sents, sj_sents)  # chopped error: 36
    # save_error_list(out_list, "out_list.txt")
    #
    purify_input_file(ck_data_rd, ck_input_sv, "out_list.txt")
    purify_input_file(sj_data_rd, sj_input_sv, "out_list.txt")

    print(count_sents(ck_input_rd))  # 41728
    print(count_sents(sj_input_rd))  # 41728

    # split file

    data_split(ck_input_rd, ck_trn_sv, ck_dev_sv, ck_tst_sv)
    data_split(sj_input_rd, sj_trn_sv, sj_dev_sv, sj_tst_sv)






