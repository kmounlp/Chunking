import numpy as np

ck_input_rd = open("../chunk_input.conllc", 'r', encoding='utf-8')
sj_input_rd = open("../sejong_input.conllu", 'r', encoding='utf-8')

# for checking each data information
sj_train = open("../sejong/train.conllu", 'r', encoding='utf-8')
sj_dev = open("../sejong/dev.conllu", 'r', encoding='utf-8')
sj_test = open("../sejong/test.conllu", 'r', encoding='utf-8')

num_sent = 41643


# 전체 입력 성분 수
def num_lines(input_file):
    input_file.seek(0)
    lines = len(input_file.readlines())
    return lines - num_sent


# 한 문장당 평균 입력 성분 수
def mean_unit(input_file):
    input_file.seek(0)
    len_unit_seq = []
    len_unit = 0
    for line in input_file:
        line = line.strip()
        if line:
            len_unit += 1
        else:
            len_unit_seq.append(len_unit)
            len_unit = 0

    return np.mean(np.array(len_unit_seq))


# 전체 형태소 수
def num_pos(input_file, loc_pos):
    input_file.seek(0)
    count = 0
    for line in input_file:
        line = line.strip()
        if line:
            postags = line.split("\t")[loc_pos]
            postags = postags.split("+")
            count += len(postags)
    return count


# 한 문장당 평균 형태소 수
def mean_pos(input_file, loc_pos):
    input_file.seek(0)
    num_pos_seq = []
    num_pos = 0
    for line in input_file:
        line = line.strip()
        if line:
            postags = line.split("\t")[loc_pos]
            postags = postags.split("+")
            num_pos += len(postags)
        else:
            num_pos_seq.append(num_pos)
            num_pos = 0
    return np.mean(np.array(num_pos_seq))


if __name__ == "__main__":
    print("[ CHUNK ]")
    print("# of whole sent_component:", num_lines(ck_input_rd))
    print("mean sent_component:", mean_unit(ck_input_rd)) 

    print("\n[ SEJONG ]")
    print("# of whole eojul:", num_lines(sj_input_rd))
    print("mean eojul:", mean_unit(sj_input_rd))

    print("\n# of whole pos:", num_pos(sj_input_rd, 4))
    print("mean pos: ", mean_pos(sj_input_rd, 4))

    # number of pos in each file
    # print(num_pos(sj_train, 4), num_pos(sj_dev, 4), num_pos(sj_test, 4))