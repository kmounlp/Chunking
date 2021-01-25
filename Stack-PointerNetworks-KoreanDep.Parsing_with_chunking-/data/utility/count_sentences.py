"""
This file pre-processes the original input file
for checking original performance of the parser.
2019-11-09
"""

from os.path import exists, abspath, join

SJ_PATH = "../data/sejong_org/sejong-non_head_final_nospc.conll"
SAVE_PATH = "../data/sejong_org"

TRAIN = "train.conllu"
DEV = "dev.conllu"
TEST = "test.conllu"


if exists(SJ_PATH):
    print("file exist.")
else:
    print("file doesn't exist.")

save_file = open(join(SAVE_PATH, TRAIN), 'w', encoding='utf=8')

count = 1
with open(SJ_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line and line[0] == "#":
            continue
        print(line, file=save_file)
        if not line:
            # print("\n", file=save_file)
            if count == 49876:
                save_file = open(join(SAVE_PATH, DEV), 'w', encoding='utf-8')
            elif count == 56110:
                save_file = open(join(SAVE_PATH, TEST), 'w', encoding='utf-8')

            count += 1  # count the number of sentences








