"""
[ BI_error_corrector_2 ]
2019-07-28
system적으로 걸러낼 수 있는 것은 완료.
1. Chunk-based Dependency Corpus의 입력이 될 pickle 파일 출력
::수정 전 파일 (입력 파일)::
".\pickle\0617_02\ch_pred_sent.pickle"
::출력 파일::
".\BI_correct\BI_correct.pickle"

2. 파일 출력 후 수동 오류 개선에 편리하게 코드 수정할 것.
1에서 출력한 파일을 코드 수정 없이 그대로 다시 입력으로 넣으면
수정 되는 항목 없어야 함.
(멀쩡한 BI까지 수정해버리면 안됨!)
(시스템 상에서 수정하지 않은 갯수들이 약 1240여야 함.)
::입력 파일::
".\BI_correct\BI_correct.pickle"
::출력 파일::
".\BI_correct\BI_correct_2.pickle" 
--> 이 파일을 입력으로 해서 toChunk-based_DepCorpus_ver3.0.0을 실행
--> BIErrorAnalysis를 통해 에러 유형별 분석
--> rule을 추가할 사항이 있으면 추가. 이외에는 수동으로 오류 수정

3. 출력한 파일을 토대로 수동으로 오류 수정 (약 1240개 ~ 6시간 소요 예상)

Y. Namgoong
"""

from utils_saveNload import load_pickle_chr_pred, save_BI_correct, save_BI_correct_pickle
from utils_saveNload import load_BI_correct, save_BI_correct_2, save_BI_correct_2_pickle
from config import BI_PATH
import os, pickle

if not os.path.isdir(BI_PATH):
    os.mkdir(BI_PATH)


def err1(chk_sent):
    """
    err1에 해당하는 에러를 규칙으로 처리했더니 과교정으로 err6에 오류가 많이 생겨 이 부분만 따로 수정하기 위해 만든 모듈
    B-SYX 뒤에 B_SYX가 여러 개 올 경우 모두 I-SYX로 변경
    (수정) SYX가 연달아 두 개 오는 경우는 있음. 이 경우 제외!
    :param chk_sent: a sequence of chunk (a sentence)
    :return: modified sequence of chunk (in terms of 'SYX')
    """
    start = 0
    for i, chk in enumerate(chk_sent):
        if chk == 'B-SYX':
            if start == 1:
                chk_sent[i] = 'I-SYX'
            elif start == 0:
                start += 1
        else:
            start = 0
    return chk_sent


def err_corrector():

    # 1번에 해당하는 입력. 이것을 입력으로 한 출력이 ChunkTagger의 결과물
    # chk_pred = load_pickle_chr_pred()  # chunk tagger로 예측한 결과물이자 toChunk-based_DepCorpus의 입력
    # 1번의 출력. 2번의 입력
    chk_pred = load_BI_correct()

    chk_sent_list = []
    for j, chk_sent in enumerate(chk_pred, 1):
        for i, chk in enumerate(chk_sent):
            # present morpheme info.
            bi_tag, chk_tag = chk.split('-')
            # next morpheme info.
            try:
                next_bi_tag, next_chk_tag = chk_sent[i+1].split('-')

                if bi_tag == "B" and next_bi_tag == "B":
                    if len(chk_tag) == 2:
                        if next_chk_tag == chk_tag:  # B-NX, B-NX
                            pass
                        elif len(next_chk_tag) == 2:  # B-NX, B-PX
                            pass
                        elif len(next_chk_tag) == 3:  # B-NX, B-JKX
                            pass
                        else: pass
                    elif len(chk_tag) == 3:
                        if next_chk_tag == chk_tag:  # B-JKX, B-JKX  # error1
                            if chk_tag == 'B-SYX':
                                chk_sent = err1(chk_sent)  # managing SYX
                            elif chk_tag == 'B-JUX':
                                chk_sent[i+1] = 'I-JUX'
                            # elif chk_tag == 'B-JKX':  # 20855:12 선 한 줄이, 23068:1 그 뿐이
                            #     chk_sent[i+1] = 'I-JKX'
                            elif chk_tag == 'B-EPX':
                                chk_sent[i+1] = 'I-EPX'
                        elif len(next_chk_tag) == 2:  # B-JKX, B-NX
                            pass
                        elif len(next_chk_tag) == 3:  # B-JKX, B-JUX
                            pass
                        else: pass
                elif bi_tag == "B" and next_bi_tag == "I":
                    if len(chk_tag) == 2:
                        if next_chk_tag == chk_tag:  # B-NX, I-NX
                            pass
                        elif len(next_chk_tag) == 2:  # B-NX, I-PX  # error2
                            # print("sent_id:", j)
                            # print(chk_sent)
                            # print(chk_sent[i+1], chk)
                            if chk_sent[i+1] == 'I-PX':
                                # print(chk_sent)
                                chk_sent[i] = 'B-PX'
                                # print(chk_sent)  # 여기서 save 해야함.
                                # input()
                        elif len(next_chk_tag) == 3:  # B-NX, I-JKX  # error3
                            pass  # 수동 교정 할 계획
                        else: pass
                    elif len(chk_tag) == 3:
                        if next_chk_tag == chk_tag:  # B-JKX, I-JKX
                            pass
                        elif len(next_chk_tag) == 2:  # B-JKX, I-NX  # error4
                            pass
                        elif len(next_chk_tag) == 3:  # B-JKX, I-JUX  # error5
                            if chk_sent[i] == 'B-SYX' and chk_sent[i+1] == 'I-ECX':
                                chk_sent[i] = 'I-ECX'
                                if chk_sent[i-1] == 'B-EFX':  # rage error 나는지 확인
                                    chk_sent[i-1] = 'B-ECX'

                        else: pass
                elif bi_tag == "I" and next_bi_tag == "B":
                    if len(chk_tag) == 2:
                        if next_chk_tag == chk_tag:  # I-NX, B-NX
                            pass
                        elif len(next_chk_tag) == 2:  # I-NX, B-PX
                            pass
                        elif len(next_chk_tag) == 3:  # I-NX, B-JKX
                            pass
                        else: pass
                    elif len(chk_tag) == 3:
                        if next_chk_tag == chk_tag:  # I-JKX, B-JKX  # error6
                            chk_sent[i+1].split('-')[0] = 'I'
                        elif len(next_chk_tag) == 2:  # I-JKX, B-NX
                            pass
                        elif len(next_chk_tag) == 3:  # I-JKX, B-JUX
                            pass
                        else: pass
                elif bi_tag == "I" and next_bi_tag == "I":
                    if len(chk_tag) == 2:
                        if next_chk_tag == chk_tag:  # I-NX, I-NX
                            pass
                        elif len(next_chk_tag) == 2:  # I-NX, I-PX  # error7
                            pass
                        elif len(next_chk_tag) == 3:  # I-NX, I-JKX  # error8
                            if chk == 'I-AX' and chk_sent[i+1] == 'I-JKX':
                                chk_sent[i+1] = 'I-AX'
                        else: pass
                    elif len(chk_tag) == 3:
                        if next_chk_tag == chk_tag:  # I-JKX, I-JKX
                            pass
                        elif len(next_chk_tag) == 2:  # I-JKX, I-NX  # error9
                            pass
                        elif len(next_chk_tag) == 3:  # I-JKX, I-JUX  # error10
                            chk_sent[i+1] = 'I-' + chk_tag
                        else: pass
                else: pass
            except IndexError:  # be raised error when the final morpheme comes.
                # processing for the last morpheme
                if len(chk_tag) == 2:  # content words
                    pass
                elif len(chk_tag) == 3:  # functional words
                    pass
                else:  # for handling exceptions
                    pass

        # print("result: ", j, chk_sent)
        # with open(os.path.join(OUT_PATH, 'BI_correct.sent'), 'a', encoding='utf-8') as f:
        #     f.write(str(chk_sent)+'\n')
        # with open(os.path.join(OUT_PATH, 'BI_correct.pickle'), 'ab') as f:
        #     pickle.dump(chk_sent, f)

        chk_sent_list.append(chk_sent)

        # save_BI_correct(chk_sent)
        save_BI_correct_2(chk_sent)
    return chk_sent_list


if __name__ == "__main__":
    # save_BI_correct_pickle(err_corrector())
    save_BI_correct_2_pickle(err_corrector())

    # err1() 모듈을 만들기 위해 test script
    # chk_sent = ['B-PX', 'B-EFX', 'B-SYX', 'B-SYX', 'B-SYX', 'B-PX', 'B-EFX', 'B-SYX', 'B-SYX', 'B-SYX', 'B-PX', 'B-EFX']
    # print("before:", chk_sent)
    # print("after:", err1(chk_sent))






