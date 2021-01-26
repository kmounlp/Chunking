"""
[compare_BI_correct]
2019-07-28
BI 수정 규칙이 제대로 되었는지 확인하는 모듈

1. BI 수정 규칙에 따라 수정한 결과에 다시 수정 규칙을 적용해서 두 문서를 비교

2. 다른 문장이 있으면 수정 규칙이 중복 적용된 것이라 추정
(과도한 수정이 일어났다고 봄)

::결과:: 재 수정된 문장: 총 7개
모두 'I-PUX', 'I-ECX' 문제였음 (error10)
문장을 확인해보진 않아서 수정된 내용이 정확한 것인지는 확인할 수 없으나
BI 오류 개선은 맞고, 그 개수가 적으므로 이후는 시스템 오류라고 상정하고 진행
(차후 여유가 있다면 왜 error10이 다시 생겼는지 생각)

::재수정된 문장::
2061
 ['B-PX', 'B-ETX', 'B-PX', 'I-PX', 'B-ETX', 'B-CX', 'I-CX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-SYX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-NX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ECX', 'B-SYX', 'B-ECX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'B-JKX', 'B-AX', 'B-PX', 'B-ECX', 'B-PX', 'B-ECX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-NX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ECX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-EPX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
 ['B-PX', 'B-ETX', 'B-PX', 'I-PX', 'B-ETX', 'B-CX', 'I-CX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-SYX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-NX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ECX', 'B-SYX', 'B-ECX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'B-JKX', 'B-AX', 'B-PX', 'B-ECX', 'B-PX', 'B-ECX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-NX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ECX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-EPX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
2159
 ['B-NX', 'I-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-AX', 'B-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JKX', 'I-JKX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-EFX', 'B-SYX']
 ['B-NX', 'I-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-AX', 'B-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JKX', 'I-JKX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-EFX', 'B-SYX']
6437
 ['B-SYX', 'B-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JUX', 'B-PX', 'B-ETX', 'B-NX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-NX', 'B-JUX', 'B-PX', 'I-PX', 'B-EFX', 'B-SYX']
 ['B-SYX', 'B-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JUX', 'B-PX', 'B-ETX', 'B-NX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-NX', 'B-JUX', 'B-PX', 'I-PX', 'B-EFX', 'B-SYX']
8025
 ['B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'B-JCX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-SYX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-CX', 'I-CX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'I-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
 ['B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'I-NX', 'B-JCX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-SYX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-NX', 'B-JKX', 'B-CX', 'I-CX', 'B-ETX', 'B-NX', 'I-NX', 'B-JKX', 'I-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
9521
 ['B-NX', 'I-NX', 'I-NX', 'B-JUX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-EPX', 'B-ECX', 'B-NX', 'B-JUX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-PX', 'B-EPX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-ECX', 'B-SYX', 'B-NX', 'B-PX', 'B-EPX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-SYX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ECX', 'B-PX', 'B-EPX', 'B-EFX', 'B-SYX']
 ['B-NX', 'I-NX', 'I-NX', 'B-JUX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-SYX', 'B-NX', 'I-NX', 'B-JKX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-EPX', 'B-ECX', 'B-NX', 'B-JUX', 'B-NX', 'I-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ECX', 'B-PX', 'B-EPX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-ECX', 'B-SYX', 'B-NX', 'B-PX', 'B-EPX', 'B-ETX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-SYX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-ETX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ECX', 'B-PX', 'B-EPX', 'B-EFX', 'B-SYX']
13502
 ['B-NX', 'B-JUX', 'B-SYX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-CX', 'I-CX', 'I-CX', 'B-ETX', 'I-ETX', 'I-ETX', 'B-PX', 'B-ECX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-SYX', 'I-PUX', 'B-ECX', 'B-SYX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JCX', 'B-PX', 'B-ETX', 'B-JKX', 'I-JKX', 'B-PX', 'I-PX', 'B-ECX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-JCX', 'B-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JUX', 'B-AX', 'B-NX', 'B-MX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
 ['B-NX', 'B-JUX', 'B-SYX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-CX', 'I-CX', 'I-CX', 'B-ETX', 'I-ETX', 'I-ETX', 'B-PX', 'B-ECX', 'B-NX', 'I-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-SYX', 'I-PUX', 'B-ECX', 'B-SYX', 'B-NX', 'I-NX', 'I-NX', 'I-NX', 'I-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JCX', 'B-PX', 'B-ETX', 'B-JKX', 'I-JKX', 'B-PX', 'I-PX', 'B-ECX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-JCX', 'B-NX', 'B-JMX', 'B-PX', 'I-PX', 'B-ETX', 'B-NX', 'B-JUX', 'B-AX', 'B-NX', 'B-MX', 'B-NX', 'B-JMX', 'B-NX', 'B-JKX', 'B-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-EPX', 'B-EFX', 'B-SYX']
60582
 ['B-AX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-MX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-JUX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-ECX', 'I-ECX', 'B-PX', 'I-PX', 'B-ETX', 'B-CX', 'I-CX', 'B-EFX', 'B-SYX']
 ['B-AX', 'B-NX', 'I-NX', 'B-JUX', 'B-SYX', 'B-MX', 'B-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'B-ETX', 'B-NX', 'B-JUX', 'B-NX', 'I-NX', 'I-NX', 'B-JKX', 'B-PX', 'I-PX', 'B-PUX', 'I-PUX', 'I-PUX', 'I-PUX', 'B-PX', 'I-PX', 'B-ETX', 'B-CX', 'I-CX', 'B-EFX', 'B-SYX']


Y. Namgoong
"""
from utils_saveNload import load_BI_correct, load_BI_correct_2

bi_corr = load_BI_correct()
bi_corr_2 = load_BI_correct_2()

for i, (bi, bi_2) in enumerate(zip(bi_corr, bi_corr_2), 1):
    if bi == bi_2:
        pass
    else:
        print(i, '\n', bi, '\n', bi_2)
