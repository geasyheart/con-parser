# -*- coding: utf8 -*-
#

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained(
    'hfl/chinese-electra-180g-small-discriminator'
)

res = t.tokenize(text='1992年，我还小')
print(res)
# ['1992', '年', '，', '我', '还', '小']
# ['[CLS]', '1992', '年', '，', '我', '还', '小', '[SEP]']
