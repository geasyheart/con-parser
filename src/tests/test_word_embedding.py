# -*- coding: utf8 -*-
#

from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from src.layers.transformer import TransformerWordEmbedding

# ltp中word embedding的实现.
def tokenize(words, tokenizer, max_length):
    res = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True
    )
    word_index = []
    for encoding in res.encodings:
        word_index.append([])

        last_word_idx = -1
        current_length = 0
        for word_idx in encoding.word_ids[1:-1]:
            if word_idx != last_word_idx:
                word_index[-1].append(current_length)
            current_length += 1
            last_word_idx = word_idx

    result = res.data
    for ids in result['input_ids']:
        ids[0] = tokenizer.cls_token_id
        ids[-1] = tokenizer.sep_token_id
    result['overflow'] = [len(encoding.overflowing) > 0 for encoding in res.encodings]
    result['word_index'] = word_index
    result['word_attention_mask'] = [[True] * len(index) for index in word_index]
    return result


def pad_tokenize(words, tokenizer, max_length):
    result = tokenize(
        words=words,
        tokenizer=tokenizer,
        max_length=max_length
    )
    batch_input_ids: List[List[int]] = result['input_ids']
    batch_token_type_ids: List[List[int]] = result['token_type_ids']
    batch_attention_mask: List[List[int]] = result['attention_mask']
    overflow: List[bool] = result['overflow']
    batch_word_index: List[List[int]] = result['word_index']
    batch_word_attention_mask: List[List[bool]] = result['word_attention_mask']

    input_ids = pad_sequence([torch.tensor(i, dtype=torch.long) for i in batch_input_ids], batch_first=True)
    token_type_ids = pad_sequence([torch.tensor(i, dtype=torch.long) for i in batch_token_type_ids], batch_first=True)
    attention_mask = pad_sequence([torch.tensor(i, dtype=torch.long) for i in batch_attention_mask], batch_first=True)
    word_index = pad_sequence([torch.tensor(i, dtype=torch.long) for i in batch_word_index], batch_first=True)
    word_attention_mask = pad_sequence([torch.tensor(i, dtype=torch.bool) for i in batch_word_attention_mask],
                                       batch_first=True, padding_value=False)

    return input_ids, token_type_ids, attention_mask, overflow, word_index, word_attention_mask


if __name__ == '__main__':
    from transformers import AutoTokenizer


    t = AutoTokenizer.from_pretrained('hfl/chinese-electra-180g-small-discriminator')
    tokenize(words=[
        ['我', '爱', '北京'],
        ['我', '爱你']
    ], tokenizer=t, max_length=512)

    tokenize(words=[
        ['我', '爱', '北京'],
        ['我', '爱你']
    ], tokenizer=t, max_length=512)


    input_ids, token_type_ids, attention_mask, overflow, word_index, word_attention_mask = pad_tokenize(words=[
        ['我', '爱', '北京'],
        ['我', '爱你']
    ], tokenizer=t, max_length=512)



    model = TransformerWordEmbedding(
        transformer='hfl/chinese-electra-180g-small-discriminator'
    )

    bert_out, word_attention_mask = model(input_ids, token_type_ids, attention_mask, word_index, word_attention_mask)
    print('h')