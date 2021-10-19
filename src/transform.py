# -*- coding: utf8 -*-
#
import json
from typing import List, Dict

import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.algo import Tree
from src.config import DATA_PATH, TRAIN_PATH, DEV_PATH, WORD_COUNT


def get_labels() -> Dict:
    label_map_path = DATA_PATH.joinpath('label_map.json')
    if label_map_path.exists():
        with open(label_map_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

    def _get_label(file):
        labels = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='get labels'):
                tree = nltk.Tree.fromstring(line)
                for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
                    labels.setdefault(label, 0)
                    labels[label] += 1
        return labels

    label1 = _get_label(TRAIN_PATH)
    label2 = _get_label(DEV_PATH)
    final_label = {'[PAD]': 0}
    for label in label1:
        final_label.setdefault(label, len(final_label))
    for label in label2:
        final_label.setdefault(label, len(final_label))
    with open(label_map_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_label, ensure_ascii=False, indent=2))
    return final_label


def get_tags() -> Dict:
    pos_map_path = DATA_PATH.joinpath('pos_map.json')
    if pos_map_path.exists():
        with open(pos_map_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

    def _get_pos(file):
        pos = {}
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='get tags'):
                tree = nltk.Tree.fromstring(line)
                words, tags = zip(*tree.pos())
                for tag in tags:
                    pos.setdefault(tag, 0)
        return pos
    pos1 = _get_pos(TRAIN_PATH)
    pos2 = _get_pos(DEV_PATH)
    final_pos = {'[PAD]': 0}

    for _pos in pos1:
        final_pos.setdefault(_pos, len(final_pos))
    for _pos in pos2:
        final_pos.setdefault(_pos, len(final_pos))
    with open(pos_map_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_pos, ensure_ascii=False, indent=2))
    return final_pos


def encoder_texts(texts: List[List[str]], tokenizer):
    # 统计句子中最大的词长度
    fix_len = max([max([len(word) for word in text]) for text in texts])

    matrix = []
    for text in texts:
        vector = []

        text = [tokenizer.cls_token, *text, tokenizer.sep_token]
        input_ids = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
        )['input_ids']

        for _input_ids in input_ids:
            # 修复例如: texts = [['\ue5f1\ue5f1\ue5f1\ue5f1']] 这种情况
            _input_ids = _input_ids or [tokenizer.unk_token_id]
            vector.append(_input_ids + (fix_len - len(_input_ids)) * [tokenizer.pad_token_id])
        matrix.append(torch.tensor(vector, dtype=torch.long))
    return pad_sequence(matrix, batch_first=True)


class ConTransform(dataset.Dataset):
    def __init__(self, path: str, transformer: str, device: torch.device = 'cpu'):
        super(ConTransform, self).__init__()
        self.device = device

        self.trees = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='transform'):
                tree = nltk.Tree.fromstring(line)
                # 显存不够
                if len(tree.pos()) > WORD_COUNT:
                    continue
                self.trees.append(tree)
        self.trees = sorted(self.trees, key=lambda x: len(x.pos()))
        self.tokenizer = AutoTokenizer.from_pretrained(transformer) if isinstance(transformer, str) else transformer

        self.labels = get_labels()
        self.tags = get_tags()

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, item):
        tree = self.trees[item]
        words, tags = zip(*tree.pos())
        # 为什么在最前面添加0,因为charts[:, 0,0]均为-1
        tag_ids = [0] + [self.tags[tag] for tag in tags]
        chart = [[-1] * (len(words) + 1) for _ in range(len(words) + 1)]
        for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
            chart[i][j] = self.labels[label]
        return words, torch.tensor(tag_ids, dtype=torch.long), tree, torch.tensor(chart, dtype=torch.long)

    def to_dataloader(self, batch_size, shuffle):
        return dataloader.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        words = encoder_texts([i[0] for i in batch], tokenizer=self.tokenizer)
        tags = pad_sequence([i[1] for i in batch], batch_first=True)
        trees = [i[2] for i in batch]
        charts = [i[3] for i in batch]
        max_chart_len = max([i.size(0) for i in charts])

        charts_matrix = torch.zeros(size=(len(charts), max_chart_len, max_chart_len), dtype=torch.long)
        for i, chart in enumerate(charts):
            l = chart.size(0)
            charts_matrix[i, :l, :l] = chart

        return words.to(self.device), tags.to(self.device), trees, charts_matrix.to(self.device)



if __name__ == '__main__':
    texts = [['\ue5f1\ue5f1\ue5f1\ue5f1']]
    encoder_texts(
        texts,
        AutoTokenizer.from_pretrained(
            'hfl/chinese-electra-180g-small-discriminator'
        )
    )