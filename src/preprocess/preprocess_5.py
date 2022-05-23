# 区分训练和测试数据集
import os.path
import random

path = '/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_nltk_data.txt'

index_map = {}
with open(path, 'r') as f:
    for index, line in enumerate(f):
        index_map.setdefault(index, line)

train_indices = random.sample(index_map.keys(), int(len(index_map) * 0.85))

train_f = open(os.path.join(os.path.dirname(path), 'train.txt'), 'w')
dev_f = open(os.path.join(os.path.dirname(path), 'dev.txt'), 'w')

for index, v in index_map.items():
    if index in train_indices:
        train_f.write(v)
    else:
        dev_f.write(v)
train_f.close()
dev_f.close()
