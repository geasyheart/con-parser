# -*- coding: utf8 -*-
#
import pathlib

DATA_PATH = pathlib.Path('.').parent.joinpath('data')

TRAIN_PATH = DATA_PATH.joinpath('train.noempty.txt')
DEV_PATH = DATA_PATH.joinpath('dev.noempty.txt')
TEST_PATH = DATA_PATH.joinpath('test.noempty.txt')
SAMPLE_PATH = DATA_PATH.joinpath('sample.noempty.txt')

MODEL_PATH = DATA_PATH.joinpath('savepoint')

# TRAIN_PATH = SAMPLE_PATH
# DEV_PATH = SAMPLE_PATH

# 如果一个句子中词语的个数超过50那么则忽略这条数据，显存不够
WORD_COUNT = 50