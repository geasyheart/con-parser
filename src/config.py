# -*- coding: utf8 -*-
#
import pathlib

DATA_PATH = pathlib.Path(__file__).parent.joinpath('data')

TRAIN_PATH = DATA_PATH.joinpath('company_data').joinpath('train.txt')
DEV_PATH = DATA_PATH.joinpath('company_data').joinpath('dev.txt')
# TEST_PATH = DATA_PATH.joinpath('test.noempty.txt')
# SAMPLE_PATH = DATA_PATH.joinpath('sample.noempty.txt')

MODEL_PATH = DATA_PATH.joinpath('savepoint')

# TRAIN_PATH = SAMPLE_PATH
# DEV_PATH = SAMPLE_PATH

# 如果一个句子中词语的个数超过50那么则忽略这条数据，显存不够
WORD_COUNT = 75

PRETRAINED_NAME_OR_PATH = 'hfl/chinese-electra-180g-small-discriminator'
