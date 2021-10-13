# -*- coding: utf8 -*-
#

from src import con_parser
from src.config import TRAIN_PATH, DEV_PATH

m = con_parser.ConParser()
m.fit(
    train_path=TRAIN_PATH,
    dev_path=DEV_PATH,
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    lr=1e-3,
    batch_size=32
)
