# -*- coding: utf8 -*-
#

from src import con_parser
from src.config import TRAIN_PATH, DEV_PATH, PRETRAINED_NAME_OR_PATH

m = con_parser.ConParser()
m.fit(
    train_path=TRAIN_PATH,
    dev_path=DEV_PATH,
    pretrained_model_name=PRETRAINED_NAME_OR_PATH,
    lr=1e-3,
    transformer_lr=1e-5,
    batch_size=32,
    epoch=300
)
