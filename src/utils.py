# -*- coding: utf8 -*-
#
import logging
import sys
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AdamW

from src.config import DATA_PATH


class TqdmHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def get_logger():
    logger = logging.getLogger('con-parser')
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'
    file_handler = logging.FileHandler(filename=DATA_PATH.joinpath('run.log'))
    tqdm_handler = TqdmHandler()
    file_handler.setFormatter(logging.Formatter(fmt))
    tqdm_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    logger.addHandler(tqdm_handler)

    return logger


logger = get_logger()


def build_optimizer_for_pretrained(model: torch.nn.Module,
                                   pretrained: torch.nn.Module,
                                   lr=1e-5,
                                   weight_decay=0.01,
                                   eps=1e-8,
                                   transformer_lr=None,
                                   transformer_weight_decay=None,
                                   no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
                                   **kwargs):
    if transformer_lr is None:
        transformer_lr = lr
    if transformer_weight_decay is None:
        transformer_weight_decay = weight_decay
    params = defaultdict(lambda: defaultdict(list))
    pretrained = set(pretrained.parameters())
    if isinstance(no_decay, tuple):
        def no_decay_fn(name):
            return any(nd in name for nd in no_decay)
    else:
        assert callable(no_decay), 'no_decay has to be callable or a tuple of str'
        no_decay_fn = no_decay
    for n, p in model.named_parameters():
        is_pretrained = 'pretrained' if p in pretrained else 'non_pretrained'
        is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
        params[is_pretrained][is_no_decay].append(p)

    grouped_parameters = [
        {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
        {'params': params['pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': transformer_lr},
        {'params': params['non_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
        {'params': params['non_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
    ]

    return AdamW(
        grouped_parameters,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        **kwargs)
