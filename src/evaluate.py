# -*- coding: utf8 -*-
#
from src import con_parser
from src.config import MODEL_PATH, DEV_PATH, PRETRAINED_NAME_OR_PATH

m = con_parser.ConParser()
m.load(
    pretrained_model_name=PRETRAINED_NAME_OR_PATH,
    model_path=str(MODEL_PATH.joinpath('dev_metric_8.5929e-01.pt')),
    device='cuda'
)

dev = m.build_dataloader(
    DEV_PATH,
    transformer=m.tokenizer,
    batch_size=2,
    shuffle=False
)
loss, metric = m.evaluate_dataloader(dev)
print(loss, metric)
