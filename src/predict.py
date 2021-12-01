# -*- coding: utf8 -*-
#


from src import con_parser
from src.config import MODEL_PATH

m = con_parser.ConParser()
m.load(
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    model_path=str(MODEL_PATH.joinpath('dev_metric_8.4596e-01.pt'))
)

sample1 = [('经济', 'NN'),
 ('专家', 'NN'),
 ('提出', 'VV'),
 ('进一步', 'AD'),
 ('扩大', 'VV'),
 ('海南', 'NN'),
 ('对', 'P'),
 ('外', 'NN'),
 ('开放', 'NN'),
 ('的', 'DEC'),
 ('系列', 'NN'),
 ('建议', 'NN')]

# sample2 = [('新华社', 'NN'),
#            ('南宁', 'NR'),
#            ('二月', 'NT'),
#            ('十四日', 'NT'),
#            ('电', 'NN'),
#            ('（', 'PU'),
#            ('记者', 'NN'),
#            ('刘水玉', 'NR'),
#            ('）', 'PU')]

m.predict(
    samples=[
        # [i[0] for i in sample1],
        # [i[0] for i in sample2],
        sample1,
        # sample2
    ]
)
