# -*- coding: utf8 -*-
#

from LAC import lac
from constituency_labeling.convert import nltk_tree_to_label
from flask import Flask, request, send_file

from src import con_parser
from src.config import MODEL_PATH

seg = lac.LAC()

app = Flask(__name__)
m = con_parser.ConParser()
m.load(
    pretrained_model_name='hfl/chinese-electra-180g-small-discriminator',
    model_path=str(MODEL_PATH.joinpath('dev_metric_7.9387e-01.pt'))
)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """
    用于展示效果
    :return:
    """
    sentence = (request.form or request.json or request.args or {}).get('sentence')
    ws, ps = seg.run(texts=[sentence])[0]
    segments = list(zip(ws, ps))

    preds = m.predict(samples=[segments])
    lt = nltk_tree_to_label(preds['trees'][0])
    lt.pretty_tree(filename='server.gv', view=False)
    return send_file('./examples/server.gv.svg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2337, processes=1)
