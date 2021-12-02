# -*- coding: utf8 -*-
#
import json
from unittest import TestCase
from src.preprocess import convert, frontend

with open('/home/yuzhang/PycharmProjects/con-parser/src/preprocess/docs/sample.json') as f:
    data = json.loads(f.read())


class TestSample(TestCase):
    def test_convert(self):
        frontend_tree = frontend.FrontendTree()
        frontend_tree.generate_tree(data)

        label_tree = convert.convert_frontend_to_label_tree(frontend_tree)
        label_tree.pretty_tree(filename='label_tree_sample.gv')
