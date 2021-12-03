# -*- coding: utf8 -*-
#
import json
from unittest import TestCase

from constituency_labeling.convert import label_tree_to_nltk

from src.preprocess import convert, frontend

with open('/home/yuzhang/PycharmProjects/con-parser/src/preprocess/docs/sample.json') as f:
    data = json.loads(f.read())


class TestSample(TestCase):
    def test_frontend_to_label_tree_convert(self):
        frontend_tree = frontend.FrontendTree()
        frontend_tree.generate_tree(data)

        label_tree = convert.convert_frontend_to_label_tree_by_single_word(frontend_tree)
        label_tree.pretty_tree(filename='label_tree_sample.gv')

    def test_convert_frontend_to_label_tree_by_single_word(self):
        frontend_tree = frontend.FrontendTree()
        frontend_tree.generate_tree(data)

        label_tree = convert.convert_frontend_to_label_tree_by_single_word(frontend_tree)
        nltk_tree = label_tree_to_nltk(label_tree)

        # nltk_tree.pretty_print()
        for subtree in nltk_tree.subtrees(lambda t: t.height() != 2):
            print(subtree.pos(), subtree.label())

    def test_convert_frontend_to_label_tree_by_cut_words(self):
        frontend_tree = frontend.FrontendTree()
        frontend_tree.generate_tree(data)

        ltree = convert.convert_frontend_to_label_tree_by_cut_words(
            frontend_tree,
            cut_words=[
                ('他', 'n'),
                ('指出', 'v'),
                ('要', 'v'),
                ('深入', 'v'),
                ('贯彻', 'v'),
                ('…', 'p'),
                ('…', 'p'),
                ('论述', 'n'),
                ('，', 'p'),

                ('提高', 'v'),
                ('…', 'p'),
                ('…', 'p'),
                ('站位', 'n'),
                ('，', 'p'),
                ('落实', 'v'),
                ('…', 'p'),
                ('…', 'p'),
                ('责任', 'n')
            ]
        )
        nltk_tree = label_tree_to_nltk(ltree)
        nltk_tree.pretty_print()

    def test_batch_convert(self):
        """
        以标注数据集为准，测试frontend_tree转成label_tree再转nltk_tree，
        以及nltk_tree转回label_tree是否正确
        :return:
        """