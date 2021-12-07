# -*- coding: utf8 -*-
#
import json
from pprint import pprint
from unittest import TestCase

from constituency_labeling import label_tree_to_nltk

from src.preprocess import convert
from src.preprocess.frontend import FrontendTree


class TestFrontendTree(TestCase):
    def test_generate_tree(self):
        with open('/home/yuzhang/PycharmProjects/con-parser/src/preprocess/docs/测试导出.json', 'r') as f:
            content = f.read()
            # without bom
            if content.startswith(u'\ufeff'):
                content = content.encode('utf8')[3:].decode('utf8')

            samples = json.loads(content)['data']
            last_sample = samples[-1]
            tree = FrontendTree()
            tree.generate_tree(last_sample)
            tree.pretty_tree(filename='1.gv')

            ltree = convert.convert_frontend_to_label_tree_by_cut_words(
                tree,
                cut_words=last_sample['cut_words'],
            )
            nltk_tree = label_tree_to_nltk(ltree)
            nltk_tree.pretty_print()
            pprint(nltk_tree.productions())

    def test_tree_to_json(self):
        with open('/home/yuzhang/PycharmProjects/con-parser/src/preprocess/docs/sample.json', 'r') as f:
            sample = json.loads(f.read())
            tree = FrontendTree()
            tree.generate_tree(sample)
            return_frontend = tree.to_dict()
            # 看看是否一样
            tree2 = FrontendTree()
            tree2.generate_tree(return_frontend)

            tree_nodes = [n1 for n1 in tree.dfs(node=tree.root)]
            tree2_nodes = [n2 for n2 in tree2.dfs(node=tree.root)]
            assert tree_nodes == tree2_nodes
