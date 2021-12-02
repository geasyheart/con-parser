# -*- coding: utf8 -*-
#
import json
from pprint import pprint
from unittest import TestCase

from src.preprocess.frontend import FrontendTree


class TestFrontendTree(TestCase):
    def test_generate_tree(self):
        with open('/home/yuzhang/PycharmProjects/con-parser/src/preprocess/docs/sample.json', 'r') as f:
            sample = json.loads(f.read())
            tree = FrontendTree()
            tree.generate_tree(sample)
            tree.pretty_tree(filename='代码格式.gv')

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

