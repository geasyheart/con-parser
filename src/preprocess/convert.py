# -*- coding: utf8 -*-
#
from constituency_labeling import label_tree

from src.preprocess.frontend import FrontendTree


def convert_frontend_to_label_tree(frontend: FrontendTree) -> label_tree.LabelTree:
    """
    去掉补的主语
    label顺序: label > type > clause
    :param frontend:
    :return:
    """
    ltree = label_tree.LabelTree()
    node_map = {}

    for index, node in enumerate(frontend.dfs(node=frontend.root)):
        # 如果为补齐短语，那么则跳过
        if node.supply_subject: continue

        label_node = label_tree.Node(
            id=f'node-{index}',
            cut_words=node.words,
            label=node.label or node.type or node.clause,
            extra=node.to_dict()
        )
        # TODO : 如果label_node.label 包含并行1这种以数字结尾的，那么就干掉
        node_map[id(node)] = label_node

        label_node_parent = node_map[id(node.parent)] if node.parent else None
        label_node.set_parent(node=label_node_parent)
        if label_node_parent is not None:
            label_node_parent.add_child(label_node)
    for _, lnode in node_map.items():
        if lnode.parent is None:
            ltree.root = lnode
    return ltree

