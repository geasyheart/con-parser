# -*- coding: utf8 -*-
#
import uuid
from typing import List, Tuple

from constituency_labeling import label_tree, convert

from src.preprocess.frontend import FrontendTree


def get_final_label(type: str, clause, label):
    size = 0
    if type:
        size += 1
    if clause:
        size += 1
    if label:
        size += 1
    if size != 1:
        raise ValueError(f'type: {type}, clause: {clause}, label: {label}')
    return type or clause or label


def convert_frontend_to_label_tree_base(frontend: FrontendTree) -> label_tree.LabelTree:
    """
    这是基本转成labeltree方法，但是还不能用，请使用下面方法：
        convert_frontend_to_label_tree_by_single_word()
        convert_frontend_to_label_tree_by_cut_words()
    因为叶子节点不会是它的哦



    去掉补的主语
    label顺序: label（成分） > type（句型） > clause（分句句型）

    为什么这样呢？
        首先constituency是允许多label存在的，但是这里为什么要设置优先级呢？
        答案：为了提升准确率，（个人猜测）

        label即成分，这个不像句型是可以通过分句句型来推测出来的，另外句型和分句句型是互斥的
    当然，你可以按照自己的思路来写这里，有可能模型的拟合能力非常强。
    :param frontend:
    :return:
    """
    ltree = label_tree.LabelTree()
    node_map = {}

    for index, node in enumerate(frontend.dfs(node=frontend.root)):
        label_node = label_tree.Node(
            id=f'node-{index}',
            cut_words=node.words,
            label=get_final_label(node.type, node.clause, node.label),
            extra=node.to_dict()
        )
        # 这里是constituency convert用
        if not node.children:
            label_node.cut_words = [(label_node.cut_words, label_node.label)]
        node_map[id(node)] = label_node

        label_node_parent = node_map[id(node.parent)] if node.parent else None
        label_node.set_parent(node=label_node_parent)
        if label_node_parent is not None:
            label_node_parent.add_child(label_node)
    for _, lnode in node_map.items():
        if lnode.parent is None:
            ltree.root = lnode
    return ltree
