# -*- coding: utf8 -*-
#
import uuid

from constituency_labeling import label_tree

from src.preprocess.frontend import FrontendTree


def label_without_number(label: str):
    """
    比如： 先事1,先事2,后事1,后事2,宾1,宾2

    这个可以推算出来，减少label数量
    :param label:
    :return:
    """
    return "".join([l for l in label if not l.isdigit()])


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
        # 如果为补齐短语，那么则跳过，因为不是在这个任务里面做
        if node.supply_subject: continue
        # 如果补的主语为空（就是标注也不知道这个主语是什么），那么则跳过
        if not node.words: continue

        label_node = label_tree.Node(
            id=f'node-{index}',
            cut_words=node.words,
            label=label_without_number(node.label or node.type or node.clause),
            extra=node.to_dict()
        )
        node_map[id(node)] = label_node

        label_node_parent = node_map[id(node.parent)] if node.parent else None
        label_node.set_parent(node=label_node_parent)
        if label_node_parent is not None:
            label_node_parent.add_child(label_node)
    for _, lnode in node_map.items():
        if lnode.parent is None:
            ltree.root = lnode
    return ltree


def convert_frontend_to_label_tree_by_single_word(ftree: FrontendTree) -> label_tree.LabelTree:
    """
    以字为单位作为叶子节点
    但是这里又有两种做法：
    1. 以字为单位
    2. 以wordpiece为单位

    这里采用1方式，为什么？如果wordpiece有边界冲突问题，虽然概率很小。。。反正你都可以尝试

     他的词性设置为`pos`，即空缺
    :param ftree:
    :return:
    """
    ltree = convert_frontend_to_label_tree_base(frontend=ftree)
    new_nodes = []
    for node in ltree.dfs(node=ltree.root):
        if node.children: continue

        for char in node.cut_words:
            new_node = label_tree.Node(
                id=str(uuid.uuid4()),
                cut_words=[(char, 'pos')],
                label='',
                extra=None
            )

            new_nodes.append((node, new_node))
    for (parent_node, new_child_node) in new_nodes:
        new_child_node.set_parent(parent_node)
        parent_node.add_child(node=new_child_node)
    return ltree


def convert_frontend_to_label_tree_by_cut_words(ftree: FrontendTree) -> label_tree.LabelTree:
    """
    以分词为单位作为叶子节点

    首先句子分词，然后找到叶子节点进行分割，如果没有边界问题，那么则皆大欢喜，证明这条路可以尝试
    如果边界问题比较严重，那么只能使用以字为单位的了
    :param ftree:
    :return:
    """
    ltree = convert_frontend_to_label_tree_base(frontend=ftree)
    raise NotImplementedError
