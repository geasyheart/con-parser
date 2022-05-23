# -*- coding: utf8 -*-
#
import uuid
from typing import List, Tuple

from LAC import lac
from constituency_labeling.label_tree import LabelTree, LabelNode

from .frontend import FrontendTree, FrontendNode


def get_without_supply_subject_tree(ft: FrontendTree):
    """
    过滤掉所有需要主语补齐的
    :param ft:
    :return:
    """
    for node in ft.dfs(node=ft.root):
        if node.supply_subject:
            if node.parent:
                node.parent.children.remove(node)
    return ft


def get_without_empty_node(ft: FrontendTree):
    """
    过滤掉主语为空的节点

    但此处是将所有节点内容为空的过滤掉
    :param ft:
    :return:
    """
    for node in ft.dfs(node=ft.root):
        if not node.words:
            if node.parent:
                node.parent.children.remove(node)
    return ft


def rebuild_ft(ft: FrontendTree):
    """
    如果一个node有多个label，则进行拆分并产生新的node
    句型、分句句型、成分 产生新的node
    :param ft:
    :return:
    """
    for node in ft.dfs(node=ft.root):
        if node.type and node.clause and node.label:
            raise ValueError('不可能存在')
        if node.type and node.clause:
            # for type
            type_node = FrontendNode(
                words=node.words,
                type=node.type
            )
            type_node.set_parent(node=node)
            type_node.children = node.children

            node.children = [type_node, ]
            node.type = ''
            for child in type_node.children:
                child.set_parent(type_node)

        if node.type and node.label:

            # for type
            type_node = FrontendNode(
                words=node.words,
                type=node.type
            )
            type_node.set_parent(node=node)
            type_node.children = node.children

            node.children = [type_node, ]
            node.type = ''
            for child in type_node.children:
                child.set_parent(type_node)

        if node.clause and node.label:

            # for label
            label_node = FrontendNode(
                words=node.words,
                label=node.label,
            )
            label_node.set_parent(node=node)
            label_node.children = node.children

            node.children = [label_node, ]
            node.label = ''
            for child in label_node.children:
                child.set_parent(label_node)

    return ft


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def contain_chinese_string(string):
    """是否包含汉字"""
    return any(is_chinese(c) for c in string)


def fillup_symbols(node: FrontendNode):
    """
    给当前的node节点补齐符号，

    :param node:
    :return:
    """
    if not node: return
    new_node_children = []
    after = node.words
    for child_index, child in enumerate(node.children):
        if not child.words: continue
        index = after.find(child.words)
        if index == -1:
            # 按理说没找到意味着这个符号在原句不存在，那么这个不应该
            # 但是发现一个有趣的标注数据现象：
            # 原句： '实现科技自立自强、建设世界科技强国，广大科技工作者责任重大、使命光荣。'
            # 标注的child：['广大科技工作者责任重大、使命光荣', '实现科技自立自强、建设世界科技强国']
            # 1. 顺序反了，所以这里不处理
            # 2. 标的本身就有问题
            raise ValueError('句子顺序反了')
        if index != 0:
            fillup_words = after[:index]
            if not fillup_words.strip():
                continue
            if fillup_words not in [',', '◎', ';', '；', '%', '？', '↓', '”', '（', '，', '’', '。', '●', '‘', '：', '、', '“',
                                    '！', '…', '）', '《', '》']:
                continue
            new_node = FrontendNode(words=fillup_words, label='符号')
            new_node.set_parent(node)
            new_node_children.append(new_node)
            new_node_children.append(child)
        else:
            new_node_children.append(child)
        after = after[index + len(child.words):]
    if after and after != node.words and after in [',', '◎', ';', '；', '%', '？', '↓', '”', '（', '，', '’', '。', '●', '‘',
                                                   '：', '、', '“', '！', '…', '）', '《', '》']:
        new_node = FrontendNode(words=after, label='符号')
        new_node.set_parent(node)
        new_node_children.append(new_node)
    if new_node_children:
        node.children = new_node_children

    if node.children and node.words != "".join([i.words for i in node.children]):
        # TODO: 这里可以后处理，如果忘记了，看下面一句话，如果还想不起来，那debug看下呢
        # 比如： “”里面的内容去掉了，这部分应该是句子预处理了
        raise ValueError
    return node


def fillup_symbols_tree(tree: FrontendTree):
    for node in tree.dfs(node=tree.root):
        try:
            fillup_symbols(node=node)
        except ValueError:
            return
    return tree


lac = lac.LAC()


def _get_segment(segments: List[Tuple[str, str]], words: str, last_index: int = 0):
    tmp_segments = []
    for index, (word, pos) in enumerate(segments):
        if index < last_index: continue
        tmp_segments.append((word, pos))
        tmp_word = "".join([word for word, pos in tmp_segments])
        if tmp_word == words:
            return tmp_segments, index + 1
        if len(tmp_word) > len(words):
            return [], None
    return [], None


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


def ft_add_segment(ft: FrontendTree) -> LabelTree:
    """
    增加分词节点，如果有边界错误，则返回none
    :param ft:
    :return:
    """
    lt = LabelTree()
    segments = lac.run(texts=[ft.root.words])[0]
    segments = list(zip(*segments))

    lt.root = LabelNode(
        id=uuid.uuid4().hex,
        cut_words=segments,
        label=get_final_label(type=ft.root.type, clause=ft.root.clause, label=ft.root.label),
        extra=None
    )

    ft_lt_map = {id(ft.root): lt.root}

    for fns in ft.bfs_by_single_node(nodes=[ft.root]):
        last_index = 0
        for fn in fns:
            if fn is ft.root:
                continue

            ln = LabelNode(
                id=uuid.uuid4().hex,
                cut_words=[],
                label=get_final_label(type=fn.type, clause=fn.clause, label=fn.label),
                extra=None
            )
            ln.parent = ft_lt_map[id(fn.parent)]

            cut_words, last_index = _get_segment(segments=ln.parent.cut_words, words=fn.words, last_index=last_index)
            if last_index is None:
                raise ValueError('分词边界冲突')
            if not cut_words:
                raise NotImplementedError('不应该')
            ln.cut_words = cut_words

            ft_lt_map[id(fn)] = ln
            ft_lt_map[id(fn.parent)].add_child(ln)
    return lt


def lt_add_leaf_node(lt: LabelTree):
    skip_node_ids = []
    for node in lt.dfs(node=lt.root):
        if id(node) in skip_node_ids: continue
        if not node.children:
            # if len(node.cut_words) <= 1: continue
            for (word, pos) in node.cut_words:
                new_node = LabelNode(
                    id=uuid.uuid4().hex,
                    cut_words=[(word, pos)],
                    label=pos,
                    extra=None
                )
                skip_node_ids.append(id(new_node))
                new_node.set_parent(node)
                node.add_child(new_node)
    return lt


def convert_special_symbol(lt: LabelTree) -> LabelTree:
    """
    nltk在解析的时候，使用正则表达式，那么需要对句子中的（）进行转义
    :param lt:
    :return:

    {'(': '-LRB-', ')': '-RRB-'}
    """
    for node in lt.dfs(node=lt.root):
        if not node.children:
            assert len(node.cut_words) == 1
            word, pos = node.cut_words[0]
            if word == '(':
                new_word = '-LRB-'
            elif word == ')':
                new_word = '-RRB-'
            else:
                new_word = None
            if new_word is not None:
                node.cut_words = [(new_word, pos)]
    return lt


def add_top_node(lt: LabelTree) -> LabelTree:
    top_node = LabelNode(
        id=uuid.uuid4().hex,
        cut_words=lt.root.cut_words,
        label='TOP',
        extra=None
    )
    top_node.add_child(lt.root)
    lt.root.set_parent(top_node)
    lt.root = top_node
    return lt


def del_empty_node(lt: LabelTree) -> LabelTree:
    for node in lt.dfs(node=lt.root):
        if not node.cut_words:
            if node.parent:
                node.parent.children.remove(node)
        for w, p in node.cut_words:
            if not w.strip() or not p.strip():
                new_cut_words = [(w, p) for w, p in node.cut_words if w.strip() and p.strip()]
                if not new_cut_words:
                    node.parent.children.remove(node)
                else:
                    node.cut_words = new_cut_words
    return lt


def rename_label(lt: LabelTree) -> LabelTree:
    def _rename(label: str):
        """
        比如宾1,宾2,将其去掉1,2
        :param label:
        :return:
        """
        if not label:
            raise ValueError('存在错误')
        if label[-1].isdigit():
            label = label[:-1]
            return _rename(label=label)
        else:
            return label

    for node in lt.dfs(node=lt.root):
        node.label = _rename(node.label)
    return lt


def check_empty_node(lt: LabelTree) -> bool:
    for node in lt.dfs(node=lt.root):
        if not node.cut_words:
            return False
        for w, p in node.cut_words:
            if not w.strip() or not p.strip():
                return False
    return True
