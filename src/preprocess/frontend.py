# -*- coding: utf8 -*-
#
from typing import List, Dict, Optional, Iterable


class FrontendNode(object):
    def __init__(
            self, words: str, type: str = '', label: str = '',
            clause: str = '', supply_subject: bool = False,
            child: List[Dict] = None
    ):
        """

        :param words: 长方形显示的字, 空表示没有主语
        :param type: 句型
        :param label: 成分，圆圈里的内容
        :param clause: 分句，椭圆里的内容
        :param supply_subject: 补充主语标识
        :param child:
        """
        self.words = words
        self.type = type
        self.label = label
        self.clause = clause
        self.supply_subject = supply_subject
        self.child = child or []

        self.check_node_valid()
        self.children: List['FrontendNode'] = []
        self.parent: 'FrontendNode' = None

    def add_child(self, node: 'FrontendNode'):
        self.children.append(node)

    def set_parent(self, node: 'FrontendNode'):
        self.parent = node

    def check_node_valid(self):
        """
        这里增加一个判断
        :return:
        """
        if (not self.label) and (not self.type) and (not self.clause):
            raise ValueError('无效的节点')
        # if self.type and self.clause:
        #     raise ValueError('句型和分句句型不应该同时存在,他俩互斥')
        # if self.clause and self.label:
        #     raise ValueError('分句句型不应该有成分')
        # if self.label and (self.type or self.clause):
        #     raise ValueError('成分有的情况下，不应该再有其他的')

    def __repr__(self):
        s = ''
        if self.type:
            s += f'句型：{self.type}\n'
        if self.label:
            s += f'成分：{self.label}\n'
        if self.clause:
            s += f'分句句型：{self.clause}\n'
        s += self.words
        return s

    def to_dict(self):
        return {
            'words': self.words,
            'type': self.type,
            'label': self.label,
            'clause': self.clause,
            'supplySubject': self.supply_subject,
            'child': [c.to_dict() for c in self.children],
        }


class FrontendTree(object):
    def __init__(self):
        self.root: Optional[FrontendNode] = None

    def generate_tree(self, sample: Dict):
        node_info = sample['data']
        self._generate(node_info=node_info, parent=None)

    def _generate(self, node_info: Dict, parent):
        node = FrontendNode(
            words=node_info['words'],
            type=node_info['type'],
            label=node_info['label'],
            clause=node_info['clause'],
            supply_subject=node_info['supplySubject'],
            child=node_info['child']
        )
        node.set_parent(node=parent)
        if parent is not None:
            parent.add_child(node=node)
        else:
            self.root = node

        for child in node_info['child']:
            self._generate(node_info=child, parent=node)

    def bfs(self, nodes: List[FrontendNode]):
        if nodes:
            yield nodes

        level_nodes = []
        for node in nodes:
            if node.children:
                level_nodes.extend(node.children)
        if level_nodes:
            yield from self.bfs(nodes=level_nodes)

    def bfs_by_single_node(self, nodes: List[FrontendNode]):
        if nodes:
            yield nodes
        for node in nodes:
            if node.children:
                yield from self.bfs_by_single_node(nodes=node.children)

    def dfs(self, node: Optional[FrontendNode] = None) -> Iterable[FrontendNode]:
        if node:
            yield node
        for child in node.children:
            yield from self.dfs(node=child)

    def pretty_tree(self, filename="", format="svg"):
        from graphviz import Digraph
        filename = filename or "tmp.gv"

        g = Digraph(format=format)
        for node in self.dfs(node=self.root):
            if node.supply_subject:
                g.node(name=str(id(node)), label=str(node), color='red')
            else:
                g.node(name=str(id(node)), label=str(node))

        for node in self.dfs(node=self.root):
            if node.parent:
                g.edge(str(id(node.parent)), str(id(node)))

        g.view(filename=filename, directory='./examples')

    def to_dict(self):
        """
        原封不动返回给前端
        :return:
        """
        return {'data': self.root.to_dict()}
