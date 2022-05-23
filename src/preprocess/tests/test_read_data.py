# -*- coding: utf8 -*-
#
import json
import os.path
import string
from unittest import TestCase

import pandas as pd
import requests

from src.preprocess import frontend
from src.preprocess.preprocess_tree import get_without_supply_subject_tree, get_without_empty_node, rebuild_ft, \
    ft_add_segment

work_dir = '/home/yuzhang/文档/句子结构标注文档/第六批'

short_sentence_df = pd.DataFrame()


def check_preprocess_result(sentence: str, preprocess_result: str, is_need_revise):
    """
    检查预处理结果
    :param sentence:
    :param preprocess_result:
    :param is_need_revise: 需要修正
    :return:
    """
    global short_sentence_df
    tmp = {index: char for index, char in enumerate(sentence)}
    result = requests.get(f'http://localhost:3666/phrase-extraction?sentence={sentence}').json()
    for type_, items in result.items():
        for item in items:
            for index in item['indices']:
                tmp[index] = None
    pop_result = "".join([v for v in tmp.values() if v is not None])
    if not pop_result:
        return
    if pop_result[0] in string.punctuation or pop_result[0] in ('，', '。'):
        pop_result = pop_result[1:]
    if is_need_revise == 'N':
        if pop_result != preprocess_result:
            short_sentence_df = short_sentence_df.append(
                {"sentence": sentence, "p1": preprocess_result, "p2": pop_result},
                ignore_index=True)
    # if is_need_revise == 'N':
    # 表示不需要修正，但是我看了下，有些地方不一定不对
    # if pop_result != preprocess_result:
    #     print('人家不需要修正哦，你的结果不对')
    # else:
    # 表示需要修正，看下自己跑出来的结果呢
    # print(pop_result)


def read_all_file(name):
    total_size, leave_size = 0, 0
    all_single_file = os.path.join(work_dir, '全部', f'{name}_全部标注.json')
    with open(all_single_file) as f:
        content = f.read()
        if content.startswith(u'\ufeff'):
            content = content.encode('utf8')[3:].decode('utf8')
        items = json.loads(content)
        for item in items['data']:
            check_preprocess_result(
                sentence=item['sentence'],
                preprocess_result=item['预处理结果'],
                is_need_revise=item['isNeedRevise']
            )
            # 检查是否进行标注了
            if not item['data']['child']:
                # print(f"[没有标注] {item['sentence']}")
                continue
            ft = frontend.FrontendTree()
            ft.generate_tree(item)
            # ft.pretty_tree(filename='tt')

            # 过滤掉需要主语补齐的
            ft = get_without_supply_subject_tree(ft=ft)
            # 过滤掉没有主语的
            ft = get_without_empty_node(ft=ft)

            # 句型、分句句型、成分 产生新的node
            ft = rebuild_ft(ft=ft)

            total_size += 1
            # 增加分词节点
            ft = ft_add_segment(ft=ft)
            if not ft:continue
            leave_size += 1
        print(total_size, leave_size)
            # ft.pretty_tree(filename='tt')


def read_mark_doubt_file(name):
    mark_doubt_file = os.path.join(work_dir, '标注存疑', f'{name}_标注存疑文件.json')


def read_need_revise_file(name):
    need_revise_file = os.path.join(work_dir, '需要修正', f'{name}_需要修正文件.json')
    with open(need_revise_file) as f:
        content = f.read()
        if content.startswith(u'\ufeff'):
            content = content.encode('utf8')[3:].decode('utf8')
        items = json.loads(content)
        for item in items['data']:
            check_preprocess_result(
                sentence=item['sentence'],
                preprocess_result=item['预处理结果'],
                is_need_revise=item['isNeedRevise']
            )
            # 检查是否进行标注了
            if not item['data']['child']:
                print(f"[没有标注] {item['sentence']}")
                continue
            ft = frontend.FrontendTree()
            ft.generate_tree(item)
            ft.pretty_tree(filename='tt')
            print('h')


class TestSample(TestCase):
    def test_read_data(self):
        read_all_file('GWY-yw-123')
        # read_need_revise_file('GWY-yw-123')
        print('h')
# [没有标注] 三、深化粤港澳民生领域合作，全面建设宜居宜业优质生活圈
