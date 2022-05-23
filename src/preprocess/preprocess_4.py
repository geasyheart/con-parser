# -*- coding: utf8 -*-
#
import ujson
from constituency_labeling.convert import label_tree_to_nltk

from src.preprocess import frontend
from src.preprocess.preprocess_tree import get_without_supply_subject_tree, get_without_empty_node, rebuild_ft, \
    ft_add_segment, fillup_symbols_tree, lt_add_leaf_node, convert_special_symbol, add_top_node, del_empty_node

final_data_path = '/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_data.json'

save_path = '/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_nltk_data.txt'
all_labels = set()
poped_labels = set()


def read_all_file():
    save_nltk_f = open(save_path, 'w')
    # 总数量
    total_size = 0
    # 需要修正
    is_need_revise = 0
    # 没有标注
    not_is_labeled = 0
    # 标注顺序反了
    reverse_err_size = 0
    # 分词边界错误
    cut_word_err_size = 0
    with open(final_data_path) as f:
        content = f.read()
        if content.startswith(u'\ufeff'):
            content = content.encode('utf8')[3:].decode('utf8')
        items = ujson.loads(content)
        for item in items['data']:
            # check_preprocess_result(
            #     sentence=item['sentence'],
            #     preprocess_result=item['预处理结果'],
            #     is_need_revise=item['isNeedRevise']
            # )
            total_size += 1

            if item['isNeedRevise'] != "N":
                is_need_revise += 1
                continue
            # 检查是否进行标注了
            if not item['data']['child']:
                # print(f"[没有标注] {item['sentence']}")
                not_is_labeled += 1
                continue
            ft = frontend.FrontendTree()
            ft.generate_tree(item)

            # 过滤掉需要主语补齐的
            ft = get_without_supply_subject_tree(ft=ft)
            # 过滤掉没有主语的
            ft = get_without_empty_node(ft=ft)

            # 句型、分句句型、成分 产生新的node
            ft = rebuild_ft(ft=ft)

            # 补齐符号，标注的数据集中比如，。
            ft = fillup_symbols_tree(tree=ft)
            if ft is None:
                reverse_err_size += 1
                continue
            # 增加分词节点
            try:
                lt = ft_add_segment(ft=ft)
            except ValueError:
                cut_word_err_size += 1
                # debug
                # lt = ft_add_segment(ft=ft)
                continue
            # 叶子节点增加分词节点
            lt = lt_add_leaf_node(lt=lt)
            # 需要将()转成另外一个符号
            lt = convert_special_symbol(lt=lt)

            lt = del_empty_node(lt=lt)
            # 增加top节点
            lt = add_top_node(lt=lt)

            # # 遍历所有的label
            # for node in lt.dfs(node=lt.root):
            #     all_labels.add(node.label)
            # 如果需要宾1,宾2这种需要区分很细标签的，则把下面注释掉
            # 重新命名label，减少label数量
            # lt = rename_label(lt=lt)
            # for node in lt.dfs(node=lt.root):
            #     poped_labels.add(node.label)
            nltk_tree = label_tree_to_nltk(label_tree=lt)
            # nt_to_str = str(nltk_tree).replace("\n", '')
            nt_to_str = nltk_tree._pformat_flat("", "()", False)
            # check
            # nt = nltk.tree.Tree.fromstring(nt_to_str)
            # l = nltk_tree_to_label(nt)
            # assert check_empty_node(l)
            save_nltk_f.write(nt_to_str + '\n')
    print(
        "总量： ", total_size,
        "标注修要修改： ", is_need_revise,
        "没有标注： ", not_is_labeled,
        "补齐符号错误： ", reverse_err_size,
        '分词边界错误： ', cut_word_err_size
    )
    save_nltk_f.close()
    # print(all_labels)
    # print(poped_labels)


if __name__ == '__main__':
    read_all_file()
