# -*- coding: utf8 -*-
#
import os

import ujson

all_path = [
    '/home/yuzhang/文档/句子结构标注文档/句子标注-第一批-部分完成数据',
    '/home/yuzhang/文档/句子结构标注文档/句子标注-第二批-国务院-数据（修改版）',
    '/home/yuzhang/文档/句子结构标注文档/句子标注-第三批-国务院-数据',
    '/home/yuzhang/文档/句子结构标注文档/句子标注-第四批-国务院-数据',
    '/home/yuzhang/文档/句子结构标注文档/05-句子标注-第五批',
    '/home/yuzhang/文档/句子结构标注文档/第六批',
]


def get_all_json_files(directory):
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            if not filename.endswith('json'): continue
            abs_path = os.path.join(dirpath, filename)
            assert os.path.exists(abs_path)
            yield abs_path


def merge_all_files():
    all_data = []
    all_sentence = {}

    all_size, dump_data_size, is_need_modify_size = 0, 0, 0
    # 按照新的来
    for path in all_path[::-1]:
        for json_file in get_all_json_files(directory=path):
            with open(json_file) as f:
                content = f.read()
                if content.startswith(u'\ufeff'):
                    content = content.encode('utf8')[3:].decode('utf8')
                data = ujson.loads(content)
                # 去重
                for item in data['data']:
                    all_size += 1

                    if item['sentence'] in all_sentence:
                        dump_data_size += 1
                        continue
                    if item.get('isNeedRevise', 'N') != "N" or item.get('isMarkDoubt', 'N') != "N":
                        is_need_modify_size += 1
                        continue
                    all_sentence[item['sentence']] = None
                    all_data.append(item)

    #
    with open('/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_data.json', 'w') as f:
        f.write(
            ujson.dumps({"data": all_data})
        )
    print(f'all_size: {all_size} dump_data_size: {dump_data_size} is_need_modify_size: {is_need_modify_size}')


if __name__ == '__main__':
    merge_all_files()
