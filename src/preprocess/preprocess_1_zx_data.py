# -*- coding: utf8 -*-
#
import os.path

import ujson

workdir = '/home/yuzhang/文档/句子结构标注文档/zx'

all_data_path = os.path.join(workdir, 'all_data.json')

# 第六批数据
result_6_data_path = os.path.join(workdir, 'result_6_new.json')


def pop_dup_data():
    """
    如果出现了重复的句子，以新的一批数据为准，
    :return:
    """
    with open(all_data_path) as f:
        all_data = ujson.loads(f.read())

    with open(result_6_data_path) as f:
        result_6_data = ujson.loads(f.read())

    final_data = []
    for _r6data in result_6_data:
        _r6data['cut_words'] = eval(_r6data['cut_words'])
        final_data.append(_r6data)

    final_data_sentence = [i['sentence'] for i in final_data]

    for _all_data in all_data:
        if _all_data['sentence'] not in final_data_sentence:
            _all_data['cut_words'] = eval(_all_data['cut_words'])
            final_data.append(_all_data)
    with open('/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_zx_data.json', 'w') as f:
        f.write(ujson.dumps({"data": final_data}))


if __name__ == '__main__':
    pop_dup_data()
