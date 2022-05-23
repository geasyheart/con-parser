# -*- coding: utf8 -*-
# 比较张翔给的两部分数据有什么区别
import ujson

final_data_path = '/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_data.json'
final_zx_data_path = '/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/final_zx_data.json'

# 判断两者句子有多少差异

with open(final_data_path, 'r') as f:
    final_data = ujson.loads(f.read())

with open(final_zx_data_path, 'r') as f:
    final_zx_data = ujson.loads(f.read())

final_data_sentence = [item['sentence'] for item in final_data['data']]

same_size = 0
all_valid_size = 0
for final_zx in final_zx_data['data']:
    sentence = final_zx['sentence']
    if final_zx.get('isNeedRevise', 'N') != "N" or final_zx.get('isMarkDoubt', 'N') != "N":
        continue
    all_valid_size += 1
    if sentence in final_data_sentence:
        same_size += 1

# final_data_len: 31537, zx valid data: 27502, same size : 26684
print(f'final_data_len: {len(final_data_sentence)}, zx valid data: {all_valid_size}, same size : {same_size}')

# 第二部分

with open('/home/yuzhang/PycharmProjects/con-parser/src/data/company_data/view_data.json', 'w') as f:
    view_data = final_data['data'][:100]
    f.write(ujson.dumps({"data": view_data}))
