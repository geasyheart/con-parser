## constituency parser

## 实验

此分为两个，一个仅使用word feature，一个使用word+tag feature。
根据训练结果，word feature跑出来的LF指标大致在80%多一点点，
word+tag feature能跑到84~85%。


```bash
# loss, metric:
1.219004377706207 UCM: 23.68% LCM: 15.43% UP: 90.96% UR: 90.09% UF: 90.52% LP: 85.72% LR: 84.90% LF: 85.30%

```

## 注意 
这里包含两份数据集，上面的结果是在开源的数据集上跑出来的结果
还有一份是公司标注的数据集，在company_data下。

运行须知：
1. 删除label_map.json, pos_map.json，会自动重新生成
2. 运行preprocess_4.py，这个是通过final_data.json生成final_nltk_data.txt
3. 运行preprocess_5.py，这个是将final_nltk_data.txt区分训练集和验证集

# 结果大概为 79