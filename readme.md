## constituency parser

## 实验

此分为两个，一个仅使用word feature，一个使用word+tag feature。
根据训练结果，word feature跑出来的LF指标大致在80%多一点点，
word+tag feature能跑到84~85%。


```bash
# loss, metric:
1.219004377706207 UCM: 23.68% LCM: 15.43% UP: 90.96% UR: 90.09% UF: 90.52% LP: 85.72% LR: 84.90% LF: 85.30%

```
