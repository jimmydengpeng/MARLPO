# MARL Policy Optimization


## Intro

## installation
Requirements:
```
Ray[RLlib] == 2.3.0
```

## TODO
- [ ] Linux(Unbuntu) deployment modification

Xavier初始化方法是一种常用的参数初始化方法，旨在使网络的输出具有相对较小的方差，从而避免在训练过程中梯度消失或爆炸的问题。该方法将权重初始化为从均匀分布 $U[-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}]$ 中随机抽取的值，其中 $n_{in}$ 和 $n_{out}$ 分别表示权重的输入和输出维度。