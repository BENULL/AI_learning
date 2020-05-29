# coding: utf-8

from IPython import display
from matplotlib import pyplot as plt
import random
import torch

# 3.3 linear regression
# 设置显示svg图
def use_svg_display():
    display.set_matplotlib_formats('svg')


# 设置图片尺寸
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 读取数据
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)

# 定义模型 linear regression
def linreg(X,w,b):
    return torch.mm(X, w)+b

# 定义损失函数 平方损失函数
def squared_loss(y,y_hat):
    return (y_hat-y.view(y_hat.size()))**2/2

# 定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data 不记录到梯度计算


