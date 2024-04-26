import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_inputs=None, n_outputs=None):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(n_inputs, 256)  # 第一个隐藏层
        self.linear2 = nn.Linear(256, 128)  # 第二个隐藏层
        self.linear3 = nn.Linear(128, 64)  # 第三个隐藏层
        self.linear4 = nn.Linear(64, n_outputs)  # 输出层
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.linear4(out)
        return out



'''
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_inputs=None, n_outputs=None):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)  # 线性层，输出节点数为类别数

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out, dim=1)  # 使用softmax函数进行分类
        return out
'''







