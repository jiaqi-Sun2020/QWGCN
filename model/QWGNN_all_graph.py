import numpy as np
from scipy.linalg import expm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from complexPyTorch.complexLayers import  ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d,complex_dropout




# 4. 量子 GCN 网络模型更新
class QuantumGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantumGNNLayer, self).__init__()
        self.fc = ComplexLinear(in_features, out_features)

    def forward(self, x,U_):

        #QGCNlayer
        x = torch.matmul(U_, x)  # 确保形状匹配
        x = self.fc(x)           # W没变！

        return x

class QuantumGNN_all_graph(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantumGNN_all_graph, self).__init__()
        self.qconv1 = QuantumGNNLayer(in_features, 16)
        self.qconv2 = QuantumGNNLayer(16, out_features)
    def forward(self, data):
        if isinstance(data, tuple):
            x, U_, batch = data
        else:
            raise ValueError("Expected a tuple of (x, U_, batch)")
        x = torch.tensor(x, dtype=torch.complex64).clone().detach()
        x = self.qconv1(x,U_)
        x = complex_relu(x)
        x = complex_dropout(x, training=self.training)
        x = self.qconv2(x, U_)
        # 全部转为实数
        # x = x.abs().to(torch.float64)
        x = x.abs()
        x = global_mean_pool(x, batch)  # 图池化
        x = F.log_softmax(x, dim=1)

        return x




if __name__ == "__main__":
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])
    model = QuantumGNNLayer(in_features=2, out_features=2)
    print(model)
