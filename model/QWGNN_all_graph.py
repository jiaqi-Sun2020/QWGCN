# ============================================================
# Copyright (C) 2025 Jiaqi Sun
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Author: Jiaqi Sun
# Affiliation: JNU Quantum Physics Lab
# Email: 1623958071@qq.com
#
# Supervisor: Zhihao Bian
# Email: zhihaobian@jiangnan.edu.cn
#
# Description:
#   This file is part of the research codebase developed at
#   JNU Quantum Physics Lab, focusing on quantum-inspired
#   graph learning and complex-valued neural networks.
#
# ============================================================

import numpy as np
from scipy.linalg import expm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_dropout

class QuantumGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = ComplexLinear(in_features, out_features)

    def forward(self, x, U_):
        x = torch.matmul(U_, x)
        x = self.fc(x)
        return x


class QuantumGNN_all_graph(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantumGNN_all_graph, self).__init__()
        self.qconv1 = QuantumGNNLayer(in_features, 16)
        self.qconv2 = QuantumGNNLayer(16, 16)
        self.qconv3 = QuantumGNNLayer(16, out_features)
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
        x = complex_relu(x)
        x = complex_dropout(x, training=self.training)

        x = self.qconv3(x, U_)

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
    ], dtype=np.float32)

    N = A.shape[0]

    H = A
    t = 1.0
    U = expm(-1j * H * t)
    U = torch.tensor(U, dtype=torch.complex64)

    x = torch.randn(N, 2, dtype=torch.complex64)
    batch = torch.zeros(N, dtype=torch.long)

    model = QuantumGNN_all_graph(
        in_features=2,
        out_features=2,
        num_nodes=N
    )

    out = model((x, U, batch))
    print(out)
