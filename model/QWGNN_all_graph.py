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
    def __init__(self, in_features, out_features, num_nodes):
        super().__init__()

        self.qconv1 = QuantumGNNLayer(in_features, 16)
        self.qconv2 = QuantumGNNLayer(16, out_features)

        # ===== 可学习量子初态 P =====
        self.p_amp = nn.Parameter(torch.ones(num_nodes, 1))
        self.p_phase = nn.Parameter(torch.zeros(num_nodes, 1))

    def get_initial_state(self):
        """
        ψ(0) = P
        """
        psi0 = self.p_amp * torch.exp(1j * self.p_phase)
        psi0 = psi0 / torch.norm(psi0)   # 归一化（非常重要）
        return psi0                      # [N, 1]

    def forward(self, data):
        if isinstance(data, tuple):
            x, U_, batch = data
        else:
            raise ValueError("Expected (x, U_, batch)")

        x = x.to(torch.complex64)

        # ===== 量子初态 =====
        psi = self.get_initial_state()   # [N, 1]

        # ===== 第一次量子行走 =====
        psi = torch.matmul(U_, psi)      # ψ¹ = U ψ⁰
        x = x * psi                      # 用量子态调制特征

        x = self.qconv1(x, U_)
        x = complex_relu(x)
        x = complex_dropout(x, training=self.training)

        # ===== 第二次量子行走 =====
        psi = torch.matmul(U_, psi)      # ψ² = U ψ¹
        x = x * psi

        x = self.qconv2(x, U_)

        # ===== 测量 =====
        x = x.abs()
        x = global_mean_pool(x, batch)
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
