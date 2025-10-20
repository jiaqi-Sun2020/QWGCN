import numpy as np
from scipy.linalg import expm
import torch
import torch.nn as nn
import torch.optim as optim


# 4. 量子 GCN 网络模型更新
class QuantumGCNExtended(nn.Module):
    def __init__(self, in_features, out_features, A, t):
        super(QuantumGCNExtended, self).__init__()
        self.A = A
        self.t = t
        self.fc = nn.Linear(in_features, out_features)

    def unitary_dilation_operator(self):
        G_t = expm(-1j * self.A * self.t)  # 获取演化矩阵
        N = self.A.shape[0]
        I_n = np.eye(N)  # NxN 单位矩阵
        G_dagger = G_t.conj().T  # G(t) 的共轭转置
        # print(np.dot(G_t, G_dagger))
        eigenvalues = np.linalg.eigvals(np.dot(G_t, G_dagger))

        sqrt_max_eigenvalue = np.sqrt(np.max(eigenvalues))  # 获取最大特征值

        Left_top = G_t / sqrt_max_eigenvalue
        Right_low = -G_t / sqrt_max_eigenvalue

        Right_top = np.sqrt(I_n - np.dot(G_t / sqrt_max_eigenvalue, G_dagger / sqrt_max_eigenvalue))
        Left_low = Right_top

        U = np.block([[Left_top, Right_top],  # 上半部分
                      [Left_low, Right_low]])  # 下半部分
        return U[:N, :N]  # 演化 U
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        G_t = torch.tensor(self.unitary_dilation_operator(), dtype=torch.float32)

        x = torch.matmul(G_t, x)  # 确保形状匹配
        x = self.fc(x)
        return torch.relu(x)



if __name__ == "__main__":
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])
    model = QuantumGCNExtended(in_features=2, out_features=2, A=A, t=1.0)
    print(model)
