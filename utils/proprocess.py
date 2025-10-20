import numpy as np
import torch
import matplotlib
import numpy as np
from scipy.linalg import expm
import torch


def unitary_dilation_operator(A, t=1):  # 与GCN 信息交换相同的操作  A和t是根据输入的时间来计算的
    G_t = expm(-1j * A * t)  # 获取演化矩阵
    N = A.shape[0]
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
    # return U[:N, :N]  # 演化   U
    return U            #  全尺寸 U




def edge_i2Adj_M(edge_index,num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for src, dst in edge_index.t():  # 转置 edge_index 遍历
        adj_matrix[src, dst] = 1
    return adj_matrix


def proprocess_QGCN(data,device):

    x, edge_index, batch = data.x, data.edge_index, data.batch
    # print(x.device)
    num_nodes = data.x.shape[0]

    U_ = unitary_dilation_operator(edge_i2Adj_M(edge_index, num_nodes))
    U_ = torch.tensor(U_, dtype=torch.complex64,device = x.device).clone().detach()

    #对于数据 进行 复制一份进行冗余！
    x = torch.cat([data.x, data.x.clone()], dim=0)
    if hasattr(data, 'batch'):
        batch = torch.cat([data.batch, data.batch.clone()], dim=0)

    return (x.to(device) , U_.to(device),   batch.to(device))



def proprocess(model_name,batch,device):
    if model_name == "QWGNN":
        data = proprocess_QGCN(batch, device)

    elif model_name == "GCN":

        data = batch.to(device)
    else:
        data = batch.to(device)


    return data



