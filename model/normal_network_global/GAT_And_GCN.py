import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import torch.nn as nn
import math


class GAT_And_GCN_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()

        # 多尺度特征提取
        self.input_dim = in_features
        self.hidden_dim = 128
        self.num_heads = 8

        # 输入投影层
        self.input_projection = nn.Linear(in_features, self.hidden_dim)
        self.input_norm = BatchNorm(self.hidden_dim)

        # 多层注意力机制 + GCN混合架构
        self.gat1 = GATConv(self.hidden_dim, self.hidden_dim // self.num_heads,
                            heads=self.num_heads, dropout=0.1, concat=True)
        self.bn1 = BatchNorm(self.hidden_dim)

        self.gcn1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.gcn_bn1 = BatchNorm(self.hidden_dim)

        self.gat2 = GATConv(self.hidden_dim, self.hidden_dim // self.num_heads,
                            heads=self.num_heads, dropout=0.1, concat=True)
        self.bn2 = BatchNorm(self.hidden_dim)

        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.gcn_bn2 = BatchNorm(self.hidden_dim)

        # 深层特征提取
        self.conv3 = GraphConv(self.hidden_dim, self.hidden_dim)
        self.bn3 = BatchNorm(self.hidden_dim)

        self.conv4 = GCNConv(self.hidden_dim, self.hidden_dim // 2)
        self.bn4 = BatchNorm(self.hidden_dim // 2)

        # 残差连接的线性层
        self.residual_proj = nn.Linear(self.hidden_dim, self.hidden_dim // 2)

        # 多尺度池化
        self.pool_projection = nn.Linear(self.hidden_dim // 2 * 3, self.hidden_dim)

        # 高级分类头 - 使用注意力机制
        self.attention_weights = nn.Linear(self.hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(self.hidden_dim // 8, out_features)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.input_dim:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 输入投影和归一化
        x = self.input_projection(x)
        x = F.relu(self.input_norm(x))
        identity1 = x

        # 第一层：GAT + 残差连接
        x_gat1 = self.gat1(x, edge_index)
        x_gat1 = F.elu(self.bn1(x_gat1))
        x_gat1 = F.dropout(x_gat1, p=0.1, training=self.training)

        # 第一层：GCN分支
        x_gcn1 = self.gcn1(identity1, edge_index)
        x_gcn1 = F.elu(self.gcn_bn1(x_gcn1))

        # 特征融合
        x = x_gat1 + x_gcn1 + identity1  # 三路残差连接
        identity2 = x

        # 第二层：GAT
        x_gat2 = self.gat2(x, edge_index)
        x_gat2 = F.elu(self.bn2(x_gat2))
        x_gat2 = F.dropout(x_gat2, p=0.1, training=self.training)

        # 第二层：GCN分支
        x_gcn2 = self.gcn2(identity2, edge_index)
        x_gcn2 = F.elu(self.gcn_bn2(x_gcn2))

        # 特征融合
        x = x_gat2 + x_gcn2 + identity2

        # 第三层：GraphConv
        x = self.conv3(x, edge_index)
        x = F.elu(self.bn3(x))
        x = F.dropout(x, p=0.15, training=self.training)

        # 第四层：最终卷积 + 残差
        residual = self.residual_proj(identity2)
        x = self.conv4(x, edge_index)
        x = F.elu(self.bn4(x))
        x = x + residual  # 残差连接

        # 多尺度图池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        # 池化特征融合
        x_pooled = torch.cat([x_mean, x_max, x_sum], dim=1)
        x_pooled = self.pool_projection(x_pooled)
        x_pooled = F.relu(x_pooled)
        x_pooled = F.dropout(x_pooled, p=0.2, training=self.training)

        # 注意力加权 (可选的全局注意力机制)
        attention_scores = torch.sigmoid(self.attention_weights(x_pooled))
        x_pooled = x_pooled * attention_scores

        # 分类
        x = self.classifier(x_pooled)

        return F.log_softmax(x, dim=1)

if __name__ == "__main__":

    model = GAT_And_GCN_layer(in_features=2, out_features=2)
    print(model)