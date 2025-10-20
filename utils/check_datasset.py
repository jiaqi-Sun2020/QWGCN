from torch_geometric.datasets import TUDataset

# 加载 PROTEINS 数据集
dataset = TUDataset(root='../dataset', name='PROTEINS')

# 每个图的节点数
num_nodes_list = [data.num_nodes for data in dataset]

# 最大节点数
max_nodes = max(num_nodes_list)

print("PROTEINS 数据集图的数量:", len(dataset))
print("最大节点数:", max_nodes)
print("平均节点数:", sum(num_nodes_list)/len(num_nodes_list))
