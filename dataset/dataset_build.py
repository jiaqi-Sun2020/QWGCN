from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os
from torch_geometric.data import InMemoryDataset

tudataset_names = [
    'DD',
    'ENZYMES',
    'PROTEINS',
    'MUTAG',
    'NCI1',
    'NCI109',
    'PTC_MR',
    'PTC_FM',
    'PTC_FR',
    'PTC_MM',
    'BZR',
    'COX2',
    'COX2_MD',
    'BZR_MD',
    'IMDB-BINARY',
    'IMDB-MULTI',
    'REDDIT-BINARY',
    'REDDIT-MULTI-5K',
    'REDDIT-MULTI-12K',
    'COLLAB',
    'MSRC_21',
    'MSRC_21C',
    'MUTAG_188',
    'SYNTHETICnew',
    'SYNTHIE',
    'FIRSTMM_DB',
    'GITHUB_COLLAB',
    'IMDB-B',
    'IMDB-M',
    'REDDIT-B',
    'REDDIT-M5K',
    'REDDIT-M12K',
    'malonaldehyde',
    "TWITTER-Real-Graph-Partial"
]


def check_dataset():
    # 加载 ENZYMES 数据集
    dataset = TUDataset(root='./', name='PROTEINS')

    # 筛选出节点数小于等于 5 的图
    small_graphs = [g for g in dataset if g.num_nodes <= 6]

    # 显示结果
    print(f"总共找到 {len(small_graphs)} 张节点数 ≤6 的图")

    # 可视化前3张小图
    for i, graph in enumerate(small_graphs[:3]):
        edge_index = graph.edge_index.numpy()
        G = nx.Graph()
        G.add_edges_from(edge_index.T)

        plt.figure(figsize=(2, 2))
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"Graph {i} (Nodes: {graph.num_nodes}, Label: {graph.y.item()})")
        plt.show()





def build_small__dataset(root='./PROTEINS_SMALL',dataset_name = "PROTEINS",dataset_node = 6):
    """
    自定义数据集类，用于处理并加载经过筛选的PROTEINS数据集。

    加载原始数据集并筛选节点数小于等于5的图。如果处理后的数据文件不存在，则会调用 process() 方法
    来生成数据文件。处理后的数据将被保存到指定路径，之后可以通过 processed_paths[0] 加载这些
    已处理好的数据。

    过程：
    1. 加载原始数据集：检查是否已经存在处理后的数据文件。如果没有，调用 process() 方法。
    2. 调用 process() 方法：在 process() 方法中，数据被筛选和处理，结果保存到 processed_paths[0]。
    3. 加载处理后的数据：在 __init__() 方法中，通过 self.processed_paths[0] 加载数据文件，并将
       处理后的数据存储到 self.data 和 self.slices 中。
    """
    # 加载原始 PROTEINS 数据集
    dataset = TUDataset(root='./', name=dataset_name)

    # 筛选节点数小于等于5的图
    small_graphs = [g for g in dataset if g.num_nodes <= dataset_node]
    print(f"筛选出 {len(small_graphs)} 张小图（节点数≤{dataset_node}）")
    root =root+"_"+str(dataset_node)
    # 创建保存目录
    target_dir = os.path.join(root, 'processed')
    os.makedirs(target_dir, exist_ok=True)

    # 自定义一个数据集类用于保存处理后的数据
    class SmallProteinDataset(InMemoryDataset):
        def __init__(self, root,transform=None, pre_transform=None):
            super(SmallProteinDataset, self).__init__(root, transform, pre_transform)

            # 检查文件是否存在
            if not os.path.exists(self.processed_paths[0]):
                raise FileNotFoundError(f"Processed file not found: {self.processed_paths[0]}")

            # 加载数据
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except ValueError:
                print(f"Error loading file {self.processed_paths[0]}")
                raise

        @property
        def raw_file_names(self):
            """
                    返回原始数据集文件名。此数据集不需要原始数据文件，因此返回空列表。
            """
            return []

        @property
        def processed_file_names(self):
            """
                    返回处理后的数据文件名。此数据集只包含一个处理后的文件 'data.pt'。
            """
            return ['data.pt']

        def download(self):
            """
                    下载数据集的方法。由于我们已经有了原始数据，因此此方法为空，不执行任何操作。
            """
            pass  # 无需下载

        def process(self):
            """
                    处理数据集，将节点数小于等于5的图筛选并转换为 (data, slices) 格式。

                    1. 将小图数据（节点数）通过 collate 函数转换为 (data, slices) 格式。
                    2. 将转换后的数据保存到 processed_paths[0] 路径。

                    在此方法中，数据被保存并存储为处理后的文件，供后续加载。
            """
            # 使用 collate 将图列表转为 (data, slices)
            data, slices = self.collate(small_graphs)
            # 保存为正确格式的 (data, slices)
            torch.save((data, slices), self.processed_paths[0])
            print(f"数据保存至：{self.processed_paths[0]}")

    # 保存并创建新的数据集
    new_dataset = SmallProteinDataset(root=root)
    print(f"新数据集已创建并保存在：{root}")

    return new_dataset

def select_dataset(args):

    if args.dataset_name in tudataset_names:
        dataset = TUDataset(root='./dataset', name=args.dataset_name)


    elif args.dataset_name == "PROTEINS_SMALL":
        dataset = build_small__dataset(root=os.path.join("./dataset/",args.dataset_name),dataset_name = "PROTEINS",dataset_node =args.dataset_node)



    else:
        print(f"Error loading dataset!")
    return dataset




if __name__ == '__main__':
    # check_dataset()
    # dataset_path = os.path.join("./", "PROTEINS_SMALL_6")
    # small_dataset = build_small__dataset(root=dataset_path,dataset_name = "PROTEINS",dataset_node=6)
    # print(small_dataset)

    dataset = TUDataset(root='./dataset', name=args.dataset_name)



