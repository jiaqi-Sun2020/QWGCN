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
    # Load the ENZYMES dataset
    dataset = TUDataset(root='./', name='PROTEINS')

    # Filter graphs with 5 or fewer nodes
    small_graphs = [g for g in dataset if g.num_nodes <= 6]

    # Display results
    print(f"Total found {len(small_graphs)} Graphs with ≤6 nodes")

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
    Custom dataset class for processing and loading a filtered version of the PROTEINS dataset.

    This class loads the original PROTEINS dataset and filters graphs whose number of nodes
    is less than or equal to 5. If the processed data file does not exist, the process() method
    will be invoked to generate the processed data. The resulting data will be saved to the
    specified directory and can later be loaded via processed_paths[0].

    Procedure:
    1. Load the raw dataset: Check whether the processed data file already exists.
       If not, call the process() method.
    2. Call the process() method: In this step, the dataset is filtered and processed,
       and the results are saved to processed_paths[0].
    3. Load the processed data: In the __init__() method, the processed file is loaded
       from self.processed_paths[0], and the processed data are stored in self.data
       and self.slices.
    """
    # Load the original PROTEINS dataset

    dataset = TUDataset(root='./', name=dataset_name)

    # 筛选节点数小于等于5的图
    small_graphs = [g for g in dataset if g.num_nodes <= dataset_node]
    print(f"Filtered {len(small_graphs)} small graphs (number of nodes ≤ {dataset_node})")
    root =root+"_"+str(dataset_node)
    # 创建保存目录
    target_dir = os.path.join(root, 'processed')
    os.makedirs(target_dir, exist_ok=True)

    # Filter graphs with 5 or fewer nodes
    class SmallProteinDataset(InMemoryDataset):
        def __init__(self, root,transform=None, pre_transform=None):
            super(SmallProteinDataset, self).__init__(root, transform, pre_transform)

            # Check if the file exists
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
                    Returns the original dataset filename. Since the original data file is not required for this dataset, an empty list is returned.
            """
            return []

        @property
        def processed_file_names(self):
            """
                    Returns the filename of the processed data. This dataset contains only one processed file, 'data.pt'.
            """
            return ['data.pt']

        def download(self):
            """
                    The method for downloading the dataset. Since we already have the original data, this method is empty and does nothing.
            """
            pass  # 无需下载

        def process(self):
            """
            Process the dataset by filtering graphs whose number of nodes is less than or equal to 5
            and converting them into the (data, slices) format.

            1. Convert the filtered small graphs (by node count) into the (data, slices) format
               using the collate function.
            2. Save the converted data to the path specified by processed_paths[0].

            In this method, the processed data are saved as a serialized file for subsequent loading.
            """

            # # Use collate to convert a list of graphs into (data, slices)
            data, slices = self.collate(small_graphs)
            # Save as (data, slices) in the correct format.
            torch.save((data, slices), self.processed_paths[0])
            print(f"Data saved to:{self.processed_paths[0]}")

    # Save and create a new dataset
    new_dataset = SmallProteinDataset(root=root)
    print(f"The new dataset has been created and saved:{root}")

    return new_dataset

def select_dataset(args):

    if args.dataset_name in tudataset_names:
        dataset = TUDataset(root='./dataset', name=args.dataset_name)


    elif args.dataset_name == "PROTEINS_SMALL":
        dataset = build_small__dataset(root=os.path.join("./dataset/",args.dataset_name),dataset_name = "PROTEINS",dataset_node =args.dataset_node)



    else:
        print(f"Error loading dataset !")
    return dataset




if __name__ == '__main__':
    # check_dataset()
    # dataset_path = os.path.join("./", "PROTEINS_SMALL_6")
    # small_dataset = build_small__dataset(root=dataset_path,dataset_name = "PROTEINS",dataset_node=6)
    # print(small_dataset)

    dataset = TUDataset(root='./dataset', name=args.dataset_name)



