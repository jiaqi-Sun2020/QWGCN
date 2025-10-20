
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os
from torch_geometric.data import InMemoryDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from collections import Counter
import random

def balance_labels_by_oversampling(indices, dataset):
    labels = [dataset[i].y.item() for i in indices]
    label_count = Counter(labels)

    # 找出最大类别数量
    max_count = max(label_count.values())

    # 平衡后保存的索引和标签
    balanced_indices = indices.copy()
    balanced_labels = labels.copy()

    for label, count in label_count.items():
        if count < max_count:
            # 找出该类的样本索引
            class_indices = [i for i in indices if dataset[i].y.item() == label]
            needed = max_count - count
            # 随机复制该类样本索引
            replicated = random.choices(class_indices, k=needed)
            balanced_indices.extend(replicated)
            balanced_labels.extend([label] * needed)
            print(f"扩增类别 {label}：原数量={count}，已复制={needed}，总数量={count + needed}")

    return balanced_indices, balanced_labels



def dataset_split(args,indices,dataset):
    print(indices[0:10])
    print(dataset[0:10])
    if args.dataset_name == "PROTEINS_SMALL":  #由于数据集的不均均衡
        # 平衡样本
        indices, balanced_labels = balance_labels_by_oversampling(indices, dataset)

        # 70% 训练，30% 临时验证+测试
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
        )

        # 15% 验证，15% 测试
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
        )

        # 创建数据集
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    else:
        # 70% 训练，15% 验证，15% 测试
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42,
                                               stratify=[dataset[i].y.item() for i in indices])

        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                             stratify=[dataset[i].y.item() for i in temp_idx])

        # 创建数据加载器
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,val_loader,test_loader
