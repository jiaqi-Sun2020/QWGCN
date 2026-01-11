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

    # Find the maximum number of categories
    max_count = max(label_count.values())

    # Indexes and tags saved after balancing
    balanced_indices = indices.copy()
    balanced_labels = labels.copy()

    for label, count in label_count.items():
        if count < max_count:
            # Find the sample index of this class
            class_indices = [i for i in indices if dataset[i].y.item() == label]
            needed = max_count - count
            # Randomly copy the index of this type of sample
            replicated = random.choices(class_indices, k=needed)
            balanced_indices.extend(replicated)
            balanced_labels.extend([label] * needed)
            print(f"Augmenting class {label}: original={count}, duplicated={needed}, total={count + needed}")


    return balanced_indices, balanced_labels



def dataset_split(args,indices,dataset):
    print(indices[0:10])
    print(dataset[0:10])
    if args.dataset_name == "PROTEINS_SMALL":  #Due to the uneven balance of the dataset
        # Balanced Samples
        indices, balanced_labels = balance_labels_by_oversampling(indices, dataset)

        # 70% Training, 30% Validation + Testing
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
        )

        # 15% Validation, 15% Testing
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
        )

        # Build dataset
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        # Build DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    else:
        # 70% Training, 15% Validation, 15% Testing
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42,
                                               stratify=[dataset[i].y.item() for i in indices])

        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                             stratify=[dataset[i].y.item() for i in temp_idx])

        # Build DataLoader
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader,val_loader,test_loader
