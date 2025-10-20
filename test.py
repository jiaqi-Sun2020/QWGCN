# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model.QGCN_all_graph import QuantumGCN
import torch.nn.functional as F
from utils.config import *
from utils.proprocess import proprocess_QGCN


# 训练函数
def train(args,):
    model.train()
    total_loss = 0
    for batch in loader:
        if args["model_type"]=="QuantumGCN":
            data = proprocess_QGCN(batch)
        if args["model_type"]=="GCN":
            data = batch

        optimizer.zero_grad()
        out = model(data)

        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 测试函数
def test(args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

def main(config_path = "./params.json"):
    args = json2args(config_path)

    dataset = TUDataset(root='./dataset', name='ENZYMES')
    print("INFO dataset.num_classes:{}".format(dataset.num_classes))
    print("INFO dataset.num_node_features:{}".format(dataset.num_node_features))
    loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN_layer(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(device)
    model = QuantumGCN(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 训练循环
    num_epochs = args["num_epochs"]
    for epoch in range(1, num_epochs + 1):
        loss = train(args)
        acc = test(args)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

    print("Training complete!")