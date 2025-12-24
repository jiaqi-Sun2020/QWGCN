# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later
import os.path

import numpy as np
import torch
import matplotlib
from torch_geometric.datasets import TUDataset

from dataset.dataset_split import dataset_split
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from utils.config import *
from utils.proprocess import proprocess
from utils.save import save_checkpoint
from utils.save_log import setup_logger
from utils.data_analysis import analyze_model
from dataset.dataset_build import *
from  model.model_get import get_model
import matplotlib.pyplot as plt
import time


def train(args):
    setup_logger(log_dir=args.save_path)

    # 保存参数为 config.json
    args2json(args, os.path.join(args.save_path,"config.json"))

    # 1. 加载完整数据集
    dataset = select_dataset(args)
    print("INFO dataset.num_classes:{}".format(dataset.num_classes))
    print("INFO dataset.num_node_features:{}".format(dataset.num_node_features))
    assert dataset.num_node_features > 0, "Error: The node feature length in the graph is 0! Please note"
    num_graphs = len(dataset)
    indices = list(range(num_graphs))


    # print([dataset[i].y.item() for i in indices])

    train_loader,val_loader,test_loader=dataset_split(args,indices,dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    model = get_model(args, dataset.num_node_features, dataset.num_classes,device).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.num_epochs)
    torch.autograd.set_detect_anomaly(True)
    # 初始化记录列表和最佳准确率
    accuracy_list = []
    train_loss_list = []
    val_loss_list = []
    f1_score_list = []
    mse_list = []
    mae_list = []
    val_acc_list = []
    best_val_acc = 0.0  # 🔧 修复：初始化最佳验证准确率
    patience_counter = 0

    start_time = time.time()  #开始时间
    # 训练循环
    num_epochs = args.num_epochs
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch in train_loader:

            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print("🚨 原始 batch.x 就有 NaN 或 Inf！！")
                print("NaN 数量:", torch.isnan(batch.x).sum().item())
                print("Inf 数量:", torch.isinf(batch.x).sum().item())
                print("范围:", batch.x.min().item(), batch.x.max().item())
                torch.save(batch, "raw_batch_with_nan.pt")

            data = proprocess(args.model, batch, device)

            optimizer.zero_grad()
            out = model(data)

            # 🔧 修复：统一设备处理
            batch_y = batch.y.to(device)
            loss = F.nll_loss(out, batch_y)
            loss.backward()
            optimizer.step()

            # 动态 batch 大小
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)

            # 计算加权 loss 总和
            total_loss += loss.item() * batch_size
            pred = out.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total_samples += batch_size

        # 平均 loss（考虑 batch size 权重）
        avg_loss = total_loss / total_samples
        avg_acc = correct / total_samples



        print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        train_loss_list.append(avg_loss)
        accuracy_list.append(avg_acc)

        # 每 N 轮保存一次评估结果 进行验证
        if epoch % int(args.save_freq) == 0 or epoch == num_epochs:
            model.eval()
            preds = []
            trues = []
            val_correct = 0
            val_total_samples = 0
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    data = proprocess(args.model, batch, device)
                    out = model(data)
                    pred = out.argmax(dim=1)
                    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)

                    # 🔧 修复：统一设备处理
                    batch_y = batch.y.to(device)
                    loss = F.nll_loss(out, batch_y)
                    val_loss += loss.item()
                    val_correct += (pred == batch_y).sum().item()
                    val_total_samples += batch_size  # 🔧 修复：累加总样本数

                    preds.append(pred.cpu())
                    trues.append(batch_y.cpu())

            # 🔧 修复：正确计算验证准确率
            val_acc = val_correct / val_total_samples
            val_loss /= val_total_samples
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)
            preds = torch.cat(preds, dim=0)
            trues = torch.cat(trues, dim=0)
            assert preds.shape == trues.shape, "Prediction and label shape mismatch!"

            print("trues:", trues.numpy())
            print("preds:", preds.numpy())

            f1 = f1_score(trues.numpy(), preds.numpy(), average='macro')
            mse = mean_squared_error(trues.numpy(), preds.numpy())
            mae = mean_absolute_error(trues.numpy(), preds.numpy())

            f1_score_list.append(f1)
            mse_list.append(mse)
            mae_list.append(mae)
            val_acc_list.append(val_acc)

            print(f"Epoch {epoch:03d} | | LOSS: {val_loss:.4f} | F1 Score: {f1:.8f} | MSE: {mse:.8f} | MAE: {mae:.8f} | ACC: {val_acc:.8f} | Current LR: {optimizer.param_groups[0]['lr']}")

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                save_checkpoint(model, optimizer, epoch, args.save_path, name="best")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f'早停于第 {epoch} 轮，最佳验证准确率: {best_val_acc:.4f}')
                break

        print(f"平均推理时间: {(time.time()-start_time) * 1000:.2f} ms")
        start_time = time.time()


    # 保存最后的模型
    save_checkpoint(model, optimizer, epoch, args.save_path, name="last")

    # 🔧 修复：测试集评估
    model.eval()
    preds = []
    trues = []
    eval_correct = 0
    test_total_samples = 0  # 🔧 修复：记录测试集总样本数

    with torch.no_grad():
        for batch in test_loader:
            data = proprocess(args.model, batch, device)
            out = model(data)
            pred = out.argmax(dim=1)
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)

            # 🔧 修复：统一设备处理
            batch_y = batch.y.to(device)
            eval_correct += (pred == batch_y).sum().item()
            test_total_samples += batch_size  # 🔧 修复：累加总样本数

            preds.append(pred.cpu())
            trues.append(batch_y.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    # 🔧 修复：正确计算测试准确率
    eval_acc = eval_correct / test_total_samples
    f1 = f1_score(trues.numpy(), preds.numpy(), average='macro')
    mse = mean_squared_error(trues.numpy(), preds.numpy())
    mae = mean_absolute_error(trues.numpy(), preds.numpy())
    cm = confusion_matrix(trues.numpy(), preds.numpy())

    print(f"Final Test Results | F1 Score: {f1:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | ACC: {eval_acc:.4f} ")
    print("Confusion Matrix:")
    print(cm)
    print("===================")
    analyze_model(train_loss_list,val_loss_list, f1_score_list, mse_list, mae_list, cm,val_acc_list,args)
