# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
#
# def analyze_model(losses, f1_scores, mses, maes, cm,val_acc_list,args):
#     """
#     综合分析模型训练过程：
#     - Loss 曲线
#     - F1、MSE、MAE 曲线
#     - 混淆矩阵图像
#     - 指标统计与趋势分析
#     """
#
#     os.makedirs(args.save_path, exist_ok=True)
#
#     # ---------- 1. Loss 曲线 ----------
#     losses = np.array(losses)
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(losses)+1), losses, marker='o', label='Loss', color='blue')
#
#     window_size = min(10, len(losses)//5) or 1
#     smooth_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
#     plt.plot(range(window_size, len(losses)+1), smooth_losses, label=f'Smoothed Loss (window={window_size})', color='orange')
#
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'{args.model} Training Loss Curve')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     loss_path = os.path.join(args.save_path, 'loss_curve.png')
#     plt.savefig(loss_path)
#     plt.close()
#     print(f"✅ Loss 曲线已保存至: {loss_path}")
#
#     # ---------- 2. F1 / MSE / MAE 曲线 ----------
#     if isinstance(f1_scores, list) and len(f1_scores) == len(losses):
#         plt.figure(figsize=(10, 6))
#         plt.plot(f1_scores, label='F1 Score', marker='o')
#         plt.plot(mses, label='MSE', marker='x')
#         plt.plot(maes, label='MAE', marker='s')
#         plt.xlabel('Epoch')
#         plt.title('Model Evaluation Metrics')
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         metrics_path = os.path.join(args.save_path, 'metrics_curve.png')
#         plt.savefig(metrics_path)
#         plt.close()
#         print(f"✅ F1 / MSE / MAE 曲线已保存至: {metrics_path}")
#
#     # ---------- 3. 混淆矩阵 ----------
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     cm_path = os.path.join(args.save_path, 'confusion_matrix_final.png')
#     plt.savefig(cm_path)
#     plt.close()
#     print(f"✅ 混淆矩阵已保存至: {cm_path}")
#
#     # ---------- 4. 指标汇总分析 ----------
#     print("\n📊 指标统计与趋势分析")
#     print(f"- 最初 Loss: {losses[0]:.4f}")
#     print(f"- 最终 Loss: {losses[-1]:.4f}")
#     print(f"- 最佳 Epoch (最低 Loss): {np.argmin(losses) + 1}, Loss={np.min(losses):.4f}")
#     print(f"- 最终 F1 Score: {f1_scores[-1]:.4f}")
#     print(f"- 最终 MSE: {mses[-1]:.4f}")
#     print(f"- 最终 MAE: {maes[-1]:.4f}")
#
#     if f1_scores[-1] > 0.8:
#         print("✅ F1 很高，模型区分效果良好。")
#     elif f1_scores[-1] > 0.6:
#         print("⚠️ F1 中等，模型有一定效果但可能可优化。")
#     else:
#         print("❌ F1 偏低，建议检查模型结构或特征。")
#
#     if losses[-1] < losses[0] * 0.5:
#         print("✅ Loss 明显下降，训练收敛良好。")
#     elif losses[-1] < losses[0] * 0.9:
#         print("⚠️ Loss 有下降，但收敛较慢。")
#     else:
#         print("❌ Loss 几乎未下降，可能存在欠拟合。")


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# def analyze_model(trian_losses,val_losses,val_f1_scores, val_mses, val_maes, cm, val_acc_list, args):
#     """
#     综合分析模型训练过程：
#     - Loss 曲线
#     - F1、MSE、MAE、Accuracy 分别生成独立图像
#     - 混淆矩阵图像
#     - 指标统计与趋势分析
#     """
#
#     # 确保保存路径存在
#     os.makedirs(args.save_path, exist_ok=True)
#
#     # 数据预处理和验证
#     losses = np.array(trian_losses) if not isinstance(trian_losses, np.ndarray) else trian_losses
#
#     # 处理可能为空的指标列表
#     has_f1 = val_f1_scores is not None and len(val_f1_scores) > 0
#     has_mse = val_mses is not None and len(val_mses) > 0
#     has_mae = val_maes is not None and len(val_maes) > 0
#     has_acc = val_acc_list is not None and len(val_acc_list) > 0
#
#     print("📊 开始生成模型分析报告...")
#
#     # ---------- 1. Loss 曲线 ----------
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss', color='blue', linewidth=2)
#
#     # 平滑曲线
#     if len(losses) > 5:
#         window_size = min(10, len(losses) // 5) or 1
#         if window_size > 1:
#             smooth_losses = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
#             plt.plot(range(window_size, len(losses) + 1), smooth_losses,
#                      label=f'Smoothed Loss (window={window_size})', color='orange', linewidth=2)
#
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.title(f'{args.model} Training Loss Curve', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=11)
#     plt.tight_layout()
#     loss_path = os.path.join(args.save_path, 'loss_curve.png')
#     plt.savefig(loss_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"✅ Loss 曲线已保存至: {loss_path}")
#
#     # ---------- 2. F1 Score 曲线 ----------
#     if has_f1:
#         plt.figure(figsize=(10, 6))
#         epochs = range(1, len(val_f1_scores) + 1)
#         plt.plot(epochs, val_f1_scores, marker='o', label='F1 Score', color='green', linewidth=2)
#
#         # 添加最佳点标记
#         best_f1_idx = np.argmax(val_f1_scores)
#         plt.plot(best_f1_idx + 1, val_f1_scores[best_f1_idx], 'ro', markersize=8,
#                  label=f'Best F1: {val_f1_scores[best_f1_idx]:.4f} (Epoch {best_f1_idx + 1})')
#
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('F1 Score', fontsize=12)
#         plt.title('F1 Score Curve', fontsize=14, fontweight='bold')
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=11)
#         plt.ylim(0, 1.05)  # F1分数范围0-1
#         plt.tight_layout()
#         f1_path = os.path.join(args.save_path, 'f1_curve.png')
#         plt.savefig(f1_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ F1 曲线已保存至: {f1_path}")
#
#     # ---------- 3. MSE 曲线 ----------
#     if has_mse:
#         plt.figure(figsize=(10, 6))
#         epochs = range(1, len(val_mses) + 1)
#         plt.plot(epochs, val_mses, marker='x', label='MSE', color='red', linewidth=2)
#
#         # 添加最佳点标记
#         best_mse_idx = np.argmin(val_mses)
#         plt.plot(best_mse_idx + 1, val_mses[best_mse_idx], 'ro', markersize=8,
#                  label=f'Best MSE: {val_mses[best_mse_idx]:.4f} (Epoch {best_mse_idx + 1})')
#
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('MSE', fontsize=12)
#         plt.title('Mean Squared Error Curve', fontsize=14, fontweight='bold')
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=11)
#         plt.tight_layout()
#         mse_path = os.path.join(args.save_path, 'mse_curve.png')
#         plt.savefig(mse_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ MSE 曲线已保存至: {mse_path}")
#
#     # ---------- 4. MAE 曲线 ----------
#     if has_mae:
#         plt.figure(figsize=(10, 6))
#         epochs = range(1, len(val_maes) + 1)
#         plt.plot(epochs, val_maes, marker='s', label='MAE', color='purple', linewidth=2)
#
#         # 添加最佳点标记
#         best_mae_idx = np.argmin(val_maes)
#         plt.plot(best_mae_idx + 1, val_maes[best_mae_idx], 'ro', markersize=8,
#                  label=f'Best MAE: {val_maes[best_mae_idx]:.4f} (Epoch {best_mae_idx + 1})')
#
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('MAE', fontsize=12)
#         plt.title('Mean Absolute Error Curve', fontsize=14, fontweight='bold')
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=11)
#         plt.tight_layout()
#         mae_path = os.path.join(args.save_path, 'mae_curve.png')
#         plt.savefig(mae_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ MAE 曲线已保存至: {mae_path}")
#
#     # ---------- 5. Accuracy 曲线 ----------
#     if has_acc:
#         plt.figure(figsize=(10, 6))
#         epochs = range(1, len(val_acc_list) + 1)
#         plt.plot(epochs, val_acc_list, marker='d', label='Validation Accuracy', color='orange', linewidth=2)
#
#         # 添加最佳点标记
#         best_acc_idx = np.argmax(val_acc_list)
#         plt.plot(best_acc_idx + 1, val_acc_list[best_acc_idx], 'ro', markersize=8,
#                  label=f'Best Acc: {val_acc_list[best_acc_idx]:.4f} (Epoch {best_acc_idx + 1})')
#
#         plt.xlabel('Epoch', fontsize=12)
#         plt.ylabel('Accuracy', fontsize=12)
#         plt.title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=11)
#         plt.ylim(0, 1.05)  # 准确率范围0-1
#         plt.tight_layout()
#         acc_path = os.path.join(args.save_path, 'accuracy_curve.png')
#         plt.savefig(acc_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ Accuracy 曲线已保存至: {acc_path}")
#
#     # ---------- 6. 混淆矩阵 ----------
#     if cm is not None:
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
#                     square=True, linewidths=0.5)
#         plt.xlabel("Predicted", fontsize=12)
#         plt.ylabel("True", fontsize=12)
#         plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         cm_path = os.path.join(args.save_path, 'confusion_matrix_final.png')
#         plt.savefig(cm_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ 混淆矩阵已保存至: {cm_path}")
#
#     # ---------- 7. 指标汇总分析 ----------
#     print("\n" + "=" * 50)
#     print("📊 指标统计与趋势分析")
#     print("=" * 50)
#
#     # Loss 分析
#     print(f"📉 Loss 分析:")
#     print(f"   - 初始 Loss: {losses[0]:.4f}")
#     print(f"   - 最终 Loss: {losses[-1]:.4f}")
#     print(f"   - 最佳 Loss: {np.min(losses):.4f} (Epoch {np.argmin(losses) + 1})")
#     print(f"   - Loss 降幅: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
#
#     # F1 分析
#     if has_f1:
#         print(f"\n🎯 F1 Score 分析:")
#         print(f"   - 初始 F1: {val_f1_scores[0]:.4f}")
#         print(f"   - 最终 F1: {val_f1_scores[-1]:.4f}")
#         print(f"   - 最佳 F1: {np.max(val_f1_scores):.4f} (Epoch {np.argmax(val_f1_scores) + 1})")
#
#     # MSE 分析
#     if has_mse:
#         print(f"\n📊 MSE 分析:")
#         print(f"   - 初始 MSE: {val_mses[0]:.4f}")
#         print(f"   - 最终 MSE: {val_mses[-1]:.4f}")
#         print(f"   - 最佳 MSE: {np.min(val_mses):.4f} (Epoch {np.argmin(val_mses) + 1})")
#
#     # MAE 分析
#     if has_mae:
#         print(f"\n📈 MAE 分析:")
#         print(f"   - 初始 MAE: {val_maes[0]:.4f}")
#         print(f"   - 最终 MAE: {val_maes[-1]:.4f}")
#         print(f"   - 最佳 MAE: {np.min(val_maes):.4f} (Epoch {np.argmin(val_maes) + 1})")
#
#     # Accuracy 分析 (新增)
#     if has_acc:
#         print(f"\n🎯 Accuracy 分析:")
#         print(f"   - 初始 Accuracy: {val_acc_list[0]:.4f}")
#         print(f"   - 最终 Accuracy: {val_acc_list[-1]:.4f}")
#         print(f"   - 最佳 Accuracy: {np.max(val_acc_list):.4f} (Epoch {np.argmax(val_acc_list) + 1})")
#         print(f"   - 准确率提升: {((val_acc_list[-1] - val_acc_list[0]) * 100):.2f}%")
#
#     # ---------- 8. 模型评估建议 ----------
#     print("\n" + "=" * 50)
#     print("💡 模型评估建议")
#     print("=" * 50)
#
#     # F1 Score 评估
#     if has_f1:
#         final_f1 = val_f1_scores[-1]
#         if final_f1 > 0.8:
#             print("✅ F1 Score 很高 (>0.8)，模型区分效果优秀。")
#         elif final_f1 > 0.6:
#             print("⚠️ F1 Score 中等 (0.6-0.8)，模型有一定效果但可继续优化。")
#         else:
#             print("❌ F1 Score 偏低 (<0.6)，建议检查模型结构、特征工程或数据质量。")
#
#     # Accuracy 评估 (新增)
#     if has_acc:
#         final_acc = val_acc_list[-1]
#         if final_acc > 0.9:
#             print("✅ 准确率很高 (>90%)，模型性能优秀。")
#         elif final_acc > 0.8:
#             print("✅ 准确率良好 (80%-90%)，模型表现不错。")
#         elif final_acc > 0.7:
#             print("⚠️ 准确率中等 (70%-80%)，还有优化空间。")
#         else:
#             print("❌ 准确率较低 (<70%)，建议检查模型配置和数据。")
#
#     # Loss 收敛性评估
#     loss_reduction = (losses[0] - losses[-1]) / losses[0]
#     if loss_reduction > 0.5:
#         print("✅ Loss 显著下降 (>50%)，训练收敛良好。")
#     elif loss_reduction > 0.1:
#         print("⚠️ Loss 有所下降 (10%-50%)，收敛较慢，可能需要调整学习率。")
#     else:
#         print("❌ Loss 几乎未下降 (<10%)，可能存在欠拟合或学习率问题。")
#
#     # 过拟合检测 (新增)
#     if has_acc and len(val_acc_list) > 10:
#         recent_acc_trend = np.mean(val_acc_list[-5:]) - np.mean(val_acc_list[-10:-5])
#         if recent_acc_trend < -0.02:
#             print("⚠️ 验证准确率有下降趋势，可能存在过拟合。")
#         elif recent_acc_trend > 0.01:
#             print("✅ 验证准确率持续提升，模型仍在学习。")
#         else:
#             print("📊 验证准确率趋于稳定。")
#
#     print("=" * 50)
#     print("📈 分析报告生成完成！")
#
#     return {
#         'best_loss': np.min(losses),
#         'best_loss_epoch': np.argmin(losses) + 1,
#         'final_loss': losses[-1],
#         'best_f1': np.max(val_f1_scores) if has_f1 else None,
#         'best_f1_epoch': np.argmax(val_f1_scores) + 1 if has_f1 else None,
#         'final_f1': val_f1_scores[-1] if has_f1 else None,
#         'best_acc': np.max(val_acc_list) if has_acc else None,
#         'best_acc_epoch': np.argmax(val_acc_list) + 1 if has_acc else None,
#         'final_acc': val_acc_list[-1] if has_acc else None,
#         'loss_reduction_rate': loss_reduction
#     }


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_model(trian_losses, val_losses, val_f1_scores, val_mses, val_maes, cm, val_acc_list, args):
    """
    综合分析模型训练过程：
    - Loss 曲线（训练与验证分别绘制）
    - F1、MSE、MAE、Accuracy 分别生成独立图像
    - 混淆矩阵图像
    - 指标统计与趋势分析
    """
    os.makedirs(args.save_path, exist_ok=True)

    print("📊 开始生成模型分析报告...")

    trian_losses = np.array(trian_losses)
    val_losses = np.array(val_losses) if val_losses is not None else None

    has_f1 = val_f1_scores is not None and len(val_f1_scores) > 0
    has_mse = val_mses is not None and len(val_mses) > 0
    has_mae = val_maes is not None and len(val_maes) > 0
    has_acc = val_acc_list is not None and len(val_acc_list) > 0

    # ---------- 1. Loss 曲线 ----------
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(trian_losses) + 1)
    plt.plot(epochs, trian_losses, label='Training Loss', color='blue', linewidth=2)

    if val_losses is not None and len(val_losses) > 0:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_losses) + 1, args.save_freq)
        plt.plot(val_epochs, val_losses, label='Validation Loss', color='red', linewidth=2, marker='o')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{args.model} Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = os.path.join(args.save_path, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss 曲线已保存至: {loss_path}")

    # ---------- 2. F1 Score ----------
    if has_f1:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_f1_scores) + 1, args.save_freq)
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_f1_scores, marker='o', label='F1 Score', color='green', linewidth=2)
        best_f1_idx = np.argmax(val_f1_scores)
        plt.plot(val_epochs[best_f1_idx], val_f1_scores[best_f1_idx], 'ro', markersize=8,
                 label=f'Best F1: {val_f1_scores[best_f1_idx]:.4f} (Epoch {val_epochs[best_f1_idx]})')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        f1_path = os.path.join(args.save_path, 'f1_curve.png')
        plt.savefig(f1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ F1 曲线已保存至: {f1_path}")

    # ---------- 3. MSE ----------
    if has_mse:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_mses) + 1, args.save_freq)
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_mses, marker='x', label='MSE', color='red', linewidth=2)
        best_idx = np.argmin(val_mses)
        plt.plot(val_epochs[best_idx], val_mses[best_idx], 'ro', markersize=8,
                 label=f'Best MSE: {val_mses[best_idx]:.4f} (Epoch {val_epochs[best_idx]})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        mse_path = os.path.join(args.save_path, 'mse_curve.png')
        plt.savefig(mse_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ MSE 曲线已保存至: {mse_path}")

    # ---------- 4. MAE ----------
    if has_mae:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_maes) + 1, args.save_freq)
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_maes, marker='s', label='MAE', color='purple', linewidth=2)
        best_idx = np.argmin(val_maes)
        plt.plot(val_epochs[best_idx], val_maes[best_idx], 'ro', markersize=8,
                 label=f'Best MAE: {val_maes[best_idx]:.4f} (Epoch {val_epochs[best_idx]})')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        mae_path = os.path.join(args.save_path, 'mae_curve.png')
        plt.savefig(mae_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ MAE 曲线已保存至: {mae_path}")

    # ---------- 5. Accuracy ----------
    if has_acc:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_acc_list) + 1, args.save_freq)
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_acc_list, marker='d', label='Validation Accuracy', color='orange', linewidth=2)
        best_idx = np.argmax(val_acc_list)
        plt.plot(val_epochs[best_idx], val_acc_list[best_idx], 'ro', markersize=8,
                 label=f'Best Acc: {val_acc_list[best_idx]:.4f} (Epoch {val_epochs[best_idx]})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        acc_path = os.path.join(args.save_path, 'accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Accuracy 曲线已保存至: {acc_path}")

    # ---------- 6. 混淆矩阵 ----------
    if cm is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    square=True, linewidths=0.5)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        cm_path = os.path.join(args.save_path, 'confusion_matrix_final.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 混淆矩阵已保存至: {cm_path}")

    # ---------- 7. 控制台总结 ----------
    print("\n" + "=" * 50)
    print("📊 指标统计与趋势分析")
    print("=" * 50)


    print(f"📉 Training Loss 分析:")
    print(f"   - 初始训练 Loss: {trian_losses[0]:.4f}")
    print(f"   - 最终训练 Loss: {trian_losses[-1]:.4f}")
    best_train_loss = np.min(trian_losses)
    best_train_epoch = np.argmin(trian_losses) + 1
    print(f"   - 最佳训练 Loss: {best_train_loss:.4f} (Epoch {best_train_epoch})")
    train_loss_reduction = (trian_losses[0] - trian_losses[-1]) / trian_losses[0]
    print(f"   - Loss 降幅: {train_loss_reduction * 100:.2f}%")

    if val_losses is not None and len(val_losses) > 0:
        val_epochs = np.arange(args.save_freq, args.save_freq * len(val_losses) + 1, args.save_freq)
        print(f"\n📉 Validation Loss 分析:")
        print(f"   - 初始验证 Loss: {val_losses[0]:.4f} (Epoch {val_epochs[0]})")
        print(f"   - 最终验证 Loss: {val_losses[-1]:.4f} (Epoch {val_epochs[-1]})")
        best_val_loss = np.min(val_losses)
        best_val_epoch = val_epochs[np.argmin(val_losses)]
        print(f"   - 最佳验证 Loss: {best_val_loss:.4f} (Epoch {best_val_epoch})")
        val_loss_reduction = (val_losses[0] - val_losses[-1]) / val_losses[0]
        print(f"   - Loss 降幅: {val_loss_reduction * 100:.2f}%")



    if has_f1:
        print(f"\n🎯 F1 Score 分析:")
        print(f"   - 初始 F1: {val_f1_scores[0]:.4f}")
        print(f"   - 最终 F1: {val_f1_scores[-1]:.4f}")
        print(f"   - 最佳 F1: {np.max(val_f1_scores):.4f} (Epoch {np.argmax(val_f1_scores) + 1})")

    # MSE 分析
    if has_mse:
        print(f"\n📊 MSE 分析:")
        print(f"   - 初始 MSE: {val_mses[0]:.4f}")
        print(f"   - 最终 MSE: {val_mses[-1]:.4f}")
        print(f"   - 最佳 MSE: {np.min(val_mses):.4f} (Epoch {np.argmin(val_mses) + 1})")

    # MAE 分析
    if has_mae:
        print(f"\n📈 MAE 分析:")
        print(f"   - 初始 MAE: {val_maes[0]:.4f}")
        print(f"   - 最终 MAE: {val_maes[-1]:.4f}")
        print(f"   - 最佳 MAE: {np.min(val_maes):.4f} (Epoch {np.argmin(val_maes) + 1})")

    # Accuracy 分析 (新增)
    if has_acc:
        print(f"\n🎯 Accuracy 分析:")
        print(f"   - 初始 Accuracy: {val_acc_list[0]:.4f}")
        print(f"   - 最终 Accuracy: {val_acc_list[-1]:.4f}")
        print(f"   - 最佳 Accuracy: {np.max(val_acc_list):.4f} (Epoch {np.argmax(val_acc_list) + 1})")
        print(f"   - 准确率提升: {((val_acc_list[-1] - val_acc_list[0]) * 100):.2f}%")

    # ---------- 8. 模型评估建议 ----------
    print("\n" + "=" * 50)
    print("💡 模型评估建议")
    print("=" * 50)

    # F1 Score 评估
    if has_f1:
        final_f1 = val_f1_scores[-1]
        if final_f1 > 0.8:
            print("✅ F1 Score 很高 (>0.8)，模型区分效果优秀。")
        elif final_f1 > 0.6:
            print("⚠️ F1 Score 中等 (0.6-0.8)，模型有一定效果但可继续优化。")
        else:
            print("❌ F1 Score 偏低 (<0.6)，建议检查模型结构、特征工程或数据质量。")

    # Accuracy 评估 (新增)
    if has_acc:
        final_acc = val_acc_list[-1]
        if final_acc > 0.9:
            print("✅ 准确率很高 (>90%)，模型性能优秀。")
        elif final_acc > 0.8:
            print("✅ 准确率良好 (80%-90%)，模型表现不错。")
        elif final_acc > 0.7:
            print("⚠️ 准确率中等 (70%-80%)，还有优化空间。")
        else:
            print("❌ 准确率较低 (<70%)，建议检查模型配置和数据。")

    # Loss 收敛性评估
    loss_reduction = (trian_losses[0] - trian_losses[-1]) / trian_losses[0]
    if loss_reduction > 0.5:
        print("✅ Loss 显著下降 (>50%)，训练收敛良好。")
    elif loss_reduction > 0.1:
        print("⚠️ Loss 有所下降 (10%-50%)，收敛较慢，可能需要调整学习率。")
    else:
        print("❌ Loss 几乎未下降 (<10%)，可能存在欠拟合或学习率问题。")

    # 过拟合检测 (新增)
    if has_acc and len(val_acc_list) > 10:
        recent_acc_trend = np.mean(val_acc_list[-5:]) - np.mean(val_acc_list[-10:-5])
        if recent_acc_trend < -0.02:
            print("⚠️ 验证准确率有下降趋势，可能存在过拟合。")
        elif recent_acc_trend > 0.01:
            print("✅ 验证准确率持续提升，模型仍在学习。")
        else:
            print("📊 验证准确率趋于稳定。")

    print("=" * 50)
    print("📈 分析报告生成完成！")








    return {
        'best_loss': np.min(trian_losses),
        'best_loss_epoch': np.argmin(trian_losses) + 1,
        'final_loss': trian_losses[-1],
        'best_f1': np.max(val_f1_scores) if has_f1 else None,
        'best_f1_epoch': val_epochs[np.argmax(val_f1_scores)] if has_f1 else None,
        'final_f1': val_f1_scores[-1] if has_f1 else None,
        'best_acc': np.max(val_acc_list) if has_acc else None,
        'best_acc_epoch': val_epochs[np.argmax(val_acc_list)] if has_acc else None,
        'final_acc': val_acc_list[-1] if has_acc else None,
    }
