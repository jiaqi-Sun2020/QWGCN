import torch
from datetime import datetime
import os


def save_checkpoint(model, optimizer, epoch, file_path,name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"已创建目录: {file_path}")
    else:
        print(f"目录已存在: {file_path}")
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间（例如：'2025-03-24_15-30-00'）
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_filename = os.path.join(file_path,name+time_str+".pth")
    # print(checkpoint_filename)
    # 创建checkpoint字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # 保存checkpoint
    torch.save(checkpoint, checkpoint_filename)
    print(f"Checkpoint saved at {checkpoint_filename}")


# if __name__ == "__main__":
#
#     # save_checkpoint("2", "2", 3, "./runs")


