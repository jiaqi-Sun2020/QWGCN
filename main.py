# Copyright (C) [2025] [jiaqi Sun]
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import torch
import matplotlib
from utils.config import *
from train import train
from pathlib import Path
import sys

# 获取当前文件所在目录，并加入 sys.path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main(config_path = "./params.json"):
    args = json2args(config_path)
    if args.mode =="run":
        pass
    else:
        print("trian")
        train(args)

if __name__ == '__main__':
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    main(config_path = r"./config.json",)




