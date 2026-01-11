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

import numpy as np
import torch
import matplotlib
from utils.config import *
from train import train
from pathlib import Path
import sys

# Get the directory where the current file is located and add it to sys.path
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
    # Set a random seed to ensure reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    main(config_path = r"./config.json",)




