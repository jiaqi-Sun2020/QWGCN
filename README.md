# QGCN
This is a quantum walk Graph Neural Network project.

---

## 📁 Directory Structure
```text
QGCN/
├── runs/                 # Model checkpoints and logs
│   └── model_name
├── dataset/              # Data directory
│   ├── dataname/         # Raw data files
│   └── dataset_build.py  # Script to build/process datasets
├── environment.yml       # Conda environment configuration
├── main.py               # Main entry point for training
├── model/                # Model architectures
│   ├── GCN.py
│   └── QGCN.py
├── config.json           # Global configuration parameters
├── test.py               # Testing/Evaluation script
├── train.py              # Training logic implementation
├── utils/                # Utility functions
│   ├── config.py         # Hyperparameter configuration & loading
│   ├── proprocess.py     # Data preprocessing tools
│   └── save.py           # Checkpoint saving utilities
└── jupyter notebook      # Experimental notebooks
```

## 🚀 Quick Start

Follow these steps to configure and run the QGCN model.

### 1. Installation
Ensure you have the required dependencies installed. You can set up the environment using `environment.yml` or `requirements.txt`.

```bash
# Option A: Using Conda
conda env create -f environment.yml
conda activate qgcn

# Option B: Using Pip
pip install -r requirements.txt
```

### 2. Configuration
You can adjust hyperparameters and model settings before training.

* **Primary Configuration:** Edit `config.json` for standard settings (e.g., dataset path, learning rate, epochs).
* **Advanced Configuration:** Modify `utils/config.py` to change how parameters are parsed or to adjust default system settings.

**Example `config.json`:**
```json
{
  "dataset": "Cora",
  "lr": 0.01,
  "epochs": 200,
  "hidden_dim": 64,
  "dropout": 0.5
}
```

### 3. Training
To start the training process, run the `main.py` script. This script loads the configuration and initiates the training loop.

```bash
python main.py
```

*Note: Checkpoints will be saved in the `runs/` directory.*
---

## 🛠 Prerequisites

### Python Environment
Please refer to `requirements.txt` or `environment.yml`.

### Hardware Specifications
This project was developed and tested on the following high-performance workstation:

**Operating System**
- **OS:** Windows 10 Pro for Workstations 64-bit (Version: 19045.5247, 22H2)

**Processor**
- **Model:** Intel Core i9-14900K @ 3.2GHz
- **Cores:** 24 (8P+16E)
- **Threads:** 32 (16P+16E)
- **Process Technology:** Intel 7

**Graphics Card**
- **Model:** NVIDIA GeForce RTX 4090
- **Brand:** ASUS
- **CUDA Cores:** 16384
- **VRAM:** 24GB (GDDR6X Micron)

**Motherboard**
- **Model:** TUF GAMING Z790-PRO WIFI
- **Manufacturer:** ASUSTeK
- **Chipset:** Intel Z790

**Memory**
- **Total Capacity:** 32GB
- **Frequency:** 4800MHz
- **Timings:** 40-39-39-76 2T
- **Brand & Model:** Acer DDR5-6600 (3300MHz) 16GB × 2

---

## 🎯 JNU Quantum Physics Lab
**Welcome to JNU Quantum Physics Lab!** We are looking for passionate individuals with Quantum and AI and cross-disciplinary CS research.

📩 **Interested? Send us your CV!**
- 📜 **Project Author:** [Jiaqi Sun](https://github.com/jiaqi-Sun2020)
  - ✉️ **Email:** 1623958071@qq.com
- 📜 **Boss:** [Zhihao Bian](https://science.jiangnan.edu.cn/info/1018/8646.htm)
  - ✉️ **Email:** zhihaobian@jiangnan.edu.cn