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

#===============================================================Global comparison model
from model.normal_network_global.GCN import GCN_layer
from .QWGNN_all_graph import QuantumGNN_all_graph
from model.normal_network_global.GAT import GAT_layer
from model.normal_network_global.GINEConv import GINE_layer
from model.normal_network_global.GraphSAGE import SAGE_layer
from model.normal_network_global.mutil_GAT import mGAT_layer
from model.normal_network_global.TransformerConv import Transformer_layer
from model.normal_network_global.Res_GraphSAGE import Res_SAGE
#===============================================================局域对比模型

from model.normal_network_local.GAT import GAT_local
from model.normal_network_local.GCN import GCN_local
from model.normal_network_local.GraphSAGE import SAGE_local
from model.normal_network_local.multi_GAT import MultiGAT_local
from model.normal_network_local.Res_SAGE import ResidualSAGE_local
from model.normal_network_local.TransformerConv import Transformer_local


#===============================================================

from model.QWGCN4 import QWGCN_Fast  #快速演化

#===============================================================

from model.normal_network_global.GAT_And_GCN import GAT_And_GCN_layer
from model.normal_network_global.HGP_SL import HGP_SL_layer
import torch

# Model Factory Dictionary
model_dict = {


    "QWGNN":QuantumGNN_all_graph,
    "QWGCN":QWGCN_Fast,
    #============================================================Local unitary dilatation contrast model embedding
    "GAT_local":GAT_local,
    "GCN_local":GCN_local,
    "GraphSAGE_local":SAGE_local,
    "MultiGAT_local":MultiGAT_local,
    "Res_SAGE_local":ResidualSAGE_local,
    "TransformerConv_local":Transformer_local,
    "GAT_And_GCN_local":GAT_And_GCN_layer,
    #============================================================Global unitary dilatation contrast model 2_layers
    "GCN": GCN_layer,   #
    "GAT": GAT_layer,         #
    "GINEConv": GINE_layer,   #
    "GraphSAGE": SAGE_layer,
    "multi_GAT": mGAT_layer,
    "TransformerConv": Transformer_layer,
    "Res_SAGE": Res_SAGE,
    "GAT_And_GCN" : GAT_And_GCN_layer,
    "HGP_SL": HGP_SL_layer,
}


def get_model(args,num_node_features,num_classes,device):
    """
Returns the corresponding model instance based on the model name.

Parameters:

name (str): Model name, should be "GCN" or "QGCN"

*args, **kwargs: Passed to the model's constructor

Returns:

Model instance
    """

    name =args.model
    if name not in model_dict:
        raise ValueError(f"Model '{name}' not found. Available: {model_dict}")

    elif name =="QWGCN":

        #   QWGCN4
        model = QWGCN_Fast(in_features=num_node_features,
                           hidden_features=64,
                           out_features=num_classes,
                           num_layers=2,
                           max_hops=2,
                           evolution_order=getattr(args, "evolution_order", 2),
                           evolution_time=getattr(args, "evolution_time", 1.0),
                           diffusion_strength=getattr(args, "diffusion_strength", 1.0),
                           dropout_rate=args.dropout)

    # elif name == "QWGCN":

    elif name in ["GAT_local","GCN_local","GraphSAGE_local","MultiGAT_local","Res_SAGE_local","TransformerConv_local"]:
        model = model_dict[name](in_features=num_node_features,hidden_features = 64,out_features=num_classes)
    else:
        model = model_dict[name](in_features=num_node_features, out_features=num_classes)

    if args.checkpoint_path is not None:
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])  # ✅ 正确加载
            # state_dict = torch.load(args.checkpoint_path, map_location=args.device,weights_only=True)
            # model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from: {args.checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load weights from {args.checkpoint_path}: {e}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model{name}Total number of parameters: {total_params}")
    # print(model)
    return model
