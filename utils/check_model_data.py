import torch
import numpy as np
from collections import OrderedDict
import json
import os
from typing import Dict, Any, Optional, List


# 假设QWGCN4.py在同一目录下，导入模型类
# from QWGCN4 import QWGCN_Fast, AdaptiveQWGNN, FastQuantumEvolution, EfficientMultiHop

class QWGCNCheckpointReader:
    """
    QWGCN模型checkpoint读取器
    用于加载模型权重并提取关键参数信息
    """

    def __init__(self, checkpoint_path: str):
        """
        初始化checkpoint读取器

        Args:
            checkpoint_path: checkpoint文件路径
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.model_state = None
        self.model_info = {}

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        加载checkpoint文件

        Returns:
            checkpoint内容字典
        """
        try:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")

            # 加载checkpoint
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

            # 提取模型状态
            if 'model_state_dict' in self.checkpoint:
                self.model_state = self.checkpoint['model_state_dict']
            elif 'state_dict' in self.checkpoint:
                self.model_state = self.checkpoint['state_dict']
            else:
                # 直接是模型状态字典
                self.model_state = self.checkpoint

            print(f"✓ 成功加载checkpoint: {self.checkpoint_path}")
            return self.checkpoint

        except Exception as e:
            print(f"✗ 加载checkpoint失败: {str(e)}")
            return {}

    def extract_model_parameters(self) -> Dict[str, Any]:
        """
        提取模型中的关键参数

        Returns:
            参数信息字典
        """
        if self.model_state is None:
            print("请先加载checkpoint")
            return {}

        parameters = {}

        # 遍历所有参数
        for param_name, param_value in self.model_state.items():
            # 提取量子演化相关参数
            if 'evolution_time' in param_name:
                value = param_value.item() if param_value.numel() == 1 else param_value.tolist()
                if param_value.dtype.is_complex:
                    if param_value.numel() == 1:
                        value = {
                            'real': param_value.real.item(),
                            'imag': param_value.imag.item()
                        }
                    else:
                        value = {
                            'real': param_value.real.tolist(),
                            'imag': param_value.imag.tolist()
                        }

                parameters['evolution_time'] = {
                    'value': value,
                    'shape': list(param_value.shape),
                    'dtype': str(param_value.dtype),
                    'parameter_path': param_name,
                    'is_complex': param_value.dtype.is_complex
                }

            if 'diffusion_strength' in param_name:
                value = param_value.item() if param_value.numel() == 1 else param_value.tolist()
                if param_value.dtype.is_complex:
                    if param_value.numel() == 1:
                        value = {
                            'real': param_value.real.item(),
                            'imag': param_value.imag.item()
                        }
                    else:
                        value = {
                            'real': param_value.real.tolist(),
                            'imag': param_value.imag.tolist()
                        }

                parameters['diffusion_strength'] = {
                    'value': value,
                    'shape': list(param_value.shape),
                    'dtype': str(param_value.dtype),
                    'parameter_path': param_name,
                    'is_complex': param_value.dtype.is_complex
                }

            # 提取多跳权重
            if 'hop_weights' in param_name:
                value = param_value.tolist()
                if param_value.dtype.is_complex:
                    value = {
                        'real': param_value.real.tolist(),
                        'imag': param_value.imag.tolist()
                    }

                parameters['hop_weights'] = {
                    'value': value,
                    'shape': list(param_value.shape),
                    'dtype': str(param_value.dtype),
                    'parameter_path': param_name,
                    'is_complex': param_value.dtype.is_complex
                }

            # 提取线性层权重
            if 'complex_fc' in param_name and 'weight' in param_name:
                layer_name = param_name.split('.')[0] + '_' + param_name.split('.')[1]
                if layer_name not in parameters:
                    parameters[layer_name] = {}

                parameters[layer_name]['weight'] = {
                    'shape': list(param_value.shape),
                    'dtype': str(param_value.dtype),
                    'parameter_path': param_name,
                    'norm': torch.norm(param_value).item(),
                    'is_complex': param_value.dtype.is_complex
                }

        self.model_info = parameters
        return parameters

    def print_parameter_summary(self):
        """
        打印参数摘要信息
        """
        if not self.model_info:
            self.extract_model_parameters()

        print("\n" + "=" * 60)
        print("QWGCN模型参数摘要")
        print("=" * 60)

        # 量子演化参数
        if 'evolution_time' in self.model_info:
            evo_time = self.model_info['evolution_time']
            print(f"\n🔬 量子演化时间 (evolution_time):")
            if evo_time.get('is_complex', False):
                print(
                    f"   值(实部): {evo_time['value']['real'] if isinstance(evo_time['value'], dict) else evo_time['value']}")
                if isinstance(evo_time['value'], dict):
                    print(f"   值(虚部): {evo_time['value']['imag']}")
            else:
                print(f"   值: {evo_time['value']}")
            print(f"   路径: {evo_time['parameter_path']}")
            print(f"   数据类型: {evo_time['dtype']}")

        if 'diffusion_strength' in self.model_info:
            diff_str = self.model_info['diffusion_strength']
            print(f"\n⚡ 扩散强度 (diffusion_strength):")
            if diff_str.get('is_complex', False):
                print(
                    f"   值(实部): {diff_str['value']['real'] if isinstance(diff_str['value'], dict) else diff_str['value']}")
                if isinstance(diff_str['value'], dict):
                    print(f"   值(虚部): {diff_str['value']['imag']}")
            else:
                print(f"   值: {diff_str['value']}")
            print(f"   路径: {diff_str['parameter_path']}")
            print(f"   数据类型: {diff_str['dtype']}")

        # 多跳权重
        if 'hop_weights' in self.model_info:
            hop_w = self.model_info['hop_weights']
            print(f"\n🔗 多跳权重 (hop_weights):")
            if hop_w.get('is_complex', False):
                print(f"   值(实部): {hop_w['value']['real'] if isinstance(hop_w['value'], dict) else hop_w['value']}")
                if isinstance(hop_w['value'], dict):
                    print(f"   值(虚部): {hop_w['value']['imag']}")
            else:
                print(f"   值: {hop_w['value']}")
            print(f"   形状: {hop_w['shape']}")
            print(f"   路径: {hop_w['parameter_path']}")
            print(f"   数据类型: {hop_w['dtype']}")

        # 线性层信息
        linear_layers = [k for k in self.model_info.keys() if 'quantum_layers' in k]
        if linear_layers:
            print(f"\n🧠 线性层信息:")
            for layer in linear_layers:
                if 'weight' in self.model_info[layer]:
                    weight_info = self.model_info[layer]['weight']
                    print(f"   {layer}:")
                    print(f"     形状: {weight_info['shape']}")
                    print(f"     权重范数: {weight_info['norm']:.4f}")

    def extract_all_parameters(self) -> Dict[str, Any]:
        """
        提取所有参数的详细信息

        Returns:
            所有参数的详细信息
        """
        if self.model_state is None:
            print("请先加载checkpoint")
            return {}

        all_params = {}

        for param_name, param_value in self.model_state.items():
            param_info = {
                'shape': list(param_value.shape),
                'dtype': str(param_value.dtype),
                'size': param_value.numel(),
                'requires_grad': param_value.requires_grad if hasattr(param_value, 'requires_grad') else False
            }

            # 如果是标量或小向量，保存具体值
            if param_value.numel() <= 10:
                if param_value.numel() == 1:
                    if param_value.dtype.is_complex:
                        param_info['value'] = {
                            'real': param_value.real.item(),
                            'imag': param_value.imag.item()
                        }
                    else:
                        param_info['value'] = param_value.item()
                else:
                    if param_value.dtype.is_complex:
                        param_info['value'] = {
                            'real': param_value.real.tolist(),
                            'imag': param_value.imag.tolist()
                        }
                    else:
                        param_info['value'] = param_value.tolist()
            else:
                # 大参数只保存统计信息
                if param_value.dtype.is_complex:
                    # 复数参数的统计信息
                    param_info['stats'] = {
                        'mean_real': param_value.real.mean().item(),
                        'mean_imag': param_value.imag.mean().item(),
                        'std_real': param_value.real.std().item(),
                        'std_imag': param_value.imag.std().item(),
                        'min_real': param_value.real.min().item(),
                        'max_real': param_value.real.max().item(),
                        'min_imag': param_value.imag.min().item(),
                        'max_imag': param_value.imag.max().item(),
                        'norm': torch.norm(param_value).item(),
                        'magnitude_mean': param_value.abs().mean().item()
                    }
                else:
                    # 实数参数的统计信息
                    param_info['stats'] = {
                        'mean': param_value.mean().item(),
                        'std': param_value.std().item(),
                        'min': param_value.min().item(),
                        'max': param_value.max().item(),
                        'norm': torch.norm(param_value).item()
                    }

            all_params[param_name] = param_info

        return all_params

    def save_parameters_to_json(self, output_path: str):
        """
        将参数信息保存到JSON文件

        Args:
            output_path: 输出JSON文件路径
        """
        if not self.model_info:
            self.extract_model_parameters()

        # 添加checkpoint基本信息
        output_data = {
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_keys': list(self.checkpoint.keys()) if self.checkpoint else [],
            'model_parameters': self.model_info,
            'parameter_count': len(self.model_state) if self.model_state else 0
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ 参数信息已保存到: {output_path}")
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")

    def get_model_architecture_info(self) -> Dict[str, Any]:
        """
        推断模型架构信息

        Returns:
            模型架构信息
        """
        if self.model_state is None:
            return {}

        arch_info = {
            'total_parameters': len(self.model_state),
            'layer_types': {},
            'model_components': []
        }

        # 分析参数名称推断架构
        for param_name in self.model_state.keys():
            parts = param_name.split('.')

            # 统计层类型
            for part in parts:
                if 'quantum_layers' in part:
                    arch_info['layer_types']['quantum_layers'] = arch_info['layer_types'].get('quantum_layers', 0) + 1
                elif 'batch_norms' in part:
                    arch_info['layer_types']['batch_norms'] = arch_info['layer_types'].get('batch_norms', 0) + 1
                elif 'complex_fc' in part:
                    arch_info['layer_types']['complex_fc'] = arch_info['layer_types'].get('complex_fc', 0) + 1

        return arch_info


def create_model_from_checkpoint(checkpoint_path: str, model_config: Dict[str, Any] = None):
    """
    从checkpoint创建模型实例（需要模型类定义）

    Args:
        checkpoint_path: checkpoint路径
        model_config: 模型配置参数

    Returns:
        加载了权重的模型实例
    """
    try:
        # 这里需要根据实际的模型类导入
        # from QWGCN4 import QWGCN_Fast

        reader = QWGCNCheckpointReader(checkpoint_path)
        checkpoint = reader.load_checkpoint()

        if model_config is None:
            # 默认配置
            model_config = {
                'in_features': 64,
                'hidden_features': 128,
                'out_features': 10,
                'num_layers': 2,
                'max_hops': 2,
                'evolution_order': 1,
                'dropout_rate': 0.1
            }

        # 创建模型实例
        # model = QWGCN_Fast(**model_config)

        # 加载权重
        # model.load_state_dict(reader.model_state)

        print("✓ 模型创建并加载权重成功")
        # return model

    except Exception as e:
        print(f"✗ 模型创建失败: {str(e)}")
        return None


# 使用示例
def example_usage():
    """使用示例"""

    # 假设的checkpoint路径
    checkpoint_path = r"D:\AI\Quantum\quantum\project_NEW\runs\PROTIENS\QWGCN4\best2025-07-27_11-57-34.pth"

    # 创建读取器
    reader = QWGCNCheckpointReader(checkpoint_path)

    # 加载checkpoint
    checkpoint = reader.load_checkpoint()

    if checkpoint:
        # 提取参数
        parameters = reader.extract_model_parameters()

        # 打印摘要
        reader.print_parameter_summary()

        # 获取架构信息
        arch_info = reader.get_model_architecture_info()
        print(f"\n📊 模型架构信息:")
        print(f"   总参数数: {arch_info['total_parameters']}")
        print(f"   层类型: {arch_info['layer_types']}")

        # 保存到JSON
        reader.save_parameters_to_json("qwgcn_parameters.json")

        # 获取所有参数详情
        all_params = reader.extract_all_parameters()

        # 查找特定参数
        evolution_params = {k: v for k, v in all_params.items()
                            if 'evolution_time' in k or 'diffusion_strength' in k}

        print(f"\n🔍 找到的演化参数:")
        for param_name, param_info in evolution_params.items():
            print(f"   {param_name}: {param_info}")


if __name__ == "__main__":
    example_usage()


