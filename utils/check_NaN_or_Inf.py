import numpy as np
import os
from torch_geometric.datasets import TUDataset
import warnings
from tqdm import tqdm


def check_nan_inf_in_tudataset(dataset_path, dataset_name='TWITTER-Real-Graph-Partial'):
    """
    检测TUDataset中的NaN和Inf值

    参数:
        dataset_path (str): TUDataset根目录路径
        dataset_name (str): 要检测的数据集名称

    返回:
        dict: 包含检测结果的字典
    """
    warnings.filterwarnings('ignore', category=UserWarning)

    results = {
        'dataset': dataset_name,
        'num_graphs': 0,
        'graphs_with_nan': 0,
        'graphs_with_inf': 0,
        'nan_details': {
            'x': 0,
            'edge_attr': 0,
            'y': 0,
            'other': 0
        },
        'inf_details': {
            'x': 0,
            'edge_attr': 0,
            'y': 0,
            'other': 0
        },
        'graph_details': {}
    }

    try:
        print(f"正在加载数据集 {dataset_name}...")
        # 使用正确的数据集加载方式
        dataset = TUDataset(root=dataset_path, name=dataset_name, use_node_attr=True, use_edge_attr=True)
        results['num_graphs'] = len(dataset)

        print("开始检测图中的异常值...")
        # 限制检测数量以避免内存问题（可以调整）
        max_graphs_to_check = min(1000, len(dataset))  # 先检查1000个图作为示例
        dataset = dataset[:max_graphs_to_check]

        for i, data in enumerate(tqdm(dataset, total=max_graphs_to_check)):
            graph_result = {
                'has_nan': False,
                'has_inf': False,
                'nan_locations': [],
                'inf_locations': []
            }

            # 检查节点特征x
            if hasattr(data, 'x') and data.x is not None:
                try:
                    x = data.x.numpy() if hasattr(data.x, 'numpy') else np.array(data.x)
                    if np.isnan(x).any():
                        results['nan_details']['x'] += 1
                        graph_result['has_nan'] = True
                        graph_result['nan_locations'].append('x')
                    if np.issubdtype(x.dtype, np.number) and np.isinf(x).any():
                        results['inf_details']['x'] += 1
                        graph_result['has_inf'] = True
                        graph_result['inf_locations'].append('x')
                except Exception as e:
                    print(f"\n处理图{i}的x属性时出错: {str(e)}")

            # 检查边属性edge_attr
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                try:
                    edge_attr = data.edge_attr.numpy() if hasattr(data.edge_attr, 'numpy') else np.array(data.edge_attr)
                    if np.isnan(edge_attr).any():
                        results['nan_details']['edge_attr'] += 1
                        graph_result['has_nan'] = True
                        graph_result['nan_locations'].append('edge_attr')
                    if np.issubdtype(edge_attr.dtype, np.number) and np.isinf(edge_attr).any():
                        results['inf_details']['edge_attr'] += 1
                        graph_result['has_inf'] = True
                        graph_result['inf_locations'].append('edge_attr')
                except Exception as e:
                    print(f"\n处理图{i}的edge_attr属性时出错: {str(e)}")

            # 检查图标签y
            if hasattr(data, 'y') and data.y is not None:
                try:
                    y = data.y.numpy() if hasattr(data.y, 'numpy') else np.array(data.y)
                    if np.isnan(y).any():
                        results['nan_details']['y'] += 1
                        graph_result['has_nan'] = True
                        graph_result['nan_locations'].append('y')
                    if np.issubdtype(y.dtype, np.number) and np.isinf(y).any():
                        results['inf_details']['y'] += 1
                        graph_result['has_inf'] = True
                        graph_result['inf_locations'].append('y')
                except Exception as e:
                    print(f"\n处理图{i}的y属性时出错: {str(e)}")

            # 更新统计结果
            if graph_result['has_nan']:
                results['graphs_with_nan'] += 1
            if graph_result['has_inf']:
                results['graphs_with_inf'] += 1

            results['graph_details'][f'graph_{i}'] = graph_result

    except Exception as e:
        print(f"\n加载或处理数据集时出错: {str(e)}")
        return None

    return results


def print_tudataset_results(results):
    """打印TUDataset检测结果"""
    if not results:
        print("未能获取检测结果")
        return

    print("\n=== TUDataset检测结果 ===")
    print(f"数据集名称: {results['dataset']}")
    print(f"检测的图数量: {results['num_graphs']} (实际检测了 {len(results['graph_details'])} 个)")
    print(
        f"包含NaN值的图数量: {results['graphs_with_nan']} ({results['graphs_with_nan'] / len(results['graph_details']):.2%})")
    print(
        f"包含Inf值的图数量: {results['graphs_with_inf']} ({results['graphs_with_inf'] / len(results['graph_details']):.2%})")

    print("\nNaN值分布:")
    for key, count in results['nan_details'].items():
        print(f"  - {key}: {count}")

    print("\nInf值分布:")
    for key, count in results['inf_details'].items():
        print(f"  - {key}: {count}")

    # 打印有问题的图示例
    problematic_graphs = {k: v for k, v in results['graph_details'].items()
                          if v['has_nan'] or v['has_inf']}
    if problematic_graphs:
        print(f"\n发现问题的图数量: {len(problematic_graphs)}")
        print("前5个有问题的图示例:")
        for i, (graph_id, details) in enumerate(problematic_graphs.items()):
            if i >= 5:
                break
            print(f"\n{graph_id}:")
            if details['has_nan']:
                print(f"  NaN位置: {', '.join(details['nan_locations'])}")
            if details['has_inf']:
                print(f"  Inf位置: {', '.join(details['inf_locations'])}")
    else:
        print("\n没有发现包含NaN或Inf值的图")


if __name__ == "__main__":
    # 替换为你的TUDataset根目录路径
    tudataset_root = "./data"  # 建议使用相对路径
    dataset_name = "TWITTER-Real-Graph-Partial"

    print(f"开始检测TUDataset: {dataset_name}")
    results = check_nan_inf_in_tudataset(tudataset_root, dataset_name)
    print_tudataset_results(results)