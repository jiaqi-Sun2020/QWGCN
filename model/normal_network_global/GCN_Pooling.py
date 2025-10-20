import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN_Protein(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, out_features)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_Protein(dataset.num_node_features, dataset.num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params}")
    print(model)

    data = next(iter(loader)).to(device)
    out = model(data)

    print(f"输出形状: {out.shape}")

