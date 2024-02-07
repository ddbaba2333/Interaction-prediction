import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv, Node2Vec
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
import data_handle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import warnings
# warnings.filterwarnings("ignore")

# 创建torch_geometric数据集
edge_index = data_handle.get_edge_index()
x = data_handle.get_x_feature()

data = Data(x=x, edge_index=edge_index)

print(data)
print(data.num_features)
data.num_features

# 分割训练边集、验证边集以及测试边集
transform = T.Compose([
    T.ToUndirected(merge=True),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
])
train_data, val_data, test_data = transform(data)

model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=20,
                 context_size=10, walks_per_node=10, num_negative_samples=1,
                 sparse=True).to(device)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)