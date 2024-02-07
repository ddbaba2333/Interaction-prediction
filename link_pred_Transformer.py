###################################################################################
# 代码改自：https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, TransformerConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
import data_handle
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# random.seed(42)
# dd = random.random()
# print('随机数：', dd)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import warnings
# warnings.filterwarnings("ignore")

# 创建torch_geometric数据集
edge_index = data_handle.get_edge_index()
x = data_handle.get_x_feature()

data = Data(x=x, edge_index=edge_index)

print(data)
print(data.num_features)

# 分割训练边集、验证边集以及测试边集
transform = T.Compose([
    T.ToUndirected(merge=True),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False),

])
train_data, val_data, test_data = transform(data)


# 构造一个简单的图卷积神经网络（两层），包含编码（节点嵌入）、解码（分数预测）等操作
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # GraphConv, GCNConv, GATConv, SAGEConv, TransformerConv   ,dropout=0.1
        # self.conv1 = TransformerConv(in_channels, hidden_channels, dropout=0.1, heads=8)
        # self.conv2 = TransformerConv(hidden_channels*8, out_channels*8, dropout=0.1, heads=8)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # 中间是将正样本与负样本拼接 shape:[2,272]

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = Net(data.num_features, 128, 64).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()


# 训练函数，每次训练重新采样负边，计算模型损失，反向传播误差，更新模型参数
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    # z所有节点的表示向量
    y_true = data.edge_label
    z = model.encode(data.x, data.edge_index)
    y_score = model.decode(z, data.edge_label_index).view(-1).sigmoid()



    y_pred = out_y_pred_to_01(y_score)
    # 各项指标 auc \ ap \ recall \ f1
    return roc_auc_score(y_true.cpu(), y_score.cpu()), \
           average_precision_score(y_true.cpu(), y_score.cpu()), \
           recall_score(y_true.cpu(), y_pred.cpu()), \
           f1_score(y_true.cpu(), y_pred.cpu())


def out_y_pred_to_01(y_score):
    # y_score是tensor格式的
    y_pred_list = []
    out_y_score_list = y_score.tolist()
    for i in out_y_score_list:
        if i > 0.5:
            i = 1
        else:
            i = 0
        y_pred_list.append(i)
    # 将 pred_list 转化成tensor格式
    return torch.tensor(y_pred_list, dtype=torch.float).view(-1)


best_test_auc = best_test_ap = best_test_recall = best_test_f1 = 0

for epoch in range(1, 201):
    loss = train()
    val_auc, val_ap, val_recall, val_f1 = test(val_data)
    test_auc, test_ap, test_recall, test_f1 = test(test_data)
    # 得到最好的test_auc值
    if test_auc > best_test_auc:
        best_test_auc = test_auc

    # 得到最好的test_ap值
    if test_ap > best_test_ap:
        best_test_ap = test_ap

    # 得到最好的test_recall值
    if test_recall > best_test_recall:
        best_test_recall = test_recall

    # 得到最好的test_f1值
    if test_f1 > best_test_f1:
        best_test_f1 = test_f1

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, || '
          f'Val_auc: {val_auc:.4f}, Test_auc: {test_auc:.4f}, || '
          f'Val_ap: {val_ap:.4f}, Test_ap: {test_ap:.4f}, || '
          f'Val_recall: {val_recall:.4f}, Test_recall: {test_recall:.4f}, || '
          f'Val_f1: {val_f1:.4f}, Test_f1: {test_f1:.4f} ')

print(f'best_test_auc: {best_test_auc:.4f}, best_test_ap: {best_test_ap:.4f}')
print(
    f'best_test_recall: {best_test_recall:.4f}, best_test_f1: {best_test_f1:.4f}')

z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)
