import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
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

# mask设置成none的原因：由于下游任务是链接预测任务，因此需要重新划分测试集和训练集；
# data.train_mask = data.val_mask = data.test_mask = None
# data = train_test_split_edges(data)

# 分割训练边集、验证边集以及测试边集
transform = T.Compose([
    T.ToUndirected(merge=True),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
])
train_data, val_data, test_data = transform(data)


# 构造一个简单的图卷积神经网络（两层），包含编码（节点嵌入）、解码（分数预测）等操作
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def encode(self):
        # import pdb
        # pdb.set_trace()
        # print(data.x)

        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        return self.conv2(x, data.edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # 将正样本与负样本拼接 shape:[2,272]
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# 将模型和数据送入设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)

# 指定优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


# 将训练集中的正边标签设置为1，负边标签设置为0
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


# 训练函数，每次训练重新采样负边，计算模型损失，反向传播误差，更新模型参数
def train():
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.edge_index.size(1),  # 负采样数量根据正样本
        force_undirected=True,
    )  # 得到负采样shape: [2,228]
    neg_edge_index = neg_edge_index.to(device)
    optimizer.zero_grad()
    z = model.encode()  # 利用正样本训练学习得到每个节点的特征 shape:[228 , 64]
    link_logits = model.decode(z, data.edge_index,
                               neg_edge_index)  # [272] 利用正样本和负样本 按位相乘 求和  (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    link_labels = get_link_labels(data.edge_index, neg_edge_index)  # [556] 前228个是1，后228个是0
    loss = F.binary_cross_entropy_with_logits(link_logits,
                                              link_labels)  # binary_cross_entropy_with_logits会自动计算link_logits的sigmoid
    loss.backward()
    optimizer.step()
    return loss


# 测试函数，评估模型在验证集和测试集上的预测准确率
@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs


# 训练模型，每次训练完，输出模型在验证集和测试集上的预测准确率
best_val_perf = test_perf = 0
for epoch in range(1, 11):
    train_loss = train() #  有问题
    tmp_test_perf = test()
    val_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))

# 利用训练好的模型计算网络中剩余所有边的分数
z = model.encode()
final_edge_index = model.decode_all(z)
