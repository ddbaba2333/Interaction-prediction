#############################
#
# 创建torch_geometric中的图
#
#############################

import os
import numpy as np
import torch


# 获取节点 source 和 target
def get_edge_index():
    source_nodes_list = []
    target_nodes_list = []
    with open("D:\Postgradu\\12.Dr\9.17论文计划\Experiment_data\data_set_copy\\maven\\node_edge\\25_P2P.net", 'r',
              encoding='utf-8') as file_obj:
        for line in file_obj.readlines():
            line = np.array(line.strip().split(' ')).astype('int')
            source_node = line[0].astype('int')-1
            # 起始节点列
            source_nodes_list.append(source_node)
            target_node = line[1].astype('int')-1
            # 终止节点列
            target_nodes_list.append(target_node)

        # print('第一列source节点：', len(source_nodes_list), source_nodes_list)
        # print('第二列target节点：', len(target_nodes_list), target_nodes_list)

        edge_index = torch.tensor([source_nodes_list,  # 起始点
                                   target_nodes_list], dtype=torch.long)  # 终止点

        return edge_index


# 获取节点特征 [ 64维的向量 ]
def get_x_feature():
    all_nodes_feature = []
    with open("D:\Postgradu\\12.Dr\9.17论文计划\Experiment_data\data_set_copy\\maven\emb\P2P\embeddings_25.emb", 'r',
              encoding='utf-8') as file_obj:
        for line in file_obj.readlines()[1:]:
            list = np.array(line.strip().split(' ')[1:]).astype('float').tolist()
            all_nodes_feature.append(list)

        # print(all_nodes_feature)

        x = torch.tensor(all_nodes_feature, dtype=torch.float)
        return x


# get_edge_index()
# get_x_feature()
