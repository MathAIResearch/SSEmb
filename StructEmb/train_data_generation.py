import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
from math_tan.math_extractor import MathExtractor
from Tools.util_for_data import GraphDict

def graph_dict_to_graph_data_obj(graph_dict, node_dict):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified node and edge features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    node_features_list = []
    edges_list = []
    node_list = list(graph_dict.keys())

    for k, v in graph_dict.items():
        if node_dict[k] in allowable_features:
            node_feature = [allowable_features.index(node_dict[k])]
        else:
            node_feature = [len(allowable_features)]
        node_features_list.append(node_feature)
        if len(v) > 0:
            i = node_list.index(k)
            for v_i in v:
                j = node_list.index(v_i)
                edges_list.append((i, j))

    x = torch.tensor(np.array(node_features_list), dtype=torch.long)
    if len(graph_dict) > 1:
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

# Generate train graph data
allowable_features = torch.load('Tools/word_11868.pt')
error=0
for i in range(101):
    df = pd.read_csv('Dataset/opt_representation_v3/'+str(i+1)+'.tsv', delimiter='\t', quotechar='"')
    formula_id_all = df.iloc[:, 0].tolist()
    post_id_all = df.iloc[:, 1].tolist()
    type_all = df.iloc[:, 3].tolist()
    visual_id_all = df.iloc[:, 6].tolist()
    formula_all = df.iloc[:, 8].tolist()
    data_list = []
    formula_information = {'id':[], 'post_id':[], 'type':[], 'visual_id':[]}
    for idx, content in tqdm(enumerate(formula_all), total=len(formula_all), desc='Processing'+str(i+1)):
        try:
            formula = MathExtractor.parse_from_xml(content, 1, operator=True, missing_tags=None, problem_files=None)
            for key in formula:
                String = formula[key].tostring()
                g = GraphDict(String)
                graph_dict = g.create_graph_dict_from_string()
                node_dict = g.node_dict
                if len(node_dict)>=3 and type_all[idx]!='comment':
                    data = graph_dict_to_graph_data_obj(graph_dict, node_dict)

                    data_list.append(data)
                    formula_information['id'].append(formula_id_all[idx])
                    formula_information['post_id'].append(post_id_all[idx])
                    formula_information['type'].append(type_all[idx])
                    formula_information['visual_id'].append(visual_id_all[idx])
        except:
            error+=1
        pass

    torch.save(data_list, 'Dataset/Train_data/'+str(i+1)+'.pt')
    torch.save(formula_information, 'Dataset/Train_data_information/'+'infor'+str(i+1)+'.pt')

# Merge train graph data
group_sizes = [20, 20, 20, 20, 21]
start = 0
for i, size in enumerate(group_sizes):
    batch = []
    for j in range(start, start + size):
        batch.extend(torch.load(f'Dataset/Train_data/{j+1}.pt'))
    torch.save(batch, f'Dataset/Train_data_batch_{i+1}.pt')
    start += size

info = [torch.load(f'Dataset/Train_data_information/infor{i+1}.pt') for i in range(101)]
torch.save(info, 'Dataset/Train_data_information.pt')