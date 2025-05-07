import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
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

# Generate query graph data
allowable_features = torch.load('Tools/word_11868.pt')
error=0
tree = ET.parse('Dataset/Topics_Task2_2022_V0.1.xml')
root = tree.getroot()
query_ids_1 = [elem.text for elem in root.findall(".//Formula_Id")]
query_ids = []
topic_numbers = [elem.attrib['number'] for elem in root.findall(".//Topic")]
lis = ['B.333','B.327','B.388','B.363','B.348','B.305','B.345','B.323','B.389','B.304','B.350','B.342','B.391','B.365','B.400','B.367','B.380','B.314','B.338','B.368','B.313','B.366','B.377','B.398','B.355','B.317','B.352','B.321','B.399','B.353','B.354','B.360','B.384','B.394','B.303','B.310','B.344','B.315','B.318','B.330','B.351','B.320','B.358','B.357','B.393','B.390','B.369','B.361','B.395','B.381','B.334','B.386','B.311','B.319','B.362','B.396','B.302','B.364','B.341','B.312','B.326','B.329','B.359','B.370','B.308','B.343','B.347','B.328','B.324','B.322','B.397','B.301','B.349','B.331','B.376','B.325']
for idx, topic_number in enumerate(topic_numbers):
    if topic_number in lis:
        query_ids.append(query_ids_1[idx])
# print(len(query_ids))
# print(query_ids)

df = pd.read_csv('Dataset/Topics_Formulas_OPT.V0.1.tsv', delimiter='\t', quotechar='"')
formula_id_all = df.iloc[:, 0].tolist()
topic_id_all = df.iloc[:, 1].tolist()
type_all = df.iloc[:, 3].tolist()
formula_all = df.iloc[:, 4].tolist()
data_list = []
formula_information = {'id':[], 'topic_id':[], 'type':[]}
for idx, content in tqdm(enumerate(formula_all), total=len(formula_all)):
    if formula_id_all[idx] in query_ids:
        try:
            formula = MathExtractor.parse_from_xml(content, 1, operator=True, missing_tags=None, problem_files=None)
            for key in formula:
                String = formula[key].tostring()
                g = GraphDict(String)
                graph_dict = g.create_graph_dict_from_string()
                node_dict = g.node_dict
                data = graph_dict_to_graph_data_obj(graph_dict, node_dict)

                data_list.append(data)
                formula_information['id'].append(formula_id_all[idx])
                formula_information['topic_id'].append(topic_id_all[idx])
                formula_information['type'].append(type_all[idx])
        except:
            error+=1
            print(formula_id_all[idx])
    pass
torch.save(data_list, 'Dataset/Query_data.pt')
torch.save(formula_information, 'Dataset/Query_data_information.pt')
