import torch
import pandas as pd
from tqdm import tqdm

from math_tan.math_extractor import MathExtractor
from Tools.util_for_data import GraphDict


node_statistics = dict()
error=0
for i in range(101):
    df = pd.read_csv('Dataset/opt_representation_v3/'+str(i+1)+'.tsv', delimiter='\t', quotechar='"')
    formulas = df.iloc[:, 8].tolist()

    for content in tqdm(formulas, total=len(formulas), desc="Processing formulas"+str(i+1)):
        try:
            formula = MathExtractor.parse_from_xml(content, 1, operator=True, missing_tags=None, problem_files=None)
            for key in formula:
                String = formula[key].tostring()
                g = GraphDict(String)
                graph_dict = g.create_graph_dict_from_string()
                node_dict = g.node_dict
                for node in node_dict.values():
                    if node in node_statistics.keys():
                        node_statistics[node]+=1
                    else:
                        node_statistics[node]=1
        except:
            error+=1
        pass
    print(error)


torch.save(node_statistics, "node_statistics.pt")