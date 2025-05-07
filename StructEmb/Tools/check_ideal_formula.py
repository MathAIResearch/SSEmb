import torch
import pandas as pd
from tqdm import tqdm
from math_tan.math_extractor import MathExtractor

df1 = pd.read_csv('Dataset/qrel_task2_2022_official.tsv', delimiter='\t', quotechar='"', header = None)
query_topic = df1.iloc[:, 0].tolist()
query_visual_id = df1.iloc[:, 2].tolist()
query_score = df1.iloc[:, 3].tolist()

error=0
formula_information = {'id':[],  'visual_id':[]}
formula_error_information = {'id':[],  'visual_id':[], 'post_id':[], 'score':[]}

for i in range(101):
    df = pd.read_csv('Dataset/opt_representation_v3/'+str(i+1)+'.tsv', delimiter='\t', quotechar='"')
    formula_id_all = df.iloc[:, 0].tolist()
    post_id_all = df.iloc[:, 1].tolist()
    visual_id_all = df.iloc[:, 6].tolist()
    formula_all = df.iloc[:, 8].tolist()
    for idx, content in tqdm(enumerate(formula_all), total=len(formula_all), desc="Processing formulas"+str(i+1)):
        try:
            formula = MathExtractor.parse_from_xml(content, 1, operator=True, missing_tags=None, problem_files=None)
            formula_information['id'].append(formula_id_all[idx])
            formula_information['visual_id'].append(visual_id_all[idx])
        except:
            error+=1
            formula_error_information['id'].append(formula_id_all[idx])
            formula_error_information['visual_id'].append(visual_id_all[idx])
            formula_error_information['post_id'].append(post_id_all[idx])
        pass

print(error)

ideal_formula_error = {'formula_id_error':[], 'visual_id_error':[], 'post_id_error':[], 'score_error':[]}
for i in range(11538):
    if query_score[i]!='0' and query_visual_id[i] not in formula_information['visual_id']:
        ideal_formula_error['visual_id_error'].append(query_visual_id[i])
        ideal_formula_error['score_error'].append(query_score[i])

        index = formula_error_information['visual_id'].index(query_visual_id[i])
        ideal_formula_error['formula_id_error'].append(formula_error_information['id'][index])
        ideal_formula_error['post_id_error'].append(formula_error_information['post_id'][index])

print(len(ideal_formula_error['formula_id_error']))
torch.save(ideal_formula_error,'ideal_formula_error.pt')
