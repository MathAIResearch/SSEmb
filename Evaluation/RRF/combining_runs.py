from rrf import reciprocal_rank_fusion
import pandas as pd
from tqdm import tqdm

Our_Result = pd.read_csv('SSEmb.txt', delimiter='\t', names=['class','formula_id','post_id','rank','score','rum_number'], header=None, encoding='utf-8')
Other_Result  = pd.read_csv('approach0-task2-fusion_alpha05-manual-both-P.tsv', delimiter='\t', names=['class','formula_id','post_id','rank','score','rum_number'], header=None, encoding='utf-8')
all_class = sorted(set(Our_Result['class'].tolist()))

final_result = pd.DataFrame()
for idx in tqdm(all_class):
    filtered_Our_Result = Our_Result[Our_Result['class'] == idx]
    filtered_Other_Result = Other_Result[Other_Result['class'] == idx]
    model1_list = filtered_Our_Result['formula_id']
    model2_list = filtered_Other_Result['formula_id']

    rankings = {
        'model1': model1_list,
        'model2': model2_list,
    }
    doc,rrf_score = reciprocal_rank_fusion(rankings, k=60)

    RRF_Result = pd.DataFrame({'class': idx, 'formula_id': doc, 'post_id': None, 'rank':None, 'score': rrf_score, 'rum_number':'RRF'})
    RRF_Result['rank'] = range(1, len(RRF_Result) + 1)
    RRF_Result = RRF_Result.merge(filtered_Our_Result[['formula_id', 'post_id']], on='formula_id', how='left', suffixes=('', '_our'))
    RRF_Result = RRF_Result.merge(filtered_Other_Result[['formula_id', 'post_id']], on='formula_id', how='left', suffixes=('', '_other'))
    RRF_Result['post_id'] = RRF_Result['post_id_our'].fillna(RRF_Result['post_id_other']).astype(int)
    RRF_Result = RRF_Result.drop(columns=['post_id_our', 'post_id_other'])
    
    RRF_Result = RRF_Result[RRF_Result['rank'] <= 1000]
    final_result = pd.concat([final_result, RRF_Result], ignore_index=True)

final_result.to_csv('Approach0+SSEmb.txt', sep='\t', index=False, header=False, encoding='utf-8')