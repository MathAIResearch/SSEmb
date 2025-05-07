import pandas as pd

# Load two_score data
result_df = pd.read_csv('two_score.txt', delimiter='\t', 
                        names=['class','formula_id','post_id','rank','formula_score','rum_number', 'embedding_score'], 
                        header=None, encoding='utf-8')
result_df['formula_score'] = pd.to_numeric(result_df['formula_score'], errors='coerce')
result_df['embedding_score'] = pd.to_numeric(result_df['embedding_score'], errors='coerce')

# Fusion
result_df['weight_score'] = result_df['formula_score']*0.5 + result_df['embedding_score']*0.5

# Rerank
result_df_sorted = result_df.sort_values(by=['class', 'weight_score'], ascending=[True, False])
result_df_sorted['rank'] = result_df_sorted.groupby('class').cumcount() + 1 

# Generate final result
final_df = result_df_sorted[['class', 'formula_id','post_id', 'rank', 'weight_score', 'rum_number']]
final_df = final_df[final_df['rank'] <= 1000] 
final_df.to_csv('SSEmb.txt', sep='\t', index=False, header=False, encoding='utf-8')