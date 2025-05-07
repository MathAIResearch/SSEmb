from lxml import etree
import torch
import re
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
from collections import Counter


# Load the first-stage result
result_df = pd.read_csv('result.txt', delimiter='\t', names=['class','formula_id','post_id','rank','score','rum_number'], header=None, encoding='utf-8')
result_df['embedding_score'] = None  

# Load the context of query and candidates
query = torch.load('text/query_content.pt')
query = {key: query[key] for key in sorted(query.keys())} #sorting
query_content = list(query.values())

collect = torch.load('text/all_post.pt') #need to execute 'python get_all_post.py' to get
lis1 = result_df['post_id'].tolist()

result_content = []
missing = 0
for post_id in tqdm(lis1, desc="Processing result_content", unit=" items"):
    key = str(int(post_id))
    if key in collect.keys():
        result_content.append(collect[key])
    else:
        result_content.append('no context')
        missing+=1
print('missing:',missing)

# Get semantic embeddings
list_all = query_content+result_content
print('Start embedding')
model_path_2 = "Sentence-transformers/"
model_2 = SentenceTransformer('all-MiniLM-L6-v2', model_path_2)
list_all = list(str(x) for x in tqdm(list_all, desc="Converting1"))
def truncate_text(text, max_length=1024):
    return text[:max_length] if len(text) > max_length else text
list_all = [truncate_text(x) for x in tqdm(list_all, desc="Converting2")]
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = model_2.encode(list_all, batch_size=2000, show_progress_bar=True, device=device, num_workers=4)
normalized_embeddings = normalize(embeddings)

# Calculate semantic similarity scores
idx_start = 76
for i in tqdm(range(10), desc="Processing cosine_similarity"):
    query_embedding = normalized_embeddings[i]
    idx_end = idx_start+500000
    cos_similarities = cosine_similarity([query_embedding], normalized_embeddings[idx_start:idx_end])[0] 
    result_df.iloc[idx_start-76:idx_end-76, result_df.columns.get_loc('embedding_score')] = cos_similarities.tolist()
    idx_start = idx_end
result_df.to_csv('two_score.txt', sep='\t', index=False, header=False, encoding='utf-8')

