import numpy as np
from tqdm import tqdm
import faiss
import torch
import argparse

# Retrieval result
train_infor = torch.load('Dataset/Train_data_information.pt')
query_infor = torch.load('Dataset/Query_data_information.pt')

train_np = np.memmap('graph_rep_0.dat', dtype='float32', mode='r', shape=(16080179, 400))
query_np = np.memmap('graph_rep_1.dat', dtype='float32', mode='r', shape=(76, 400))

top_k = 1000
query_np = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
train_np = train_np / np.linalg.norm(train_np, axis=1, keepdims=True)

index = faiss.IndexFlatIP(400) 
index.add(train_np.astype(np.float32))
scores, topk_indices = index.search(query_np.astype(np.float32), top_k)

parser = argparse.ArgumentParser(description="Run model with different checkpoints.")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output result file.")
args = parser.parse_args()
output_path = args.output_file

with open(output_path, 'w', encoding='utf-8') as t:
    for i, (topk, score_list) in tqdm(enumerate(zip(topk_indices, scores)), total=len(topk_indices)):
        query_id = query_infor['topic_id'][i].replace("A", "B")
        for j, (idx, score) in enumerate(zip(topk, score_list)):
            formula_id = train_infor['id'][idx] 
            post_id = train_infor['post_id'][idx]
            t.write(f"{query_id}\t{formula_id}\t{post_id}\t{j+1}\t{score:.6f}\tRun_0\n")
print("Results saved successfully!")