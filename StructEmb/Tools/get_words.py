import torch
from tqdm import tqdm

words_dict = torch.load('node_statistics.pt')
words = list(words_dict.keys())
number = list(words_dict.values())
lis =[]
for i in range(len(words)):
    if number[i]>20:
        lis.append(words[i])
torch.save(lis,'words_'+str(len(lis))+'.pt')