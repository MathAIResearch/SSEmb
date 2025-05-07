import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
import numpy as np
from tqdm import tqdm
import argparse
from Tools.SubTree import SubTree

# Load dataset
print('Loading Dataset')
train_dataset = []
for i in tqdm(range(5)):
    train_data_i = torch.load('Dataset/Train_data_batch_'+str(i+1)+'.pt')
    train_dataset.extend(train_data_i)
train_infor = torch.load('Dataset/Train_data_information.pt')
query_dataset = torch.load('Dataset/Query_data.pt')
query_infor = torch.load('Dataset/Query_data_information.pt')
words = torch.load('Tools/word_11868.pt')
words.append('unif')
print(len(train_dataset),len(train_infor['id']),len(query_dataset),len(query_infor['id']))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, initial_node, pool_mode):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.initial_node = initial_node
        self.pool_mode = pool_mode
        if self.initial_node == 'random':
            self.x_embedding = torch.nn.Embedding(11868, input_dim)
            torch.nn.init.xavier_uniform_(self.x_embedding.weight)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))
    def forward(self, x, edge_index, batch):
        try:
            vec_random = torch.randn(1, 300).to(device)
            vec_all = torch.cat([self.x_embedding.weight, vec_random], dim=0)
            z = vec_all[x[:, 0]]
        except:
            vec_random = torch.randn(1, 300).to(device)
            vec_all = torch.cat([self.x_embedding.weight, vec_random], dim=0)
            z = vec_all(x[:, 0])
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        if self.pool_mode=='add':
            gs = [global_add_pool(z, batch) for z in zs]
        elif self.pool_mode=='mean':
            gs = [global_mean_pool(z, batch) for z in zs]
        else:
            gs = [global_max_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, words):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.aug1, self.aug2 = augmentor
    def forward(self,  x, edge_index, batch):
        x1, edge_index1, edge_weight1  = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2  = self.aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2

aug_p1 = 0.01
aug_p3_1 = 0.3
aug_p3_2 = 0.005
aug_p3_3 = 0.002
num_layers = 2
tau = 0.012
input_dim = 300
hidden_dim = 200
initial_node = 'random'
pool_mode = 'add'

aug = {1: A.FeatureMasking(pf=aug_p1), 
       2: A.EdgeRemoving(pe=aug_p1), 
       3: A.NodeDropping(pn=aug_p1),
       4: SubTree(words, pn1=aug_p3_1, pn2=aug_p3_2, pn3=aug_p3_3),
       5: A.RandomChoice([A.FeatureMasking(pf=aug_p1),
                          SubTree(words, pn1=aug_p3_1, pn2=aug_p3_2, pn3=aug_p3_3)],2),
       6: A.RandomChoice([A.FeatureMasking(pf=aug_p1),
                          A.EdgeRemoving(pe=aug_p1), 
                          A.NodeDropping(pn=aug_p1)],3),
        }
aug1 = A.Identity()
aug2 = aug[5]
gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, initial_node=initial_node, pool_mode=pool_mode).to(device)
encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), words=words).to(device)
parser = argparse.ArgumentParser(description="Run model with different checkpoints.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
args = parser.parse_args()
encoder_model.load_state_dict(torch.load(args.model_path, map_location=device))
encoder_model.eval()
print("Model loaded successfully!")

# Generate vector
print('Start generating vector')
test_loaders = [
    DataLoader(train_dataset, batch_size=1024, shuffle=False),
    DataLoader(query_dataset, batch_size=512, shuffle=False)
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i, test_loader in enumerate(test_loaders):
    if i == 0:
        num_samples = 16080179
    else:
        num_samples = 76
    save_path = f"graph_rep_{i}.dat"
    graph_rep_memmap = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, 400))
    offset = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        with torch.no_grad():
            _, graph_rep, _, _, _, _ = encoder_model(batch.x, batch.edge_index, batch.batch)
        batch_size = graph_rep.shape[0]
        end = min(offset + batch_size, num_samples)
        graph_rep_memmap[offset:end] = graph_rep.cpu().numpy()
        offset += batch_size
        del batch, graph_rep
        torch.cuda.empty_cache()