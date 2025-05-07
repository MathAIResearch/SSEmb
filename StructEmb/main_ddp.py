import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
import random
import numpy as np
from tqdm import tqdm
import os
import torch.multiprocessing as mp
import argparse
from Tools.SubTree import SubTree


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
        vec_random = torch.randn(1, 300).to(x.device)
        vec_all = torch.cat([self.x_embedding.weight, vec_random], dim=0)
        z = vec_all[x[:, 0]]
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        if self.pool_mode == 'add':
            gs = [global_add_pool(z, batch) for z in zs]
        elif self.pool_mode == 'mean':
            gs = [global_mean_pool(z, batch) for z in zs]
        else:
            gs = [global_max_pool(z, batch) for z in zs]

        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

class Encoder(nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.aug1, self.aug2 = augmentor

    def forward(self, x, edge_index, batch):
        x1, edge_index1, _ = self.aug1(x, edge_index)
        x2, edge_index2, _ = self.aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2

def train(encoder_model, contrast_model, dataloader, optimizer, rank):
    encoder_model.train()
    epoch_loss = 0
    for data in tqdm(dataloader, disable=(rank != 0)):
        data = data.to(rank)
        optimizer.zero_grad()
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=5)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def main(rank, world_size, config, words):

    # Confirm the device
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # Load train dataset
    print(f'Rank {rank}: Loading Dataset')
    train_dataset = []
    for i in tqdm(range(5)):
        train_data_i = torch.load(f'Dataset/Train_data_batch_{i+1}.pt')
        train_dataset.extend(train_data_i)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config['bs'], sampler=train_sampler, num_workers=0)
    print(f'Rank {rank}: Data import succeeded.')
    
    # Model definition
    aug = {1: A.FeatureMasking(pf=config['aug_p1']), 
           2: A.EdgeRemoving(pe=config['aug_p1']), 
           3: A.NodeDropping(pn=config['aug_p1']),
           4: SubTree(words, pn1=config['aug_p3_1'], pn2=config['aug_p3_2'], pn3=config['aug_p3_3']),
           5: A.RandomChoice([A.FeatureMasking(pf=config['aug_p1']),
                              SubTree(words, pn1=config['aug_p3_1'], pn2=config['aug_p3_2'], pn3=config['aug_p3_3'])],2),
           6: A.RandomChoice([A.FeatureMasking(pf=config['aug_p1']),
                              A.EdgeRemoving(pe=config['aug_p1']), 
                              A.NodeDropping(pn=config['aug_p1'])],3),
            }
    gconv = GConv(input_dim=300, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], initial_node=config['initial_node'], pool_mode=config['pool_mode']).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(A.Identity(), aug[5])).to(device)
    encoder_model = torch.nn.parallel.DistributedDataParallel(encoder_model, device_ids=[rank], find_unused_parameters=True)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=config['tau']), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=config['lr'])
    
    # Model training
    for epoch in range(1, config['epoch'] + 1):
        train_sampler.set_epoch(epoch)
        loss = train(encoder_model, contrast_model, train_dataloader, optimizer, rank)
        if rank == 0:
            print(f'Epoch {epoch}: Loss {loss}')
            if epoch > 20:
                torch.save(encoder_model.module.state_dict(), f"Model/model-run{config['run_number']}-{epoch}.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--aug_p1", type=float, required=True)
    parser.add_argument("--aug_p3_1", type=float, required=True)
    parser.add_argument("--aug_p3_2", type=float, required=True)
    parser.add_argument("--aug_p3_3", type=float, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--initial_node", type=str, required=True)
    parser.add_argument("--pool_mode", type=str, required=True)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    mp.set_sharing_strategy('file_system')
    world_size = torch.cuda.device_count()
    words = torch.load('Tools/word_11868.pt')
    words.append('unif')
    config = vars(args)

    mp.spawn(main, args=(world_size, config, words), nprocs=world_size, join=True)