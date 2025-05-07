import torch
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature

def find_leaf(x, edge_index):
    row, col = edge_index  
    leaf_nodes = set(range(len(x))) - set(row.tolist())  
    return list(leaf_nodes)

def find_parent(x, edge_index, leaf_nodes):
    leaf_parent_nodes = set()
    src, dst = edge_index
    leaf_nodes = torch.tensor(leaf_nodes, dtype=dst.dtype).to(dst.device)
    parent_indices = (dst.unsqueeze(1) == leaf_nodes).nonzero(as_tuple=True)[0]
    leaf_parent_nodes = src[parent_indices].unique().tolist()
    return leaf_parent_nodes if leaf_parent_nodes else False

def find_grandparent(x, edge_index, leaf_parent_nodes):
    leaf_grandparent_nodes = set()
    src, dst = edge_index
    leaf_parent_nodes = torch.tensor(leaf_parent_nodes, dtype=dst.dtype).to(dst.device)
    grandparent_indices = (dst.unsqueeze(1) == leaf_parent_nodes).nonzero(as_tuple=True)[0]
    leaf_grandparent_nodes = src[grandparent_indices].unique().tolist()
    return leaf_grandparent_nodes if leaf_grandparent_nodes else False

def sub_1(x, edge_index, leaf_nodes, words, pn):
    indices = torch.tensor(leaf_nodes, dtype=torch.long).to(edge_index.device)
    op_dict = {'N':1, 'V':2}
    words_tensor = torch.tensor([op_dict.get(word[0], 3) for word in words], dtype=torch.long).to(edge_index.device)
    first_chars = words_tensor[x[indices,:].squeeze()]
    mask = (first_chars != 3)
    filtered_indices = indices[mask]

    mask = torch.bernoulli(torch.full((len(filtered_indices),), pn)).bool().to(edge_index.device)
    x[filtered_indices[mask]] = 11868
    return x, edge_index

def sub_2(x, edge_index, leaf_parent_nodes, words, pn):
    indices = torch.tensor(leaf_parent_nodes, dtype=torch.long).to(edge_index.device)
    op_dict = {'U!plus':1, 'O!minus':2, 'U!times':3, 'O!divide':4, 'O!SUB':5, 'O!SUP':6}
    words_tensor = torch.tensor([op_dict.get(word, 7) for word in words], dtype=torch.long).to(edge_index.device)
    first_chars = words_tensor[x[indices,:].squeeze()]
    mask = (first_chars != 7)
    filtered_indices = indices[mask]

    mask = torch.bernoulli(torch.full((len(filtered_indices),), pn)).bool().to(edge_index.device)
    selected_indices = filtered_indices[mask]
    x[selected_indices] = 11868

    src, dst = edge_index
    mask = ~torch.isin(src, selected_indices)
    edge_index = edge_index[:, mask]
    return x, edge_index

def sub_3(x, edge_index, leaf_grandparent_nodes, words, pn):
    indices = torch.tensor(leaf_grandparent_nodes, dtype=torch.long).to(edge_index.device)
    op_dict = {'U!plus':1, 'O!minus':2, 'U!times':3, 'O!divide':4, 'O!SUB':5, 'O!SUP':6}
    words_tensor = torch.tensor([op_dict.get(word, 7) for word in words], dtype=torch.long).to(edge_index.device)
    first_chars = words_tensor[x[indices,:].squeeze()]
    mask = (first_chars != 7)
    filtered_indices = indices[mask]

    mask = torch.bernoulli(torch.full((len(filtered_indices),), pn)).bool().to(edge_index.device)
    selected_indices = filtered_indices[mask]
    x[selected_indices] = 11868

    src, dst = edge_index
    mask = ~torch.isin(src, selected_indices)
    edge_index = edge_index[:, mask]
    return x, edge_index

class SubTree(Augmentor):
    def __init__(self, words, pn1: float, pn2: float, pn3: float):
        super(SubTree, self).__init__()
        self.pn1 = pn1
        self.pn2 = pn2
        self.pn3 = pn3
        self.words = words
    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weight = g.unfold()
        leaf_nodes = find_leaf(x, edge_index)
        leaf_parent_nodes = find_parent(x, edge_index, leaf_nodes)
        if leaf_parent_nodes!=False:
            leaf_grandparent_nodes = find_grandparent(x, edge_index, leaf_parent_nodes)
            if leaf_grandparent_nodes!=False:
                x, edge_index = sub_1(x, edge_index, leaf_nodes, self.words, self.pn1)
                x, edge_index = sub_2(x, edge_index, leaf_parent_nodes, self.words, self.pn2)
                x, edge_index = sub_3(x, edge_index, leaf_grandparent_nodes, self.words, self.pn3)
            else:
                x, edge_index = sub_1(x, edge_index, leaf_nodes, self.words, self.pn1)
                x, edge_index = sub_2(x, edge_index, leaf_parent_nodes, self.words, self.pn2)
        else:
            x, edge_index = sub_1(x, edge_index, leaf_nodes, self.words, self.pn1)

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weight)



   

