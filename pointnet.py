import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class PartMaxPooling(MessagePassing):
    """ A block to incorporate the time embedding information to a sparse representation of node features. 
        Use to intergrade the time embedding information to sparse voxels
    """

    def __init__(self):
        super().__init__('max', flow='target_to_source')

    def forward(self, x, pid, num_parts):
        # x: N x F features of each point
        # pid: a tensor indicate the batch index of each node

        node_idx = torch.arange(0, x.shape[0], device=x.device)
        edge_index = torch.stack([pid, node_idx]).long()
        out = self.propagate(edge_index=edge_index, x=x)

        return out[:num_parts]
    
    def message(self, x_j):
        return x_j

class PointnetBIG(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, 1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, 1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 512, 1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, 1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x, pid):

        '''
            x: torch.tensor(B, 3, N)
            pid: torch.tensor(B, N)
        '''

        #B x 3 x N -> B x F x N
        feats = self.mlp(x)
        
        part_feats = []
        max_num_parts = pid.max() + 1
        B, F, _ = feats.shape


        for i in range(B):
            current_shape_feats = []
            for j in range(pid[i].max() + 1):

                current_shape_feats.append(feats[i][:, pid[i] == j].max(dim=-1).values)

            current_shape_feats += [torch.zeros(F).to(feats.device)] * (max_num_parts - len(current_shape_feats))
            part_feats.append(torch.stack(current_shape_feats))
    
        #B x M x F
        return torch.stack(part_feats)

    def forward_parts(self, x, pid, part_labels):

        '''
            x: torch.tensor(B, 3, N)
            pid: torch.tensor(B, N)
            part_labels: torch.tensor(B, N)
        '''

        if x.dim() == 2: raise ValueError("x must be 3-dimensional. Got x:{}".format(x.shape))
        if x.shape[-1] == 3: x = x.permute(0,2,1)

        #B x 3 x N -> B x F x N
        feats = self.mlp(x)
        
        part_feats = []
        part_lbs = [] 
        max_num_parts = pid.max() + 1
        B, F, _ = feats.shape

        for i in range(B):
            current_shape_feats = []
            current_part_labels = []

            #adding part feature vectors (after per-part pooling) and corresponding labels
            for j in range(pid[i].max() + 1):
                current_shape_feats.append(feats[i][:, pid[i] == j].max(dim=-1).values)
                current_part_labels.append(part_labels[i][pid[i] == j].unique().item())

            #padding to match the maximum number of parts per shape
            current_shape_feats += [torch.zeros(F).to(feats.device)] * (max_num_parts - len(current_shape_feats))
            current_part_labels += [-1] * (max_num_parts - len(current_part_labels))

            #adding to the list
            part_feats.append(torch.stack(current_shape_feats))
            part_lbs.append(torch.Tensor(current_part_labels))

        #B x M x F, B x M
        return torch.stack(part_feats), torch.stack(part_lbs).long().to(part_labels.device)

    def forward_parts_only(self, x, pid, part_labels):

        '''
            x: torch.tensor(B, N, 3)
            pid: torch.tensor(B, N)
            part_labels: torch.tensor(B, N)
        
            
            out: torch.tensor(M, F), where M is the total 
            number of parts in the entire batch
        '''

        if x.dim() == 2: raise ValueError("x must be 3-dimensional. Got x:{}".format(x.shape))
        if x.shape[-1] == 3: x = x.permute(0,2,1)

        #B x 3 x N -> B x F x N
        feats = self.mlp(x)

        part_feats = []
        part_lbs = []
        B, F, _ = feats.shape

        #iterating shapes
        for i in range(B):

            #iterating parts (+1 because pid is 0 indexed)
            for j in range(pid[i].max()+1):
                part_feats.append(feats[i][:, pid[i] == j].max(dim=-1).values)
                part_lbs.append(part_labels[i][pid[i] == j].unique().item())

        #M x F, M
        return torch.stack(part_feats), torch.Tensor(part_lbs).long().to(part_labels.device)

    def forward_parts_only2(self, x, pid, part_labels):

        '''
            x: torch.tensor(N, 3)
            pid: torch.tensor(N)
            part_labels: torch.tensor(N)
        
            
            out: torch.tensor(M, F), where M is the total 
            number of parts in the entire batch
        '''

        if x.dim() == 2: raise ValueError("x must be 3-dimensional. Got x:{}".format(x.shape))
        if x.shape[-1] == 3: x = x.permute(0,2,1)

        #B x 3 x N -> B x F x N
        feats = self.mlp(x)

        part_feats = []
        part_lbs = []
        B, F, _ = feats.shape

        #iterating shapes
        for i in range(B):

            #iterating parts (+1 because pid is 0 indexed)
            for j in range(pid[i].max()+1):
                part_feats.append(feats[i][:, pid[i] == j].max(dim=-1).values)
                part_lbs.append(part_labels[i][pid[i] == j].unique().item())

        #M x F, M
        return torch.stack(part_feats), torch.Tensor(part_lbs).long().to(part_labels.device)

class PointnetNew(torch.nn.Module):

    '''
        Newer version of the original model,
        includes a graph based max pooling for better speed,
        linear layers after the max pooling for better interpoint relationship processing
        and requires the parts to be stacked on top of each other, omitting the batch dimension.
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512)
        )
        self.mp = PartMaxPooling()
        self.mlp2 = nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, out_channels)
        )

    def forward(self, x, pid, max_num_parts):

        x = self.mlp1(x)
        x = self.mp(x, pid, max_num_parts)
        x = self.mlp2(x)

        return x

if __name__ == "__main__":

    from utils import normalize_and_split

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = PointnetNew(3).cuda()
    print("model total parameters: ", count_parameters(model))

    x = torch.rand(2, 10, 3).cuda()

    pid = torch.Tensor([[0,0,0,1,1,1,2,2,2,2], [0,0,0,0,1,1,2,2,2,2]]).long().cuda()
    part_labels = torch.Tensor([[0,0,0,1,1,1,2,2,2,2], [0,0,0,0,1,1,2,2,2,2]]).long().cuda()
    x, pid, part_labels, max_num_parts = normalize_and_split(x, pid, part_labels)
    out = model(x, pid, max_num_parts)
    print(out.shape)

    # input = torch.rand(1, 349, 3).cuda()
    # pid = torch.zeros(1, input.shape[1]).long()

    # out = model(input.permute(0,2,1), pid)

    # print(out.shape)
    # print(out)