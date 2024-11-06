import torch
import os
import open3d as o3d
from .visualization import rotation_matrix_from_vectors2, quick_vis, coord_frame

def round_tensor_to_k_digits(tensor, k=2):
    # Multiply the tensor by 10^k to shift the decimal point to the right
    shifted_tensor = tensor * (10 ** k)
    
    # Round the shifted tensor to the nearest integer
    rounded_tensor = torch.round(shifted_tensor)
    
    # Divide by 10^k to shift the decimal point back to its original position
    rounded_tensor /= 10 ** k
    
    return rounded_tensor

#replaces any label with an increasing integer 0-N e.g: labels = [0,1,5,9,1] -> [0,1,2,3,1]
def convert_labels(labels):
    
    assert len(labels.shape) == 1
    categories = torch.unique(labels)
    new_labels = torch.zeros_like(labels)
    
    for i, l in enumerate(categories):
        new_labels[labels == l] = i
    
    return new_labels

def map_labels(labels, lmap):

    assert isinstance(lmap, dict)
    assert len(labels.shape) < 2 

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)

    new_labels = torch.zeros_like(labels)

    for l in labels.unique():
        new_labels[labels == l] = lmap[str(int(l.item()))]

    return new_labels

def generate_label_map(labels):

    assert len(labels.shape) == 1
    categories = torch.unique(labels)
    map = {}

    for i, l in enumerate(categories):
        map[str(int(l.item()))] = i

    return map

def pad_parts(parts, pad_point):

    max_points = max([part.shape[0] for part in parts])
    padded_parts = torch.zeros(len(parts), max_points, 3).to(pad_point.device)

    for i, part in enumerate(parts):
        padded_part = torch.cat((part.to(pad_point.device), pad_point.repeat(max_points - part.shape[0], 1)), dim=0)
        padded_parts[i, :, :] = padded_part
    
    return padded_parts

def create_pad_mask(part_labels):

    '''
        part_labels: B*N
        Create a mask BN*BN where a 0 indicates that the row and column correspond to padding
        All other values are 1
    '''


    part_labels = part_labels.unsqueeze(0).repeat(part_labels.shape[0], 1)
    part_labels = part_labels == -1

    part_labels = ~(part_labels + part_labels.T)

    return part_labels.long()

#given a B x N x 3 batch of point clouds with corresponding Pids, returns a K x M x 3 tensor of padded parts 
def split_into_parts(pcs, pids, pad_point, mask=False):

    assert len(pcs.shape) == 3
    assert len(pids.shape) == 2
    assert len(pad_point.shape) == 1

    #pcs: B x N x 3
    #pids: B x N
    #pad_point: 3

    max_num_parts = pids.max() + 1
    parts = []
    active_elements = []
    pad = pad_point.clone().unsqueeze(0).to(pcs.device)

    for i, (point_cloud, pid) in enumerate(zip(pcs, pids)):

        for j in pid.unique():
            part = point_cloud[pid == j]
            parts.append(part)
            active_elements.append(1)
        
        #padding to match the maximum number of parts per shape
        parts += [pad] * (max_num_parts - pid.max()-1)
        active_elements += [0] * (max_num_parts - pid.max()-1)

    max_num_points = max([part.shape[0] for part in parts])

    padded_parts = torch.zeros(len(parts), max_num_points, 3).to(pcs.device)
    point_mask = torch.zeros(len(parts), max_num_points+1).to(pcs.device) #+1 to account for the extra padded point of each point cloud

    for i, part in enumerate(parts):

        padded_part = torch.cat((part, pad_point.repeat(max_num_points - part.shape[0], 1)), dim=0)
        point_mask[i, :part.shape[0]] = 1
        padded_parts[i, :, :] = padded_part


    #adding an additional pad point, so that ALL point clouds, have at least one.
    pad_row = pad.reshape(1, 1, -1).repeat(padded_parts.shape[0], 1, 1)
    padded_parts = torch.cat((padded_parts, pad_row), dim=1)

    if mask:
        return padded_parts, point_mask, torch.Tensor(active_elements).to(torch.int32).to(pcs.device)

    #maybe the extra pad point needs to be reflected in the point mask as well?
    #B * max_num_parts x max_num_points + 1 x 3
    return padded_parts, point_mask

#given a B x N x 3 batch of point clouds with corresponding Pids, returns a M x 3 tensor of parts, padded with a given pad_point to match in size
def split_into_parts_without_padding(pcs, pids, pad_point):

    assert len(pcs.shape) == 3
    assert len(pids.shape) == 2
    assert len(pad_point.shape) == 1

    #pcs: B x N x 3
    #pids: B x N
    #pad_point: 3

    parts = []
    pad = pad_point.clone().unsqueeze(0).to(pcs.device)

    #separating all the parts from the batch
    for i, (point_cloud, pid) in enumerate(zip(pcs, pids)):

        for j in pid.unique():
            part = point_cloud[pid == j]
            parts.append(part)

    #padding the parts so that they match in size
    max_num_points = max([part.shape[0] for part in parts])

    #preallocating memory for the parts and the point mask
    padded_parts = torch.zeros(len(parts), max_num_points, 3).to(pcs.device)

    #+1 to account for the extra padded point of each point cloud
    #All parts need to have at least one pad_point, so that the chamfer distance between them is calculated correctly
    point_mask = torch.zeros(len(parts), max_num_points+1).to(pcs.device) 

    #copying the parts to the padded_parts tensor
    for i, part in enumerate(parts):

        padded_part = torch.cat((part, pad_point.repeat(max_num_points - part.shape[0], 1)), dim=0)
        point_mask[i, :part.shape[0]] = 1
        padded_parts[i, :, :] = padded_part


    #adding an additional pad point, so that ALL point clouds, have at least one.
    pad_row = pad.reshape(1, 1, -1).repeat(padded_parts.shape[0], 1, 1)
    padded_parts = torch.cat((padded_parts, pad_row), dim=1)

    #The pad point position in the point mask is 0 by default
    
    #Μ x Ν + 1 x 3
    return padded_parts, point_mask

#given a B x Max x 3 batch of padded directional vectors, returns a M x 3 tensor of vectors for each actual part in the batch 
def split_vectors(batch_vectors, return_mask):

    '''
        batch_vectors: B x Max x 3
        out: K x 3

        Note: The 2nd dimension of the vectors, is not the max of the batch, but rather the whole dataset.
        A new max needs to be calculated.
    '''

    max_num_vectors = (batch_vectors.abs().sum(dim=-1) > 0).sum(dim=-1).max()
    vectors = batch_vectors[:, :max_num_vectors, :].reshape(-1, 3)
    
    mask = (vectors.abs().sum(dim=-1) > 0)

    if return_mask:
        return vectors, mask
    
    else:
        return vectors

def split_vectors_without_padding(batch_vectors):

    #reshaping the vectors to a 2D tensor
    vectors = batch_vectors.reshape(-1, 3)

    #discarding the 0 elements
    mask = (vectors.abs().sum(dim=-1) > 0)
    vectors = vectors[mask]

    return vectors

#normalizes parts given only parts and pid
def normalize_parts_1(parts, pid, return_centers = False):

    #parts: B x N x 3
    #pid: B x N

    max_num_parts = pid.max() + 1

    B = pid.shape[0]

    new_pid = []
    new_parts = []
    centers = []

    for i in range(B):

        new_pid_i = []
        new_parts_i = []
        centers_i = []

        for j in pid[i].unique().long():

            #separating part and normalizing it
            part = parts[i][pid[i] == j]
            centroid = part.mean(dim=0)
            part = part - centroid.unsqueeze(0)
            new_parts_i.append(part)
            centers_i.append(centroid)

            #rearranging pids
            new_pid_i.append(torch.ones(part.shape[0], dtype = torch.int32).to(pid.device) * j)
    
        new_pid.append(torch.cat(new_pid_i, dim=0))
        new_parts.append(torch.cat(new_parts_i, dim=0))
        centers_i += [torch.zeros(3).to(pid.device)] * (max_num_parts - len(centers_i))
        centers.append(torch.stack(centers_i))

    if return_centers:
        return torch.stack(new_parts), torch.stack(new_pid), torch.stack(centers)

    return torch.stack(new_parts), torch.stack(new_pid)

#normalizes parts but requires part labels to assure that they are properly aligned
def normalize_parts(parts, pid, part_labels, return_centers = False):

    #parts: B x N x 3
    #pid: B x N
    #part_labels: B x N

    max_num_parts = pid.max() + 1

    B = pid.shape[0]

    new_pid = []
    new_parts = []
    new_part_labels = []
    centers = []

    for i in range(B):

        new_pid_i = []
        new_parts_i = []
        new_part_labels_i = []
        centers_i = []

        for j in pid[i].unique().long():

            #separating part and normalizing it
            part = parts[i][pid[i] == j]
            centroid = part.mean(dim=0)
            part = part - centroid.unsqueeze(0)
            new_parts_i.append(part)
            centers_i.append(centroid)

            #finding corresponding part label
            label = part_labels[i][pid[i] == j].unique()
            new_part_labels_i.append(torch.ones(part.shape[0], dtype = torch.int32).to(pid.device) * label)

            #rearranging pids
            new_pid_i.append(torch.ones(part.shape[0], dtype = torch.int32).to(pid.device) * j)
    
        new_pid.append(torch.cat(new_pid_i, dim=0))
        new_parts.append(torch.cat(new_parts_i, dim=0))
        new_part_labels.append(torch.cat(new_part_labels_i, dim=0))
        centers_i += [torch.zeros(3).to(pid.device)] * (max_num_parts - len(centers_i))
        centers.append(torch.stack(centers_i))

    if return_centers:
        return torch.stack(new_parts), torch.stack(new_pid), torch.stack(new_part_labels), torch.stack(centers)

    return torch.stack(new_parts), torch.stack(new_pid), torch.stack(new_part_labels)

def normalize_and_split(shape, pid, part_labels, vectors, include_vectors = False, include_centroids=False, include_class_map=False):

    """
        shape: B x N x 3
        pid: B x N
        part_labels: B x N
        vectors: B x N x 3

        out: M x 3     
    """

    B, N, _ = shape.shape

    new_parts = []
    new_pid = []
    new_part_labels = []
    class_map = []
    centroids = []
    pid_total = 0

    for i in range(B):

        current_pid = pid[i]
        current_part_labels = part_labels[i]

        for j in current_pid.unique():

            #grabbing current part, , 
            part = shape[i][current_pid == j]
            
            #normalizing its translation
            cent = part.mean(dim=0)
            part = part - cent

            #aligning with the z-axis
            if vectors is not None:
                current_vector = vectors[i][j.item()]
                R = torch.from_numpy(
                    rotation_matrix_from_vectors2(
                        current_vector, torch.tensor([0,0,1])
                    )
                ).to(shape.device).to(part.dtype)

                part = (R.unsqueeze(0) @ part.unsqueeze(-1)).squeeze(-1)

            #adding it to the list
            new_parts.append(part)

            #saving the centroid
            centroids.append(cent)

            #generating a new pid for the part, and incrementing the total
            new_pid.append(torch.ones(part.shape[0], dtype = torch.int32).to(shape.device) * pid_total)
            pid_total += 1

            #adding part_label
            new_part_labels.append(current_part_labels[current_pid == j].unique().item())

            #adding the object class this current part belongs to
            class_map.append(i)

    parts = torch.cat(new_parts, dim=0)
    pid = torch.cat(new_pid, dim=0)
    part_labels = torch.tensor(new_part_labels).to(shape.device)
    
    out = [parts, pid, part_labels]

    if include_vectors:
        vectors = split_vectors_without_padding(vectors)
        out.append(vectors)

    if include_centroids:
        out.append(torch.stack(centroids).to(shape.device))

    if include_class_map:
        out.append(torch.Tensor(class_map).to(shape.device))

    return out

def subsample_parts(parts, pid, part_labels, vectors, num_parts_keep):

    '''
        parts: N x 3
        pid: N
        part_labels: N
        vectors: M x 3
        num_parts_keep: int

        out:  num_parts_keep x 3, num_parts_keep, num_parts_keep
    '''

    assert vectors.shape[0] == pid.max() + 1

    new_parts = []
    new_pid = []
    max_num_parts = pid.max() + 1
    perm = torch.randperm(max_num_parts)[:num_parts_keep]

    for i, id in enumerate(perm):

        part = parts[pid == id]
        new_parts.append(part)
        new_pid.append(torch.ones(part.shape[0], dtype = torch.int32).to(parts.device) * i)

    parts = torch.cat(new_parts, dim=0)
    pid = torch.cat(new_pid, dim=0)
    part_labels = part_labels[perm]
    vectors = vectors[perm]

    return parts, pid, part_labels, vectors

def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gen_one_hot(B, M, d):

    return torch.cat((torch.eye(M), torch.zeros(M, d - M)), dim=-1).unsqueeze(0).repeat(B, 1, 1)

class BoundingBox():

    def __init__(self, center, pmin, pmax):

        self.center = center
        self.xmin = pmin[0]
        self.ymin = pmin[1]
        self.zmin = pmin[2]
        self.xmax = pmax[0]
        self.ymax = pmax[1]
        self.zmax = pmax[2]

    def scale(self, factor):
        
        #subtracting center from the extreme points
        self.xmin -= self.center[0]
        self.ymin -= self.center[1]
        self.zmin -= self.center[2]
        self.xmax -= self.center[0]
        self.ymax -= self.center[1]
        self.zmax -= self.center[2]

        #scaling the extreme points
        self.xmin *= factor
        self.ymin *= factor
        self.zmin *= factor
        self.xmax *= factor
        self.ymax *= factor
        self.zmax *= factor

        #adding the center back
        self.xmin += self.center[0]
        self.ymin += self.center[1]
        self.zmin += self.center[2]
        self.xmax += self.center[0]
        self.ymax += self.center[1]
        self.zmax += self.center[2]

    def contains(self, pts):

        #pts: N x 3

        mx1 = pts[:,0] >= self.xmin
        mx2 = pts[:,0] <= self.xmax
        my1 = pts[:,1] >= self.ymin
        my2 = pts[:,1] <= self.ymax
        mz1 = pts[:,2] >= self.zmin
        mz2 = pts[:,2] <= self.zmax

        return mx1 * mx2 * my1 * my2 * mz1 * mz2

    @property
    def _o3d(self):

        pts = torch.tensor([
            [self.xmin, self.ymin, self.zmin],
            [self.xmin, self.ymin, self.zmax],
            [self.xmin, self.ymax, self.zmin],
            [self.xmin, self.ymax, self.zmax],
            [self.xmax, self.ymin, self.zmin],
            [self.xmax, self.ymin, self.zmax],
            [self.xmax, self.ymax, self.zmin],
            [self.xmax, self.ymax, self.zmax]
        ])

        o3d_bb = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(pts),
            lines = o3d.utility.Vector2iVector([
                [0,1], [0,2], [0,4], [1,3], [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]
            ])
        )

        return o3d_bb

class NormalizeTranslation():

    def __init__(self):
        pass

    def __call__(self, x):

        if x.dim() == 3:
            x = x - x.mean(dim=1).unsqueeze(1)
        elif x.dim() == 2:
            x = x - x.mean(dim=0).unsqueeze(0)
        
        return x

class NormalizeScale():

    def __init__(self):
        pass

    def __call__(self, x):
        
        abs = (x * x).sum(dim=-1).max(dim=-1).values.sqrt()

        if x.dim() == 3:
            x = x / abs.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 2:
            x = x / abs.unsqueeze(-1)
        
        return x

def pc_to_bounding_box(pc):

    '''
        pc: N x 3
    '''

    pc = pc.squeeze().cpu().detach()

    center = pc.mean(dim=0)
    xmin, ymin, zmin = pc.min(dim=0).values
    xmax, ymax, zmax = pc.max(dim=0).values

    return BoundingBox(center, torch.Tensor([xmin, ymin, zmin]), torch.Tensor([xmax, ymax, zmax]))

if __name__ == "__main__":

    #-----------------------------

    # parts = [
    #     torch.zeros(3,3), torch.zeros(5,3), torch.zeros(2,3)
    # ]

    # print(pad_parts(parts, torch.ones(3)))

    #-----------------------------
    import sys
    sys.path.append("/home/beast/Desktop/vlassis/retrieval/experiments/")
    from scripts.extensions import chamfer_dist
    from scripts.visualization import quick_vis
    from tqdm import tqdm
    import open3d as o3d
    import numpy as np

    # criterion = chamfer_dist.ChamferDistanceL2_unr()

    # class model(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.params = torch.nn.Parameter(torch.randn(1, 1024, 3))

    #     def forward(self):
    #         return self.params
    

    # target = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution =  25)
    # target = torch.from_numpy(np.asarray(target.vertices)).cuda()[:1024].unsqueeze(0).float()
    # m = model().cuda()

    # opt = torch.optim.Adam(m.parameters(), lr=0.0001)

    # quick_vis(m.params.clone().cpu().detach().squeeze())
    # quick_vis(target.clone().cpu().detach().squeeze())
    
    # print(criterion(m(), target))
    # losses = []

    # for i in tqdm(range(100000)):
    #     opt.zero_grad()
    #     loss = criterion(m(), target)
    #     print(loss[0].shape, loss[1].shape)
    #     exit()
    #     loss.backward()
    #     opt.step()

        
    # quick_vis(m.params.clone().cpu().detach().squeeze())
    # quick_vis(target.clone().cpu().detach().squeeze())

    #-----------------------------
    from scripts.dataset import Items2Dataset
    from torch.utils.data import DataLoader

    dset = Items2Dataset(cat = [0,3,4])
    dataloader = DataLoader(dset, batch_size=2, shuffle=True)

    batch = next(iter(dataloader))
    parts, _, part_labels, pid = batch
    
    # print("PID:")
    # print(pid[0,:200])
    # print("PART LABELS:")
    # print(part_labels[0,:200])

    # parts, pid, part_labels = normalize_parts(parts, pid, part_labels)
    
    # print("PID:")
    # print(pid[0,:200])
    # print("PART LABELS:")
    # print(part_labels[0,:200])

    shapes = torch.randn(2, 10, 3)
    pids = torch.Tensor([
        [0,0,1,1,0,0,0,1,2,2],
        [0,1,1,1,1,0,2,2,3,3]
        ])
    part_labels = torch.Tensor([
        [10,10,11,11,10,10,10,11,12,12],
        [10,11,11,11,11,10,12,12,13,13]
        ])

    parts, pid, part_labels = normalize_parts(shapes, pids, part_labels)
    print(pid, part_labels)