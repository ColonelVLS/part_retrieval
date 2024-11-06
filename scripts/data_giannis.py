import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import open3d as o3d

def quick_vis(x, extra_geometries = [], title=None):

    assert isinstance(extra_geometries, list)

    o3d.visualization.draw_geometries([
        o3d.geometry.PointCloud(
            points = o3d.utility.Vector3dVector(x)
        )
    ] + extra_geometries, window_name=title if title is not None else "Open3D")

def quick_vis_with_parts(x, parts, title=None):

    if isinstance(parts, torch.Tensor):
        parts = parts.cpu().numpy()
    
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    num_parts = np.max(parts)
    palette = torch.rand(np.max(parts), 3).numpy()
    colors = np.zeros_like(x)

    for i in range(num_parts):
        colors[parts == i] = palette[i,:]
    
    point_cloud = o3d.geometry.PointCloud(
            points = o3d.utility.Vector3dVector(x),
        )

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([
        point_cloud
    ], window_name=title if title is not None else "Open3D")

#Same as ItemsDataset, but part labels are also included
class Items2Dataset(Dataset):
    
    def __init__(self, path = None, cat = None):
        
        s = time.time()
        
        if path is None:
            path = "/home/beast/Desktop/vlassis/retrieval/experiments/data/items2.h5"
            
        self.f = h5py.File(path, 'r')
        self.index = 0
        self.cat = cat
        
        if cat is None:
            self.f = self.process()
        else:
            #converting int to list, list of categories is accepted
            if isinstance(cat, int):
                cat = [cat]
            self.f = self.subset(cat)
            
        print(f"Items2 dataset initialization complete (t = {time.time() - s})")  
    
    def process(self):
        
        new_objectlabels = torch.from_numpy(self.f["labels"][:])
        new_pid = torch.from_numpy(self.f["pid"][:])
        new_partlabels = torch.from_numpy(self.f["parts"][:])
        objects = torch.from_numpy(self.f["data"][:])
        
        self.size = new_objectlabels.shape[0]
        
        return {
            "data": objects,
            "labels": new_objectlabels,
            "parts": new_partlabels,
            "pid": new_pid
        }
                                                   
    def subset(self, cat):
        
        assert isinstance(cat, list), "Expected type list but got " + str(type(cat)) + "instead."
    
        #finding all indices, where labels match one of the specified categories
        indices = np.where(np.isin(self.f["labels"][:], cat))[0].reshape(-1)

        #setting the dataset size
        self.size = indices.shape[0]

        #creating a new dataset, containing only the specified categories
        newdata = torch.from_numpy(self.f["data"][:])[indices]
        new_objectlabels = torch.from_numpy(self.f["labels"][:])[indices]
        new_partlabels = torch.from_numpy(self.f["parts"][:])[indices]
        new_pid = torch.from_numpy(self.f["pid"][:])[indices]

        return {
            "data": newdata,
            "labels": new_objectlabels,
            "parts": new_partlabels,
            "pid": new_pid
        }
    
    def __len__(self):
    
        return self.size
    
    def __getitem__(self, idx):
        
        return self.f["data"][idx].float(), self.f["labels"][idx], self.f["parts"][idx], self.f["pid"][idx]
    
    def __next__(self):
        
        if self.index < self.size:
            result = (self.f["data"][self.index], self.f["labels"][self.index], self.f["parts"][self.index], self.f["pid"][self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def __iter__(self):
        
        self.index = 0
        return self


if __name__ == '__main__':

    import visualization as vis
    import numpy as np
    
    path = "/home/beast/Desktop/vlassis/retrieval2/experiments/data/giannis_items2_partnet.h5"
    categories = [0, 1, 2, 3]
    dataset = Items2Dataset(cat=categories, path = path)

    bad_samples = 0
    for sample, _, _, pid in dataset:
        if pid.unique().shape[0] == 1:
            bad_samples += 1
        
    print(f"Bad samples: {bad_samples}")