import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import os
from tqdm import tqdm

class Warehouse2Dataset(Dataset):
    
    def __init__(self, path = None, enc_path = None, cat = None):
        
        s = time.time()
        
        if path is None:
            path = "/home/beast/Desktop/vlassis/retrieval/experiments/data/warehouse2.h5"
            
        self.f = h5py.File(path, 'r')
        self.index = 0
        self.cat = cat
        self.encodings = torch.load(enc_path)["data"] if enc_path is not None else None
        
        if cat is None:
            self. f = self.process()
        else:
            if isinstance(cat, int):
                cat = [cat]
            self.f = self.subset(cat)
            
        print(f"Warehouse2 dataset initialization complete (t = {time.time() - s})")  
    
    def process(self):
        
        newdata = {}
        
        new_objectlabels = torch.from_numpy(self.f["object_labels"][:])
        new_partlabels = torch.from_numpy(self.f["part_labels"][:])
        
        self.size = new_objectlabels.shape[0]
        
        for i in range(self.size):
            newdata[str(i)] = self.f["data"][str(i)][:]
            
        return {
            "data": newdata,
            "object_labels": new_objectlabels,
            "part_labels": new_partlabels,
        }
                                                       
    def subset(self, cat):
        
        assert isinstance(cat, list), "Expected type list but got " + str(type(cat)) + "instead."
    
        #finding all indices, where labels match one of the specified categories
        indices = np.where(np.isin(self.f["object_labels"][:], cat))[0].reshape(-1)

        #setting the dataset size
        self.size = indices.shape[0]

        #creating a new dataset, containing only the specified categories
        newdata = {}
        new_objectlabels = []
        new_partlabels = []
        
        for i, idx in enumerate(indices):
            newdata[str(i)] = self.f["data"][str(idx)][:]
            new_objectlabels.append(self.f["object_labels"][:][idx])
            new_partlabels.append(self.f["part_labels"][:][idx])
        
        #adding encodings as well, if they are available
        if self.encodings is not None:
            self.encodings = self.encodings[torch.from_numpy(indices)]

        return {
            "data": newdata,
            "object_labels": torch.Tensor(new_objectlabels),
            "part_labels": torch.Tensor(new_partlabels)
        }
    
    def __len__(self):
    
        return self.size
    
    def __getitem__(self, idx):
        
        if self.encodings is not None:
            return self.f["data"][str(idx)], self.f["object_labels"][idx], self.f["part_labels"][idx], self.encodings[idx]
        else:
            return self.f["data"][str(idx)], self.f["object_labels"][idx], self.f["part_labels"][idx]
    
    def __next__(self):
        
        if self.index < self.size:
            result = (self.f["data"][str(self.index)], self.f["object_labels"][self.index], self.f["part_labels"][self.index], self.encodings[self.index]) if self.encodings is not None \
                else (self.f["data"][str(self.index)], self.f["object_labels"][self.index], self.f["part_labels"][self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def __iter__(self):
        
        self.index = 0
        return self

def warehouse2_collate_fn(batch):

    #batch: list of tuples (part, object_label, part_label, encoded_sample?)
    
    parts = []
    object_labels = []
    part_labels = []
    encoded_samples = []

    for tup in batch:
        parts.append(torch.from_numpy(tup[0]))
        object_labels.append(tup[1])
        part_labels.append(tup[2])
        
    if len(batch[0]) == 4:
        for tup in batch:
            encoded_samples.append(tup[3])
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels), torch.stack(encoded_samples)
    else:
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels)

class Warehouse3Dataset(Dataset): 
    
    def __init__(self, path = None, enc_path = None, cat = None):
        
        s = time.time()
        
        if path is None:
            path = "/home/beast/Desktop/vlassis/retrieval2/experiments/data/vectors_warehouse2_partnet.h5"
            
        self.f = h5py.File(path, 'r')
        self.index = 0
        self.cat = cat
        self.encodings = torch.load(enc_path)["data"] if enc_path is not None else None
        
        if cat is None:
            self. f = self.process()
        else:
            if isinstance(cat, int):
                cat = [cat]
            self.f = self.subset(cat)
            
        print(f"Warehouse3 dataset initialization complete (t = {time.time() - s})")  
    
    def process(self):
        
        newdata = {}
        
        new_objectlabels = torch.from_numpy(self.f["object_labels"][:])
        new_partlabels = torch.from_numpy(self.f["part_labels"][:])
        new_vectors = torch.from_numpy(self.f["vectors"][:])

        self.size = new_objectlabels.shape[0]
        
        for i in range(self.size):
            newdata[str(i)] = self.f["data"][str(i)][:]
            
        return {
            "data": newdata,
            "object_labels": new_objectlabels,
            "part_labels": new_partlabels,
            "vectors": new_vectors
        }
                                                       
    def subset(self, cat):
        
        assert isinstance(cat, list), "Expected type list but got " + str(type(cat)) + "instead."
    
        #finding all indices, where labels match one of the specified categories
        indices = np.where(np.isin(self.f["object_labels"][:], cat))[0].reshape(-1)

        print(indices.shape)
        #setting the dataset size
        self.size = indices.shape[0]

        #creating a new dataset, containing only the specified categories
        newdata = {}
        new_objectlabels = []
        new_partlabels = []
        new_vectors = []
        
        for i, idx in enumerate(indices):
            newdata[str(i)] = self.f["data"][str(idx)][:]
            new_objectlabels.append(self.f["object_labels"][:][idx])
            new_partlabels.append(self.f["part_labels"][:][idx])
            new_vectors.append(self.f["vectors"][:][idx])

        #adding encodings as well, if they are available
        if self.encodings is not None:
            self.encodings = self.encodings[torch.from_numpy(indices)]

        return {
            "data": newdata,
            "object_labels": torch.Tensor(new_objectlabels),
            "part_labels": torch.Tensor(new_partlabels),
            "vectors": torch.Tensor(new_vectors)
        }
    
    def __len__(self):
    
        return self.size
    
    def __getitem__(self, idx):
        
        if self.encodings is not None:
            return self.f["data"][str(idx)], self.f["object_labels"][idx], self.f["part_labels"][idx], self.f["vectors"][idx], self.encodings[idx]
        else:
            return self.f["data"][str(idx)], self.f["object_labels"][idx], self.f["part_labels"][idx], self.f["vectors"][idx]
    
    def __next__(self):
        
        if self.index < self.size:
            result = (self.f["data"][str(self.index)], self.f["object_labels"][self.index], self.f["part_labels"][self.index], self.f["vectors"][self.index], self.encodings[self.index]) if self.encodings is not None \
                else (self.f["data"][str(self.index)], self.f["object_labels"][self.index], self.f["part_labels"][self.index], self.f["vectors"][self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def __iter__(self):
        
        self.index = 0
        return self

def warehouse3_collate_fn(batch):

    #batch: list of tuples (part, object_label, part_label, encoded_sample?)
    
    parts = []
    object_labels = []
    part_labels = []
    vectors = []
    encoded_samples = []

    for tup in batch:
        parts.append(torch.from_numpy(tup[0]))
        object_labels.append(tup[1])
        part_labels.append(tup[2])
        vectors.append(tup[3])

    if len(batch[0]) == 5:
        for tup in batch:
            encoded_samples.append(tup[4])
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels), torch.stack(vectors), torch.stack(encoded_samples)
    else:
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels), torch.stack(vectors)

#BUILT TO NOT BE IN-MEMORY
class Warehouse4Dataset(Dataset): 
    
    def __init__(self, path = None, encodings = False, cat = None):
        
        print("Initializing warehouse dataset")
        s = time.time()
        
        #path fixing
        if path is None:
            self.path = "/home/beast/Desktop/vlassis/retrieval2/experiments/data/vectors_warehouse_partnet"
        else:
            self.path = path

        #category selection
        if cat is None:
            self.cat = [int(i) for i in sorted(os.listdir(self.path))]
        else:
            self.cat = cat

        #keeping track of the number of samples per category
        self.size_map = {
            str(c) : len(os.listdir(os.path.join(self.path, str(c)))) for c in self.cat
        }

        #total number of samples
        self.size = sum([self.size_map[str(c)] for c in self.cat])

        #running index, for dataset iteration
        self.index = 0
            
        #boolean flag, indicating whether encodings should be used
        self.encodings = encodings

        #asserting that encodings are present when the flag is true
        if self.encodings == True:

            for i in tqdm(range(self.size)):
                f = self.get_sample(i)
                assert "encoding" in f.keys()


        print(f"Warehouse dataset initialization complete (t = {time.time() - s})")  
    
    def get_sample(self, idx):

        source_dir = ""
        for key, val in self.size_map.items():
            if idx >= val:
                idx -= val
            else:
                source_dir = key
                break

        f = torch.load(os.path.join(self.path, source_dir, str(idx) + ".pt"))
        
        return f

    def add_encoding(self, idx, enc):

        f = self.get_sample(idx)
        f["encoding"] = enc

        source_dir = ""
        for key, val in self.size_map.items():
            if idx >= val:
                idx -= val
            else:
                source_dir = key
                break

        torch.save(f, os.path.join(self.path, source_dir, str(idx) + ".pt"))

    def __len__(self):
    
        return self.size
    
    def __getitem__(self, idx):
        
        f = self.get_sample(idx)

        if self.encodings:
            return f["sample"], f["object_label"], f["part_label"], f["vector"], f["encoding"]
        else:
            return f["sample"], f["object_label"], f["part_label"], f["vector"]

    def __next__(self):
        
        if self.index < self.size:

            f = self.get_sample(self.index)
            self.index += 1

            if self.encodings:
                return f["sample"], f["object_label"], f["part_label"], f["vector"], f["encoding"]
            else:
                return f["sample"], f["object_label"], f["part_label"], f["vector"]
            
        else:
            raise StopIteration
    
    def __iter__(self):
        
        self.index = 0
        return self

def warehouse4_collate_fn(batch):

    #batch: list of tuples (part, object_label, part_label, vector, encoded_sample?)
    
    parts = []
    object_labels = []
    part_labels = []
    vectors = []
    encoded_samples = []

    for tup in batch:
        parts.append(torch.from_numpy(tup[0]))
        object_labels.append(tup[1])
        part_labels.append(tup[2])
        vectors.append(torch.from_numpy(tup[3]))

    if len(batch[0]) == 5:
        for tup in batch:
            encoded_samples.append(tup[4])
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels), torch.stack(vectors), torch.stack(encoded_samples)
    else:
        return parts, torch.Tensor(object_labels), torch.Tensor(part_labels), torch.stack(vectors)


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

class Items3Dataset(Dataset):
    
    def __init__(self, path = None, cat = None):
        
        s = time.time()
        
        if path is None:
            path = '/home/beast/Desktop/vlassis/retrieval2/experiments/data/vectors_items2_partnet.h5'
            
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
        vectors = torch.from_numpy(self.f["vectors"][:])
        objects = torch.from_numpy(self.f["data"][:])
        
        self.size = new_objectlabels.shape[0]
        
        return {
            "data": objects,
            "labels": new_objectlabels,
            "parts": new_partlabels,
            "pid": new_pid,
            "vectors": vectors
        }
                                                   
    def subset(self, cat):
        
        assert isinstance(cat, list), "Expected type list but got " + str(type(cat)) + "instead."
    
        #finding all indices, where labels match one of the specified categories
        indices = np.where(np.isin(self.f["labels"][:], cat))[0].reshape(-1)

        print(self.f["labels"][:].shape)

        #setting the dataset size
        self.size = indices.shape[0]

        #creating a new dataset, containing only the specified categories
        newdata = torch.from_numpy(self.f["data"][:])[indices]
        new_objectlabels = torch.from_numpy(self.f["labels"][:])[indices]
        new_partlabels = torch.from_numpy(self.f["parts"][:])[indices]
        new_pid = torch.from_numpy(self.f["pid"][:])[indices]
        new_vectors = torch.from_numpy(self.f["vectors"][:])[indices]

        return {
            "data": newdata,
            "labels": new_objectlabels,
            "parts": new_partlabels,
            "pid": new_pid,
            "vectors": new_vectors
        }
    
    def __len__(self):
    
        return self.size
    
    def __getitem__(self, idx):
        
        return self.f["data"][idx].float(), self.f["labels"][idx], self.f["parts"][idx], self.f["pid"][idx], self.f["vectors"][idx]
    
    def __next__(self):
        
        if self.index < self.size:
            result = (self.f["data"][self.index], self.f["labels"][self.index], self.f["parts"][self.index], self.f["pid"][self.index], self.f["vectors"][self.index])
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
    
    dataset = Items2Dataset(cat=[0, 1, 2], path = "/home/beast/Desktop/vlassis/retrieval2/experiments/data/items2_partnet.h5")

    global_max = 0

    for i, stuff in enumerate(dataset):

        pid = stuff[-1]
        global_max = max(global_max, pid.max())

    
    print(global_max)