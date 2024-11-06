from typing import Any
import torch
from scripts.extensions import chamfer_dist

class AccuracyBinary():

    def __init__(self):
        
        self.positive = 0
        self.all = 0

    def __call__(self, out, label):
        
        #out: torch.tensor(N) scores
        #label: torch.tensor(N) labels

        self.positive += (out > 0.5).eq(label).sum().item()
        self.all += len(label)

    def get(self):
        
        return self.positive / self.all
    
class AccuracyMultiClass():

    def __init__(self):
        
        self.positive = 0
        self.all = 0

    def __call__(self, out, label):
        
        #out: torch.tensor(B, N, K) scores
        #label: torch.tensor(B, N) labels

        self.positive += out.argmax(dim=-1).eq(label).sum().item()
        self.all += len(label.flatten())

    def get(self):
        
        return self.positive / self.all

class ChamferDistance():
    
        def __init__(self):
            
            self.distance = 0
            self.all = 0
            self.crit = chamfer_dist.ChamferDistanceL2()
    
        def __call__(self, out, gt):
            
            #out: torch.tensor(B, N, 3) points
            #gt: torch.tensor(B, M, 3) points
    
            B = out.shape[0]
            assert B == gt.shape[0]

            self.distance += self.crit(out, gt).cpu()
            self.all += B
    
        def get(self):
            
            return self.distance / self.all

if __name__ == "__main__":

    acc = AccuracyMultiClass()

    scores = torch.rand(2, 2, 2)
    labels = torch.tensor([[0, 1], [1, 0]])

    print(scores)
    print(labels)

    acc(scores, labels)
    print(acc.get())