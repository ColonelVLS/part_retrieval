import torch
from torch_geometric.utils import to_dense_batch
from .pointnet import PointnetBIG, PointnetNew
from .transformer import CrossAttention
from .visualization import quick_vis_many
from .sparse_transformer import CrossAttention as SparseCrossAttention

class PartFinderPipeline(torch.nn.Module):

    def __init__(self, in_channels=3, out_channels=384, num_classes=3, num_attention_blocks=3, pos_emb_dim=3, pool_method="cls_token_pool"):

        super().__init__()

        self.pos_emb_dim = pos_emb_dim
        self.num_feats = out_channels + self.pos_emb_dim

        #encodes each part, is cardinality invariant
        # self.encoder = PointnetB(in_channels, out_channels)
        self.encoder = PointnetBIG(in_channels, out_channels)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.num_feats))

        #model the relationship between parts
        self.rel = CrossAttention(self.num_feats, self.num_feats, num_attention_blocks, pool_method = pool_method)

        #classify the validity of the parts
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.num_feats, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

        self.description = """Pipeline: Pointnet Encoder -> Cross Attention -> MLP Classifier
            The pointnet expects a tensor of shape (B, N, 3) and a tensor of shape (B, N) containing the part ids.
            Each part must be prenormalized so that its centered around the origin.
            The pooling is performed using a classification token. Make sure that its always the first element of the 'x' tensor,
            in the input of the CrossAttention module. The center of each part should be kept in a separate tensor so that it can
            be passed to this module and be concatenated as a positional embedding.

            This and every other class must contain a 'from_config' method that returns an instance of the class, given a configuration dictionary.
            This will be used to instantiate the model from within the gui.


        """

    @classmethod
    def from_config(cls, config):
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        num_classes = config["num_classes"]
        att_blocks = config["num_attention_blocks"]
        pos_emb_dim = config["pos_emb_dim"]
        pool_method = config["pool_method"]

        return cls(in_channels, out_channels, num_classes, att_blocks, pos_emb_dim, pool_method)

    #RUNS ONLY THE ENCODER AND RETURNS THE ENCODED PARTS
    def forward_encoder(self, shape, pid, part_label):

        '''
            x: torch.tensor(B, 3, N)
            pid: torch.tensor(B, N)
            part_labels: torch.tensor(B, N)
        '''

        return self.encoder.forward_parts(shape, pid, part_label)

    def forward_encoder_wo_padding(self, shape, pid, part_label):

        '''
            x: torch.tensor(B, 3, N)
            pid: torch.tensor(B, N)
            part_labels: torch.tensor(B, N)
        '''

        return self.encoder.forward_parts_only(shape, pid, part_label)

    #STANDARD FORWARD & ENCODE
    def forward(self, x, pid, centroids, normalize=False):

        #x: torch.tensor(B, N, 3) point clouds
        #pid: torch.tensor(B, N) part ids corresponding to a maximum of M parts
        #centroids: torch.tensor(B, M, 3) the centroids of each part
        #The parts are expected to be normalized, centered around the origin

        #B x N x 3 -> B x 3 x N
        x = x.permute(0,2,1)

        #encoding the input parts B x 3 x N -> B x M x F
        encoded_parts = self.encoder(x, pid)

        #optional normalization
        if normalize:
            encoded_parts = torch.nn.functional.normalize(encoded_parts, p=2, dim=-1)

        #concatenating the centroids as positional embeddings -> B x M x (F + 3)
        encoded_parts = torch.cat([encoded_parts, centroids], dim=-1)

        #concatenating with the cls token
        cls_token = self.cls_token.repeat(encoded_parts.shape[0], 1, 1)
        encoded_parts = torch.cat([cls_token, encoded_parts], dim=1)

        #****** Make sure that the cls token is passed in the first augment, as the 0th element ******
        #modeling the relations between parts using attention B x M x F -> B x M x F
        relations = self.rel(encoded_parts[:, :-1], encoded_parts[:, -1].unsqueeze(1))

        #classification B x M x F -> B x M x K
        scores = self.cls(relations)


        return scores

    #
    def forward_retrieval(self, query_features, query_centroids, 
                          warehouse_features, warehouse_centroids, normalize=False):
        
        #For retrieval, all batch dimensions must be 1
        #query_features: 1 x N x F
        #query_centroids: 1 x N x 3
        #warehouse_features: 1 x M x F
        #warehouse_centroids: 1 x M x 3
        
        #optional normalization
        if normalize:
            query_features = torch.nn.functional.normalize(query_features, p=2, dim=-1)
            warehouse_features = torch.nn.functional.normalize(warehouse_features, p=2, dim=-1)
        
        #concatenating the centroids as positional embeddings -> 1 x N x (F + 3), 1 x M x (F + 3)
        query_features = torch.cat([query_features, query_centroids], dim=-1)
        warehouse_features = torch.cat([warehouse_features, warehouse_centroids], dim=-1)

        #concatenating with the cls token 1 x (N + 1) x (F + 3)
        cls_token = self.cls_token.repeat(query_features.shape[0], 1, 1)
        query_features = torch.cat([cls_token, query_features], dim=1)

        #****** Make sure that the cls token is passed in the first augment, as the 0th element ******
        #modeling the relations between parts using attention {B x (N + 1) x F, B x M x F} -> B x M x F
        relations = self.rel(query_features, warehouse_features)

        #classification B x M x F -> B x M x K
        scores = self.cls(relations)

        return scores

class PartFinderPipeline2(torch.nn.Module):

    '''
        Same as PartFinderPipeline but using the PointnetNew encoder
        which accepts parts without the batch dimension. This pipeline is
        for using a larger batch size, shuffling the parts randomly and keeping
        only a subset of the parts. 
    '''

    def __init__(self, in_channels=3, out_channels=384, num_classes=3, num_attention_blocks=3, pos_emb_dim=3, pool_method="cls_token_pool"):

        super().__init__()

        self.pos_emb_dim = pos_emb_dim
        self.num_feats = out_channels + self.pos_emb_dim

        #encodes each part, is cardinality invariant
        # self.encoder = PointnetB(in_channels, out_channels)
        self.encoder = PointnetNew(in_channels, out_channels)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.num_feats))

        #model the relationship between parts
        self.rel = SparseCrossAttention(self.num_feats, self.num_feats, num_attention_blocks, pool_method = pool_method)

        #classify the validity of the parts
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.num_feats, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

        self.description = """Pipeline: Pointnet Encoder -> Cross Attention -> MLP Classifier
            The pointnet expects a tensor of shape (B, N, 3) and a tensor of shape (B, N) containing the part ids.
            Each part must be prenormalized so that its centered around the origin.
            The pooling is performed using a classification token. Make sure that its always the first element of the 'x' tensor,
            in the input of the CrossAttention module. The center of each part should be kept in a separate tensor so that it can
            be passed to this module and be concatenated as a positional embedding.

            This and every other class must contain a 'from_config' method that returns an instance of the class, given a configuration dictionary.
            This will be used to instantiate the model from within the gui.


        """

    @classmethod
    def from_config(cls, config):
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        num_classes = config["num_classes"]
        att_blocks = config["num_attention_blocks"]
        pos_emb_dim = config["pos_emb_dim"]
        pool_method = config["pool_method"]

        return cls(in_channels, out_channels, num_classes, att_blocks, pos_emb_dim, pool_method)

    #RUNS ONLY THE ENCODER AND RETURNS THE ENCODED PARTS
    def forward_encoder(self, x, pid, max_num_parts = None):

        '''
            x: torch.tensor(N, 3)
            pid: torch.tensor(N)
            max_num_parts: int
        '''

        if max_num_parts is None:
            max_num_parts = pid.max() + 1

        return self.encoder(x, pid, max_num_parts)

    def forward_transformer(self, encoded_parts, centroids, batch_id, normalize=False):

        '''
            encoded_parts: torch.tensor(N, F)
            centroids: torch.tensor(N, 3)
            batch_id: torch.tensor(N)
            max_num_parts: int
        '''

        #optional normalization
        if normalize:
            encoded_parts = torch.nn.functional.normalize(encoded_parts, p=2, dim=-1)

        #concatenating the centroids as positional embeddings -> M x (F + 3)
        encoded_parts = torch.cat([encoded_parts, centroids], dim=-1)

        #padding into B x Pmax x (F + 3)
        encoded_parts, mask = to_dense_batch(encoded_parts, batch_id.to(torch.int64))
        B = encoded_parts.shape[0]

        #concatenating with the cls token B x Pmax x (F + 3)
        cls_token = self.cls_token.repeat(B, 1, 1)
        encoded_parts = torch.cat([cls_token, encoded_parts], dim=1)

        #adding a 'True' entry in the mask for the classification token
        mask = torch.cat([torch.ones(B, 1).bool().to(mask.device), mask], dim=1)

        #****** Make sure that the cls token is passed in the first augment, as the 0th element ******
        #modeling the relations between parts using sparse attention to ignore padding B x Pmax x F -> B x F
        relations = self.rel(encoded_parts, mask)

        #classification B x F -> B x K
        scores = self.cls(relations)


        return scores
    
    #FORWARD FOR USAGE IN THE CLASSIFICATION LOOP
    def forward_classification(self, x, pid, class_id, centroids, max_num_parts = None, normalize=False):

        #x: torch.tensor(N, 3) point clouds
        #pid: torch.tensor(N) part ids corresponding to a maximum of M parts
        #class_id: torch.tensor(M) indicating which parts belong to which classes
        #centroids: torch.tensor(M, 3) the centroids of each part
        #The parts are expected to be normalized, centered around the origin

        if max_num_parts is None:
            max_num_parts = pid.max() + 1

        #encoding the input parts N x 3 -> M x F
        encoded_parts = self.encoder(x, pid, max_num_parts)

        #optional normalization
        if normalize:
            encoded_parts = torch.nn.functional.normalize(encoded_parts, p=2, dim=-1)

        #concatenating the centroids as positional embeddings -> M x (F + 3)
        encoded_parts = torch.cat([encoded_parts, centroids], dim=-1)

        #padding into B x Pmax x (F + 3)
        encoded_parts, mask = to_dense_batch(encoded_parts, class_id.to(torch.int64))
        B = encoded_parts.shape[0]

        #concatenating with the cls token B x Pmax x (F + 3)
        cls_token = self.cls_token.repeat(B, 1, 1)
        encoded_parts = torch.cat([cls_token, encoded_parts], dim=1)

        #adding a 'True' entry in the mask for the classification token
        mask = torch.cat([torch.ones(B, 1).bool().to(mask.device), mask], dim=1)

        #****** Make sure that the cls token is passed in the first augment, as the 0th element ******
        #modeling the relations between parts using sparse attention to ignore padding B x Pmax x F -> B x F
        relations = self.rel(encoded_parts, mask)

        #classification B x F -> B x K
        scores = self.cls(relations)


        return scores

    #FORWARD FOR USAGE IN THE RETRIEVAL TASK
    def forward_retrieval(self, query_features, query_centroids, warehouse_features, warehouse_centroids, normalize=False):
        
        #query_features: torch.tensor(M, F) encoded query shape part features
        #query_centroids: torch.tensor(M, 3) the respective centroids
        #warehouse_features: torch.tensor(K, F) encoded warehouse part features
        #warehouse_centroids: torch.tensor(K, 3) the respective centroids of each warehouse part

        M, K = query_features.shape[0], warehouse_features.shape[0]

        #Normalize the features if needed
        if normalize:
            query_features = torch.nn.functional.normalize(query_features, p=2, dim=-1)
            warehouse_features = torch.nn.functional.normalize(warehouse_features, p=2, dim=-1)

        #concatenating the centroids as positional embeddings -> M x (F + 3)
        query_features = torch.cat([query_features, query_centroids], dim=-1)
        #concatenating the warehouse centroids as positional embeddings -> K x (F + 3)
        warehouse_features = torch.cat([warehouse_features, warehouse_centroids], dim=-1)

        #creating a complete shape using all the query parts 
        #and each of the warehouse parts M x F, K x F -> K x (M + 1) x F
        query_features = query_features.unsqueeze(0).repeat(K, 1, 1)
        warehouse_features = warehouse_features.unsqueeze(1)
        final_features = torch.cat([query_features, warehouse_features], dim=1)

        #concatenating with the cls token K x (M + 2) x F
        cls_token = self.cls_token.repeat(K, 1, 1)
        final_features = torch.cat([cls_token, final_features], dim=1)

        #creating a boolean mask containing only True values 
        mask = torch.ones(K, M + 2).bool().to(final_features.device)

        #****** Make sure that the cls token is passed in the first augment, as the 0th element ******
        #modeling the relations between parts using sparse attention to ignore padding K x (M + 2) x F -> K x F
        relations = self.rel(final_features.float(), mask)

        #classification K x F -> K x Num_classes
        scores = self.cls(relations)

        return scores


if __name__ == "__main__":

    ##---------------Test PartFinderPipeline---------------
    model = PartFinderPipeline(in_channels=3, out_channels=384, num_classes=3, num_attention_blocks=3, pos_emb_dim=3, pool_method="cls_token_pool").cuda()

    x = torch.randn(1, 10, 3)
    pid = torch.Tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]]).long()
    centroids = torch.randn(1, 3, 3)

    score = model(x, pid, centroids, normalize=True)

    print(score.shape)