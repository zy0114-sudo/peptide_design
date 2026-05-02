import torch
import torch.nn as nn  
import torch.nn.functional as F      

from diffpep.models_con.gvp.gvp_encoder import GVPEncoder_V1  #

class NodeEmbedder(nn.Module):       

    def __init__(self, args):
        super().__init__()
        self.args = args    #
        self.pocket_encoder = GVPEncoder_V1(args.pocket_encoder)
        self.peptide_encoder = GVPEncoder_V1(args.peptide_encoder)

    def forward(self, data, R, t):    #R为旋转   t为平移

        B, L = data["aa"].shape
        aa = data['aa']
        res_nb = data['chain_nb']
        chain_nb = data['chain_nb']
        pos_atoms = data['pos_heavyatom']
        mask_atoms = data['mask_heavyatom']
        angles = data['torsion_angle']
        res_mask_1 = torch.logical_and(data['res_mask'], ~data['generate_mask'])  #受体
        res_mask_2 = torch.logical_and(data['res_mask'], data['generate_mask'])   #配体

        node_embeddings_1, edge_embeddings_1 = self.pocket_encoder(aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, res_mask_1, R, t)
        node_embeddings_2, edge_embeddings_2 = self.peptide_encoder(aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, res_mask_2, R, t)

        node_embeddings = torch.where(res_mask_1[:, :, None].expand(B, L, node_embeddings_1[0].shape[-1]), 
                                      node_embeddings_1, node_embeddings_2)   #把整条序列上每个 residue 的 embedding 选出来：受体位置用 pocket embedding，配体位置用 peptide embedding
        return node_embeddings