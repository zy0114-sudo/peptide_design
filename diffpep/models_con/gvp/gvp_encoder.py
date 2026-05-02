# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.   #

from argparse import Namespace  

import torch   
import torch.nn as nn  
import torch.nn.functional as F

from .features import GVPGraphEmbedding, GVPGraphEmbedding_V1
from .gvp_modules import GVP, GVPConvLayer, LayerNorm
from .gvp_utils import unflatten_graph   
from .features import normalize

from diffpep.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles, get_backbone_orientations
from diffpep.modules.common.layers import AngularEncoding
from diffpep.modules.protein.constants import BBHeavyAtom, AA

class GVPEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                           args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                           args.edge_hidden_dim_vector)

        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=args.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for i in range(args.num_encoder_layers)
        )

    def forward(self, coords, coord_mask, padding_mask, confidence):
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(
            coords, coord_mask, padding_mask, confidence)

        for i, layer in enumerate(self.encoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                                                     edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        return node_embeddings

class GVPEncoder_V1(nn.Module):  #

    def __init__(self, args):
        super().__init__()
        self.args = args   #args.peptide_encoder
        self.embed_graph = GVPGraphEmbedding_V1(args)

        node_hidden_dim = (args.node_hidden_dim_scalar, args.node_hidden_dim_vector)   #(128,16)
        edge_hidden_dim = (args.edge_hidden_dim_scalar, args.edge_hidden_dim_vector)   #(128,16)

        conv_activations = (F.relu, torch.sigmoid)

        ### 多加几层
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,  #（128,16）
                edge_hidden_dim,  #（128,16）
                drop_rate=args.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for i in range(args.num_encoder_layers)
        )

        infeat_dim = args.node_hidden_dim_scalar + 3 * args.node_hidden_dim_vector

        #### 直接换成输出头: res_type /
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, args.feat_dim * 2), nn.ReLU(),
            nn.Linear(args.feat_dim * 2, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim)
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, mask_residue, R, t):
        batch_size, num_res = aa.shape
        node_embeddings, edge_embeddings, edge_index = self.embed_graph(   #拆解  主要是得到节点以及边的标量和向量信息   
            aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, mask_residue, R, t)     #
        #(torch.Size([11264, 128]), torch.Size([11264, 16, 3]))

        for i, layer in enumerate(self.encoder_layers):  #GVPConvLayer2  过gvp
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_index, edge_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, batch_size)
        node_embeddings = (node_embeddings[0] * mask_residue[:, :, None],
                           node_embeddings[1] * mask_residue[:, :, None, None])

        node_embeddings = torch.cat([node_embeddings[0], torch.reshape(node_embeddings[1],node_embeddings[1].shape[:-2] + (
                                         3 * node_embeddings[1].shape[-2],))], dim=-1)   #将向量特征拉平后与标量特征拼接
        node_embeddings=self.mlp(node_embeddings)      
        return node_embeddings, edge_embeddings
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   