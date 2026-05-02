# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions of this file were adapted from the open source code for the following
# two papers:
#
#   Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative
#   models for graph-based protein design. Advances in Neural Information
#   Processing Systems, 32.
#
#   Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L., & Dror, R. (2020).
#   Learning from Protein Structure with Geometric Vector Perceptrons. In
#   International Conference on Learning Representations.
#
# MIT License
#
# Copyright (c) 2020 Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael Townshend, Ron Dror
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ================================================================
# The below license applies to the portions of the code (parts of
# src/datasets.py and src/models.py) adapted from Ingraham, et al.
# ================================================================
#
# MIT License
#
# Copyright (c) 2019 John Ingraham, Vikas Garg, Regina Barzilay, Tommi Jaakkola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F

from .gvp_utils import flatten_graph  #
from .gvp_modules import GVP, LayerNorm
from .util import normalize, norm, nan_to_num, rbf

class GVPInputFeaturizer(nn.Module):

    @staticmethod
    def get_node_features(coords, coord_mask, with_coord_mask=True):
        # scalar features
        node_scalar_features = GVPInputFeaturizer._dihedrals(coords)
        if with_coord_mask:
            node_scalar_features = torch.cat([
                node_scalar_features,
                coord_mask.float().unsqueeze(-1)
            ], dim=-1)
            # vector features
        X_ca = coords[:, :, 1]
        orientations = GVPInputFeaturizer._orientations(X_ca)
        sidechains = GVPInputFeaturizer._sidechains(coords)
        node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        return node_scalar_features, node_vector_features

    @staticmethod
    def _orientations(X):
        forward = normalize(X[:, 1:] - X[:, :-1])
        backward = normalize(X[:, :-1] - X[:, 1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _sidechains(X):
        n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        c, n = normalize(c - origin), normalize(n - origin)
        bisector = normalize(c + n)
        perp = normalize(torch.cross(c, n, dim=-1))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    @staticmethod
    def _dihedrals(X, eps=1e-7):
        X = torch.flatten(X[:, :, :3], 1, 2)
        bsz = X.shape[0]
        dX = X[:, 1:] - X[:, :-1]
        U = normalize(dX, dim=-1)
        u_2 = U[:, :-2]
        u_1 = U[:, 1:-1]
        u_0 = U[:, 2:]

        # Backbone normals
        n_2 = normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [bsz, -1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], -1)
        return D_features

    @staticmethod
    def _positional_embeddings(edge_index,
                               num_embeddings=None,
                               num_positional_embeddings=16,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                         device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _dist(X, coord_mask, padding_mask, top_k_neighbors, eps=1e-8):
        """ Pairwise euclidean distances """
        bsz, maxlen = X.size(0), X.size(1)
        coord_mask_2D = torch.unsqueeze(coord_mask, 1) * torch.unsqueeze(coord_mask, 2)
        residue_mask = ~padding_mask
        residue_mask_2D = torch.unsqueeze(residue_mask, 1) * torch.unsqueeze(residue_mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = coord_mask_2D * norm(dX, dim=-1)

        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D) + (~coord_mask_2D) * (1e8 + Dseq * 1e6) + (
            ~residue_mask_2D) * (1e10)

        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            E_idx = seqpos.repeat(
                *D_neighbors.shape[:-1], 1)
        else:
            # Identify k nearest neighbors (including self)
            k = min(top_k_neighbors, X.size(1))
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)

        coord_mask_neighbors = (D_neighbors < 5e7)
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, coord_mask_neighbors, residue_mask_neighbors


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class DihedralFeatures(nn.Module):
    def __init__(self, node_embed_dim):
        """ Embed dihedral angle features. """
        super(DihedralFeatures, self).__init__()
        # 3 dihedral angles; sin and cos of each angle
        node_in = 6
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in, node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, X):
        """ Featurize coordinates as an attributed graph """
        V = self._dihedrals(X)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X, eps=1e-7, return_angles=False):
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)

        if return_angles:
            return phi, psi, omega

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features


class GVPGraphEmbedding(GVPInputFeaturizer):

    def __init__(self, args):
        super().__init__()
        self.top_k_neighbors = args.top_k_neighbors
        self.num_positional_embeddings = 16
        self.remove_edges_without_coords = True
        node_input_dim = (7, 3)
        edge_input_dim = (34, 1)
        node_hidden_dim = (args.node_hidden_dim_scalar,
                           args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                           args.edge_hidden_dim_vector)
        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4)
        )
        self.embed_confidence = nn.Linear(16, args.node_hidden_dim_scalar)

    def forward(self, coords, coord_mask, padding_mask, confidence):
        with torch.no_grad():
            node_features = self.get_node_features(coords, coord_mask)
            edge_features, edge_index = self.get_edge_features(
                coords, coord_mask, padding_mask)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)
        edge_embeddings = self.embed_edge(edge_features)

        ### 这里我们不需要
        rbf_rep = rbf(confidence, 0., 1.)
        node_embeddings = (
            node_embeddings_scalar + self.embed_confidence(rbf_rep),
            node_embeddings_vector
        )

        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    def get_edge_features(self, coords, coord_mask, padding_mask):
        X_ca = coords[:, :, 1]
        # Get distances to the top k neighbors
        E_dist, E_idx, E_coord_mask, E_residue_mask = GVPInputFeaturizer._dist(
            X_ca, coord_mask, padding_mask, self.top_k_neighbors)
        # Flatten the graph to be batch size 1 for torch_geometric package
        dest = E_idx
        B, L, k = E_idx.shape[:3]
        src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        # After flattening, [B, E]
        E_dist = E_dist.flatten(1, 2)
        E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
        E_residue_mask = E_residue_mask.flatten(1, 2)
        # Calculate relative positional embeddings and distance RBF
        pos_embeddings = GVPInputFeaturizer._positional_embeddings(
            edge_index,
            num_positional_embeddings=self.num_positional_embeddings,
        )
        D_rbf = rbf(E_dist, 0., 20.)
        # Calculate relative orientation
        X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        X_dest = torch.gather(
            X_ca,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, 3])
        )
        coord_mask_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
        coord_mask_dest = torch.gather(
            coord_mask,
            1,
            edge_index[1, :, :].expand([B, L * k])
        )
        E_vectors = X_src - X_dest
        # For the ones without coordinates, substitute in the average vector
        E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1,
                                  keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
        E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~(E_coord_mask)
        # Normalize and remove nans
        edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
        edge_v = normalize(E_vectors).unsqueeze(-2)
        edge_s, edge_v = map(nan_to_num, (edge_s, edge_v))
        # Also add indications of whether the coordinates are present
        edge_s = torch.cat([
            edge_s,
            (~coord_mask_src).float().unsqueeze(-1),
            (~coord_mask_dest).float().unsqueeze(-1),
        ], dim=-1)
        edge_index[:, ~E_residue_mask] = -1
        if self.remove_edges_without_coords:
            edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)


from diffpep.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles, get_backbone_orientations, dihedral_from_four_points
from diffpep.modules.common.layers import AngularEncoding
from diffpep.modules.protein.constants import BBHeavyAtom, AA
from diffpep.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals

class GVPGraphEmbedding_V1(nn.Module):#

    def __init__(
            self, args):
        super().__init__()
        self.args = args
        self.max_num_atoms = args.max_num_atoms   ##15
        self.max_aa_types = args.max_aa_types   #22
        self.max_relpos = args.max_relpos    #16
        self.top_k_neighbors = args.top_k_neighbors  #8
        self.feat_dim = args.feat_dim
        self.node_hidden_dim_scalar = args.node_hidden_dim_scalar  #128
        self.node_hidden_dim_scalar = args.node_hidden_dim_vector  #16
        self.edge_hidden_dim_scalar = args.edge_hidden_dim_scalar  #128
        self.edge_hidden_dim_vector = args.edge_hidden_dim_vector  #16
        self.rbf_num_bins = args.rbf_num_bins
        self.use_atom_presence_flag = args.use_atom_presence_flag
        self.normalize_vectors = args.normalize_vectors
        self.add_distance_rbf = args.add_distance_rbf

        # 1. 氨基酸类型嵌入（标量通道部分）
        self.aatype_embed_dim = args.feat_dim  # 可与 feat_dim 不同，自行调参 128
        self.aa_pair_embed_dim = args.feat_dim #128
        self.relpos_embed_dim = args.feat_dim #128
        self.distance_embed_dim = args.feat_dim

        # 5.定义模型
        ### Node
        self.aatype_embed = nn.Embedding(self.max_aa_types, args.feat_dim)   #（22,128）
        self.aachain_embed = nn.Embedding(2, args.feat_dim)
        self.dihed_embed = AngularEncoding()
        self.angle_embed = AngularEncoding(12)
        ### Edge
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, args.feat_dim)
        self.relpos_embed = nn.Embedding(2 * args.max_relpos + 1, args.feat_dim)
        self.aapair_to_distcoef = nn.Embedding(args.max_aa_types * args.max_aa_types, args.max_num_atoms * args.max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.edge_distance_embed =  nn.Sequential(
            nn.Linear(args.max_num_atoms*args.max_num_atoms, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim), nn.ReLU(),
        )
        self.dihedral_embed = AngularEncoding()

        node_scalar_in_dim = self.aatype_embed_dim  + self.dihed_embed.get_out_dim(3) + self.angle_embed.get_out_dim(5)
        if self.add_distance_rbf:
            node_scalar_in_dim += self.max_num_atoms * self.rbf_num_bins  #15*16
        if self.use_atom_presence_flag:
            node_scalar_in_dim += self.max_num_atoms   #15
        node_vector_in_dim = self.max_num_atoms + 2  # +2 for backbone orientation vectors  15——2

        edge_scaler_in_dim = self.aa_pair_embed_dim + self.relpos_embed_dim + self.distance_embed_dim + self.dihedral_embed.get_out_dim(2)
        edge_vector_in_dim = 1  # relative orientation vector

        ### GVPEmbedding
        node_input_dim = (node_scalar_in_dim, node_vector_in_dim)  #？，15
        node_hidden_dim = (args.node_hidden_dim_scalar, args.node_hidden_dim_vector)  #128,16
        edge_input_dim = (edge_scaler_in_dim, edge_vector_in_dim)
        edge_hidden_dim = (args.edge_hidden_dim_scalar, args.edge_hidden_dim_vector)   #128,16

        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4),
        )

    def get_node_features(self, aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, mask_residue, R, t):
        """
        提取节点的标量和向量特征。
        
        Args:
            aa: 氨基酸类型 (N, L)
            res_nb: 残基编号 (N, L)
            chain_nb: 链编号 (N, L)
            pos_atoms: 原子坐标 (N, L, A, 3)
            angles: 侧链角度 (N, L, *)
            mask_atoms: 原子掩码 (N, L, A)
            mask_residue: 残基掩码 (N, L)
            R: 旋转矩阵 (N, L, 3, 3)
            t: 平移向量 (N, L, 3)
        
        Returns:
            node_scalar: 标量特征 (N, L, scalar_in_dim)
            node_vector: 向量特征 (N, L, vector_in_dim, 3)
        """
        N, L = aa.size()  # 批次大小N，序列长度L
        # 截取最大原子数量
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # === 标量特征 ===
        # 氨基酸类型嵌入
        aa_feat = self.aatype_embed(aa)  # (N, L, 128)

        # 原子到CA的距离RBF编码
        if self.add_distance_rbf:
            dist_raw = torch.linalg.norm(pos_atoms - pos_atoms[:, :, BBHeavyAtom.CA].unsqueeze(2), dim=-1)  # 各原子到CA的欧氏距离
            dist_raw = torch.where(mask_atoms, dist_raw, torch.zeros_like(dist_raw))  # 掩码位置置零
            dist_rbf = rbf(dist_raw, v_min=0.0, v_max=15.0, n_bins=self.rbf_num_bins)  # RBF编码 (N,L,maxA,16)
            dist_rbf = dist_rbf.reshape(N, L, self.max_num_atoms * self.rbf_num_bins)  # 展平 (N,L,maxA*16)
        else:
            dist_rbf = None

        # 原子存在标志
        if self.use_atom_presence_flag:
            atom_presence = mask_atoms.float()  # (N, L, maxA)
        else:
            atom_presence = None

        # 骨架二面角（phi, psi, omega）
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # 嵌入并应用掩码
        dihed_feat = dihed_feat.reshape(N, L, -1)  # 展平 (N, L, dihed_dim)
        dihed_feat = dihed_feat * mask_residue[:, :, None]

        # 侧链角度特征
        side_chain_feat = self.angle_embed(angles).reshape(N, L, -1) * mask_residue[:, :, None]

        # === 向量特征 ===
        # 骨架方向特征（N-C方向和C-N'方向）
        bb_orient, mask_bb_orient = get_backbone_orientations(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        orient_feat = bb_orient * mask_residue[:, :, None, None]  # (N, L, 2, 3)

        # 局部坐标系下的原子位置向量
        crd = global_to_local(R, t, pos_atoms)  # 转换到局部坐标系 (N, L, A, 3)
        pos_local = crd.reshape(N, L, self.max_num_atoms, 3)
        atom_vector = pos_local
        if self.normalize_vectors:
            # 归一化保留方向，零向量保持不变
            norms = torch.linalg.norm(atom_vector, dim=-1, keepdim=True) + 1e-8
            atom_vector = torch.where(norms > 1e-6, atom_vector / norms, atom_vector)
        atom_vector = atom_vector * mask_residue[:, :, None, None]  # (N, L, maxA, 3)

        # === 特征拼接 ===
        # 拼接所有标量特征
        scalar_parts = [aa_feat, dihed_feat, side_chain_feat]
        if dist_rbf is not None:
            scalar_parts.append(dist_rbf)
        if atom_presence is not None:
            scalar_parts.append(atom_presence)
        node_scalar = torch.cat(scalar_parts, dim=-1)  # (N, L, scalar_in_dim)
        node_scalar = node_scalar * mask_residue[:, :, None]

        # 拼接所有向量特征
        node_vector = torch.cat([orient_feat, atom_vector], dim=2)  # (N, L, vector_in_dim, 3)
        node_vector = node_vector * mask_residue[:, :, None, None]

        return node_scalar, node_vector

    def get_edge_features(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, mask_residue):

        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            trans, sc_trans: (N,L,3)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        x_ca = pos_atoms[:, :, BBHeavyAtom.CA]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # 构建边的特征
        edge_dist, dest, edge_residue_mask = GVPGraphEmbeddingAdapted._dist(x_ca, mask_residue, self.top_k_neighbors)
        B, L, k = dest.shape[:3]
        # B,L,K
        src = torch.arange(L, device=dest.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        edge_residue_mask = edge_residue_mask.flatten(1, 2) # B,E
        # B,L,3 -> B,L,k,3 -> B,L*K,3

        # 边的向量特征: 计算相对位置向量（保证平移不变性）
        x_src = x_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        x_dest = torch.gather(x_ca,1, edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, 3]))
        edge_vectors = x_src - x_dest
        edge_v = normalize(edge_vectors).unsqueeze(-2)

        # 边的标量特征:
        ### 边的标量特征:多肽对特征
        aa = torch.where(mask_residue, aa, torch.full_like(aa, fill_value=AA.UNK))
        x_src_aa = aa.unsqueeze(2).expand(-1, -1, k).flatten(1, 2) # B, L*K
        x_dest_aa = torch.gather(aa,1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
        aa_pair = x_src_aa * self.max_aa_types + x_dest_aa  # B, L*K
        feat_aapair = self.aa_pair_embed(aa_pair)

        ###  边的标量特征:相对位置编码
        x_src_chain = chain_nb.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)  # B, L*K
        x_dest_chain = torch.gather(chain_nb,1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
        same_chain = (x_src_chain == x_dest_chain)
        x_src_res_nb = res_nb.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)  # B, L*K
        x_dest_res_nb = torch.gather(res_nb,1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
        relpos = torch.clamp(
            x_src_res_nb - x_dest_res_nb,
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L*k)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, None]

        ### 边的标量特征: 距离特征

        x_src_pos_atom = pos_atoms.unsqueeze(2).expand(-1, -1, k, -1, -1).flatten(1, 2)  # B, L*K, A, 3
        x_dest_pos_atom = torch.gather(
            pos_atoms,
            1,
            edge_index[1, :, :].unsqueeze(-1).unsqueeze(-1).expand([B, L * k, self.max_num_atoms, 3])
        )  # B,L*K,A,3

        d = angstrom_to_nm(torch.linalg.norm(
            # B,L*K,A,1,3 - B,L*K,1,A,3 -> B,L*K,A,A
            x_src_pos_atom[:,:,:,None,:] - x_dest_pos_atom[:,:,None,:,:],
            dim=-1, ord=2,
        )).reshape(N, L* k, -1)  # (N, L*k, A*A)

        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L*K, A*A)
        d_gauss = torch.exp(-1 * c * d ** 2)
        
        x_src_mask_atoms = mask_atoms.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)  # B, L*K, A
        x_dest_mask_atoms = torch.gather(
            mask_atoms,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, self.max_num_atoms])
        )  # B,L*K,A
        mask_atom_pair = (x_src_mask_atoms[:, :, :, None] * x_dest_mask_atoms[:, :, None, :]).reshape(N, L* k, -1)
        feat_dist = self.edge_distance_embed(d_gauss * mask_atom_pair)

        ### 边的标量特征: 二面角特征
        dihed = GVPGraphEmbedding_V1._pairwise_dihedrals(x_src_pos_atom, x_dest_pos_atom)  # (N, L* k, 2)
        feat_dihed = self.dihedral_embed(dihed)

        ### 合并边的标量特征
        edge_s = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)

        # 移除无效的边
        edge_index[:, ~edge_residue_mask] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)

    def forward(self, aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, mask_residue, R, t):
        
        node_features = self.get_node_features(aa, res_nb, chain_nb, pos_atoms, angles, mask_atoms, mask_residue, R, t)   #node包含标量特征和向量特征
        edge_features, edge_index = self.get_edge_features(aa, res_nb, chain_nb, pos_atoms, mask_atoms, mask_residue)
        
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features)  #过GVP node_scalar, node_vector
        node_embeddings_scalar = node_embeddings_scalar * mask_residue[:,:,None]
        node_embeddings_vector = node_embeddings_vector * mask_residue[:,:,None,None]
        node_embeddings = (node_embeddings_scalar, node_embeddings_vector)
        edge_embeddings_scalar, edge_embeddings_vector = self.embed_edge(edge_features) #过GVP
        # 会在flatten_graph中进行mask
        edge_embeddings_scalar = edge_embeddings_scalar 
        edge_embeddings_vector = edge_embeddings_vector 
        edge_embeddings = (edge_embeddings_scalar, edge_embeddings_vector)
        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    @staticmethod
    def _dist(X, residue_mask, top_k_neighbors, eps=1e-8):
        """
        Pairwise euclidean distances.
        coord_mask: 坐标掩码，指示哪些位置有实际坐标（非填充），形状为 (bsz, maxlen)
        residue_mask: indicates whether the residue exists (not padding),
        """
        bsz, maxlen = X.size(0), X.size(1)
        residue_mask_2D = torch.unsqueeze(residue_mask, 1) * torch.unsqueeze(residue_mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = norm(dX, dim=-1)

        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D) +  (Dseq * 1e6) + (~residue_mask_2D) * (1e10)

        # 根据top_k_neighbors参数决定返回所有邻居还是前k个邻居
        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            # 创建邻居索引矩阵：对于每个位置，所有其他位置的索引
            E_idx = seqpos.repeat(*D_neighbors.shape[:-1], 1)
        else:
            k = min(top_k_neighbors, X.size(1))
            # torch.topk返回最小的k个值和对应的索引
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)

        # 如果距离小于5e9，表示这个邻居是真实残基（不是填充）
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, residue_mask_neighbors
    
    @staticmethod
    def _pairwise_dihedrals(x_src_pos_atom, x_dest_pos_atom):
        N, L = x_src_pos_atom.shape[:2]
        
        x_src_pos_N  = x_src_pos_atom[:, :, BBHeavyAtom.N]   # (N, L, 3)
        x_src_pos_CA = x_src_pos_atom[:, :, BBHeavyAtom.CA]
        x_src_pos_C  = x_src_pos_atom[:, :, BBHeavyAtom.C]
        
        x_dest_pos_N  = x_dest_pos_atom[:, :, BBHeavyAtom.N]   # (N, L, 3)
        x_dest_pos_CA = x_dest_pos_atom[:, :, BBHeavyAtom.CA]
        x_dest_pos_C  = x_dest_pos_atom[:, :, BBHeavyAtom.C]

        ### 检查一下
        ir_phi = dihedral_from_four_points(
            x_src_pos_C, 
            x_dest_pos_N, 
            x_dest_pos_CA, 
            x_dest_pos_C
        )
        ir_psi = dihedral_from_four_points(
            x_src_pos_N, 
            x_src_pos_CA, 
            x_src_pos_C, 
            x_dest_pos_N
        )
        ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
        return ir_dihed


class GVPGraphEmbeddingAdapted(nn.Module):

    def __init__(
            self,
            feat_dim: int,  #128
            max_num_atoms: int,
            max_aa_types: int = 22,
            max_relpos: int = 32,
            top_k_neighbors: int = 8,
            node_hidden_dim_scalar: int = 128,
            node_hidden_dim_vector: int = 16,
            edge_hidden_dim_scalar: int = 128,
            edge_hidden_dim_vector: int = 16,
            rbf_num_bins: int = 8,
            use_atom_presence_flag: bool = True,
            normalize_vectors: bool = True,
            add_distance_rbf: bool = True,
            dihedral_encoder_cls=None,  # 传入你的 AngularEncoding 类
    ):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types  #22
        self.max_relpos = max_relpos
        self.top_k_neighbors = top_k_neighbors
        self.feat_dim = feat_dim   #128
        self.node_hidden_dim_scalar = node_hidden_dim_scalar
        self.node_hidden_dim_scalar = node_hidden_dim_vector
        self.edge_hidden_dim_scalar = edge_hidden_dim_scalar
        self.edge_hidden_dim_vector = edge_hidden_dim_vector
        self.rbf_num_bins = rbf_num_bins
        self.use_atom_presence_flag = use_atom_presence_flag
        self.normalize_vectors = normalize_vectors
        self.add_distance_rbf = add_distance_rbf

        # 1. 氨基酸类型嵌入（标量通道部分）
        self.aatype_embed_dim = feat_dim  # 可与 feat_dim 不同，自行调参
        self.chain_embed_dim = feat_dim
        self.aa_pair_embed_dim = feat_dim
        self.relpos_embed_dim = feat_dim
        self.distance_embed_dim = feat_dim


        # 5.定义模型
        ### Node
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.aachain_embed = nn.Embedding(2, feat_dim) #(2,128)
        self.dihed_embed = AngularEncoding()
        ### Edge
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)  #（15*15,128）
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)
        self.edge_dist_embed =  nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )
        self.dihedral_embed = AngularEncoding()

        node_scalar_in_dim = self.aatype_embed_dim + self.chain_embed_dim + self.dihed_embed.get_out_dim(3)
        if self.add_distance_rbf:
            node_scalar_in_dim += self.max_num_atoms * self.rbf_num_bins
        if self.use_atom_presence_flag:
            node_scalar_in_dim += self.max_num_atoms
        node_vector_in_dim = self.max_num_atoms + 2  # +2 for backbone orientation vectors

        edge_scaler_in_dim = self.aa_pair_embed_dim + self.relpos_embed_dim + self.distance_embed_dim + self.dihedral_embed.get_out_dim(2)
        edge_vector_in_dim = 1  # relative orientation vector

        ### GVPEmbedding
        node_input_dim = (node_scalar_in_dim, node_vector_in_dim)
        node_hidden_dim = (node_hidden_dim_scalar, node_hidden_dim_vector)
        edge_input_dim = (edge_scaler_in_dim, edge_vector_in_dim)
        edge_hidden_dim = (edge_hidden_dim_scalar, edge_hidden_dim_vector)

        self.embed_node = nn.Sequential(
            GVP(node_input_dim, node_hidden_dim, activations=(None, None)),
            LayerNorm(node_hidden_dim, eps=1e-4)
        )
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4),
        )

    def get_node_features(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, generate_mask):

        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # 标量特征: 氨基酸类型嵌入
        aa_feat = self.aatype_embed(aa)  # (N, L, feat)

        # 标量特征:氨基酸原子坐标特征
        if self.add_distance_rbf:  #True
            # dist_raw: (N, L, maxA)
            dist_raw = torch.linalg.norm(pos_atoms - pos_atoms[:, :, BBHeavyAtom.CA].unsqueeze(2), dim=-1)
            dist_raw = torch.where(mask_atoms, dist_raw, torch.zeros_like(dist_raw))
            # RBF 编码
            dist_rbf = rbf(dist_raw, v_min=0.0, v_max=15.0, n_bins=self.rbf_num_bins)  # (N,L,maxA,n_bins)
            dist_rbf = dist_rbf.reshape(N, L, self.max_num_atoms * self.rbf_num_bins)
        else:
            dist_rbf = None

        # 标量特征: 原子存在标志（mask_atoms）作为标量特征
        if self.use_atom_presence_flag:
            atom_presence = mask_atoms.float()  # (N, L, maxA)
        else:
            atom_presence = None

        #标量特征: 属于哪一条链
        chain_id = generate_mask.long()
        chain_feat = self.aachain_embed(chain_id)
        chain_feat = chain_feat * mask_residue[:, :, None]

        # 标量特征: 骨架二面角
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)  omega,phi,psi
        dihed_feat = dihed_feat.reshape(N, L, -1)
        dihed_feat = dihed_feat * mask_residue[:, :, None]

        # 向量特征: 骨架方向特征 Backbone orientation features
        bb_orient, mask_bb_orient = get_backbone_orientations(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)  #前一个Ca-后一个Ca  以及后一个减去前一个
        orient_feat = bb_orient * mask_residue[:, :, None, None]  # (N,L,2,3) * ()

        # 向量特征：每个原子一个向量通道
        # 需要 shape: (N, L, vector_in_dim, 3)
        # Coordinate features
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA],
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)  # (N, L, A, 3)  从全局坐标构建局部坐标  先平移再旋转(先减去CA坐标再乘以旋转矩阵)
        pos_local = crd.reshape(N, L, self.max_num_atoms, 3)  # (N, L, 原子数15, 3)
        atom_vector = pos_local  # (N, L, maxA, 3)
        if self.normalize_vectors:  #True
            # 归一化（保留方向），若某向量全 0 则保持 0
            norms = torch.linalg.norm(atom_vector, dim=-1, keepdim=True) + 1e-8
            atom_vector = torch.where(norms > 1e-6, atom_vector / norms, atom_vector)
        atom_vector = atom_vector * mask_residue[:, :, None, None]  # (N,L,原子数,3)

        # 拼接所有标量部分
        scalar_parts = [aa_feat, chain_feat, dihed_feat]  #氨基酸信息+属于那一条链+二面角
        if dist_rbf is not None:
            scalar_parts.append(dist_rbf)   #距离CA的的距离RBF编码
        if atom_presence is not None:
            scalar_parts.append(atom_presence)  #原子是否存在
        node_scalar = torch.cat(scalar_parts, dim=-1)  # (N, L, scalar_in_dim)
        node_scalar = node_scalar * mask_residue[:, :, None]

        # 拼接所有的向量部分
        node_vector = torch.cat([orient_feat, atom_vector], dim=2)   #骨架方向和原子的局部特征
        node_vector = node_vector * mask_residue[:, :, None, None]

        return node_scalar, node_vector

    def get_edge_features(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):

        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            trans, sc_trans: (N,L,3)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()
        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        x_ca = pos_atoms[:, :, BBHeavyAtom.CA]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]  # (N, L, L)

        ### 边的标量特征:多肽对特征
        aa = torch.where(mask_pair, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, 1) + (N, 1, L) -> (N, L, L) -> (N, L, 0-483)  自动进行广播
        feat_aapair = self.aa_pair_embed(aa_pair)

        ###  边的标量特征:相对位置编码  序列距离远近 序列相邻（|i-j|小）：更可能有局部几何约束、短程相互作用 序列远邻（|i-j|大）：更可能是长程接触（折叠/界面）
        same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        ### 边的标量特征: 距离特征
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)  # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d ** 2)
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        ### 边的标量特征: 二面角特征
        dihed = pairwise_dihedrals(pos_atoms)   # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)

        ### 合并边的标量特征
        edge_s = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        # 将无效的边视设置为0
        edge_s = edge_s * mask_pair[:, :, :, None]

        # 边的向量特征: 计算相对位置向量
        edge_dist, dest, edge_residue_mask = GVPGraphEmbeddingAdapted._dist(x_ca, mask_residue, self.top_k_neighbors)
        B, L, k = dest.shape[:3]
        # B,L,K
        src = torch.arange(L, device=dest.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        edge_residue_mask = edge_residue_mask.flatten(1, 2)
        # B,L,3 -> B,L,k,3 -> B,L*K,3
        x_src = x_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        x_dest = torch.gather(
            x_ca,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, 3])
        )
        edge_vectors = x_src - x_dest
        edge_v = normalize(edge_vectors).unsqueeze(-2)
        # 移除无效的边
        edge_index[:, ~edge_residue_mask] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, generate_mask):
        node_features = self.get_node_features(aa, res_nb, chain_nb, pos_atoms, mask_atoms, generate_mask)
        edge_features, edge_index = self.get_edge_features(aa, res_nb, chain_nb, pos_atoms, mask_atoms)
        node_embeddings_scalar, node_embeddings_vector = self.embed_node(node_features) 
        node_embeddings = (node_embeddings_scalar, node_embeddings_vector)
        edge_embeddings = self.embed_edge(edge_features)
        node_embeddings, edge_embeddings, edge_index = flatten_graph(
            node_embeddings, edge_embeddings, edge_index)
        return node_embeddings, edge_embeddings, edge_index

    @staticmethod
    def _dist(X, residue_mask, top_k_neighbors, eps=1e-8):
        """
        Pairwise euclidean distances.
        coord_mask: 坐标掩码，指示哪些位置有实际坐标（非填充），形状为 (bsz, maxlen)
        residue_mask: indicates whether the residue exists (not padding),
        """
        bsz, maxlen = X.size(0), X.size(1)
        residue_mask_2D = torch.unsqueeze(residue_mask, 1) * torch.unsqueeze(residue_mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = norm(dX, dim=-1)

        # sorting preference: first those with coords, then among the residues that
        # exist but are masked use distance in sequence as tie breaker, and then the
        # residues that came from padding are last
        seqpos = torch.arange(maxlen, device=X.device)
        Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
        D_adjust = nan_to_num(D) +  (Dseq * 1e6) + (~residue_mask_2D) * (1e10)

        # 根据top_k_neighbors参数决定返回所有邻居还是前k个邻居
        if top_k_neighbors == -1:
            D_neighbors = D_adjust
            # 创建邻居索引矩阵：对于每个位置，所有其他位置的索引
            E_idx = seqpos.repeat(*D_neighbors.shape[:-1], 1)
        else:
            k = min(top_k_neighbors, X.size(1))
            # torch.topk返回最小的k个值和对应的索引
            D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)

        # 如果距离小于5e9，表示这个邻居是真实残基（不是填充）
        residue_mask_neighbors = (D_neighbors < 5e9)
        return D_neighbors, E_idx, residue_mask_neighbors
















