import torch
import torch.nn as nn
import torch.nn.functional as F


from diffpep.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals, dihedral_from_four_points
from diffpep.modules.common.layers import AngularEncoding
from diffpep.modules.protein.constants import BBHeavyAtom, AA

from diffpep.models_con.gvp.gvp_utils import flatten_graph
from diffpep.models_con.gvp.gvp_modules import GVP, LayerNorm
from diffpep.models_con.gvp.util import normalize, norm, nan_to_num, rbf


class EdgeEmbedder(nn.Module):   

    def __init__(self, args):
        super().__init__()
        self.max_num_atoms = args.max_num_atoms
        self.max_aa_types = args.max_aa_types
        self.max_relpos = args.max_relpos
        self.aa_pair_embed_dim = args.feat_dim
        self.relpos_embed_dim = args.feat_dim
        self.distance_embed_dim = args.feat_dim

        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, args.feat_dim)
        self.relpos_embed = nn.Embedding(2 * args.max_relpos + 1, args.feat_dim)
        self.aapair_to_distcoef = nn.Embedding(args.max_aa_types * args.max_aa_types,
                                               args.max_num_atoms * args.max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.edge_distance_embed = nn.Sequential(
            nn.Linear(args.max_num_atoms * args.max_num_atoms, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim), nn.ReLU(),
        )
        self.dihedral_embed = AngularEncoding()

        edge_scaler_in_dim = self.aa_pair_embed_dim + self.relpos_embed_dim + self.distance_embed_dim + self.dihedral_embed.get_out_dim(2)
        edge_vector_in_dim = 1  # relative orientation vector
        edge_input_dim = (edge_scaler_in_dim, edge_vector_in_dim)
        edge_hidden_dim = (args.edge_hidden_dim_scalar, args.edge_hidden_dim_vector)
        self.embed_edge = nn.Sequential(
            GVP(edge_input_dim, edge_hidden_dim, activations=(None, None)),
            LayerNorm(edge_hidden_dim, eps=1e-4),
        )

        infeat_dim = args.edge_hidden_dim_scalar + args.edge_hidden_dim_vector * 3
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, args.feat_dim * 2), nn.ReLU(),
            nn.Linear(args.feat_dim * 2, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim), nn.ReLU(),
            nn.Linear(args.feat_dim, args.feat_dim)
        )

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
        edge_dist, dest, edge_residue_mask = EdgeEmbedder._dist(x_ca, mask_residue, -1)
        B, L, k = dest.shape[:3]
        # B,L,K
        src = torch.arange(L, device=dest.device).view([1, L, 1]).expand(B, L, k)
        # After flattening, [2, B, E]
        edge_index = torch.stack([src, dest], dim=0).flatten(2, 3)
        edge_residue_mask = edge_residue_mask.flatten(1, 2)  # B,E
        # B,L,3 -> B,L,k,3 -> B,L*K,3

        # 边的向量特征: 计算相对位置向量
        x_src = x_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
        x_dest = torch.gather(x_ca, 1, edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, 3]))
        edge_vectors = x_src - x_dest
        edge_v = normalize(edge_vectors).unsqueeze(-2)

        # 边的标量特征:
        ### 边的标量特征:多肽对特征
        aa = torch.where(mask_residue, aa, torch.full_like(aa, fill_value=AA.UNK))
        x_src_aa = aa.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)  # B, L*K
        x_dest_aa = torch.gather(aa, 1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
        aa_pair = x_src_aa * self.max_aa_types + x_dest_aa  # B, L*K
        feat_aapair = self.aa_pair_embed(aa_pair)

        ###  边的标量特征:相对位置编码
        x_src_chain = chain_nb.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)  # B, L*K
        x_dest_chain = torch.gather(chain_nb, 1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
        same_chain = (x_src_chain == x_dest_chain)
        x_src_res_nb = res_nb.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)  # B, L*K
        x_dest_res_nb = torch.gather(res_nb, 1, edge_index[1, :, :].expand([B, L * k]))  # B, L*K
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
            x_src_pos_atom[:, :, :, None, :] - x_dest_pos_atom[:, :, None, :, :],
            dim=-1, ord=2,
        )).reshape(N, L * k, -1)  # (N, L*k, A*A)

        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L*K, A*A)
        d_gauss = torch.exp(-1 * c * d ** 2)

        x_src_mask_atoms = mask_atoms.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)  # B, L*K, A
        x_dest_mask_atoms = torch.gather(
            mask_atoms,
            1,
            edge_index[1, :, :].unsqueeze(-1).expand([B, L * k, self.max_num_atoms])
        )  # B,L*K,A
        mask_atom_pair = (x_src_mask_atoms[:, :, :, None] * x_dest_mask_atoms[:, :, None, :]).reshape(N, L * k, -1)
        feat_dist = self.edge_distance_embed(d_gauss * mask_atom_pair)

        ### 边的标量特征: 二面角特征
        dihed = EdgeEmbedder._pairwise_dihedrals(x_src_pos_atom, x_dest_pos_atom)  # (N, L* k, 2)
        feat_dihed = self.dihedral_embed(dihed)

        ### 合并边的标量特征
        edge_s = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)

        # 移除无效的边
        edge_index[:, ~edge_residue_mask] = -1
        return (edge_s, edge_v), edge_index.transpose(0, 1)

    def forward(self, data, rotmats_t, trans_t_c):
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
        B, L = data["aa"].shape
        aa = data['aa']
        res_nb = data['chain_nb']
        chain_nb = data['chain_nb']
        pos_atoms = data['pos_heavyatom']
        mask_atoms = data['mask_heavyatom']
        mask_residue = data['res_mask']

        edge_features, edge_index = self.get_edge_features(aa, res_nb, chain_nb, pos_atoms, mask_atoms, mask_residue)
        edge_embeddings_scalar, edge_embeddings_vector = self.embed_edge(edge_features)
        # print(edge_embeddings_scalar.shape, edge_embeddings_vector.shape)
        edge_embeddings_scalar = edge_embeddings_scalar.reshape(B,L,L,-1)
        edge_embeddings_vector = edge_embeddings_vector.reshape((B,L,L,)+edge_embeddings_vector.shape[-2:]).reshape(B,L,L,-1)
        edge_embeddings = torch.cat([edge_embeddings_scalar, edge_embeddings_vector], dim=-1)
        edge_embeddings = self.mlp(edge_embeddings)
        return edge_embeddings

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
        D_adjust = nan_to_num(D) + (Dseq * 1e6) + (~residue_mask_2D) * (1e10)

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

        x_src_pos_N = x_src_pos_atom[:, :, BBHeavyAtom.N]  # (N, L, 3)
        x_src_pos_CA = x_src_pos_atom[:, :, BBHeavyAtom.CA]
        x_src_pos_C = x_src_pos_atom[:, :, BBHeavyAtom.C]

        x_dest_pos_N = x_dest_pos_atom[:, :, BBHeavyAtom.N]  # (N, L, 3)
        x_dest_pos_CA = x_dest_pos_atom[:, :, BBHeavyAtom.CA]
        x_dest_pos_C = x_dest_pos_atom[:, :, BBHeavyAtom.C]

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

