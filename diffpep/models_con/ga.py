import torch
from torch import nn

from diffpep.models_con import ipa_pytorch as ipa_pytorch
from data import utils as du

from diffpep.models_con.utils import get_index_embedding, get_time_embedding

from diffpep.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from diffpep.modules.common.layers import AngularEncoding

import math

class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            # 序列特征
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            # 预测头
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            # 节点更新和刚体更新
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            # Backbone update
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def embed_t(self, timesteps, mask):
        """
        用于时间步嵌入
        """
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, node_embed, edge_embed, generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        # B,1,L * B,L,1 -> B,L,L
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        # 节点特征编码
        node_embed = self.res_feat_mixer(torch.cat([node_embed,
                                                    self.current_seq_embedder(seqs_t),
                                                    self.embed_t(t,node_mask),
                                                    self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
        node_embed = node_embed * node_mask[..., None]

        # 创建刚体表示
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        #
        for b in range(self._ipa_conf.num_blocks):
            # 编码
            ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            # 残差连接 + LayerNorm
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            # transformer
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).bool())
            # 残差连接 + feed_forward
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            # 刚体更新
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])
            # 边的更新
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # 预测的结果
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob

class GAEncoder(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for contact condition 
        self.current_contact_embedder = nn.Embedding(2, self._ipa_conf.c_z)
        self.contact_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),    
            nn.Linear(self._ipa_conf.c_z, 2), 
        )
        
        # mixer
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(4 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        
        self.edge_feat_mixer = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_z, self._ipa_conf.c_z),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # if b < self._ipa_conf.num_blocks-1:
            # edge update on the last block.
            edge_in = self._ipa_conf.c_z
            self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self._ipa_conf.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self._ipa_conf.c_z,
            )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, contact_t, node_embed, edge_embed, 
                generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        # contact embedding and information fusion 
        contact_embed = self.current_contact_embedder(contact_t.long())  # (B,L,L,c_s)
        contact_node_feat = torch.bmm(contact_t.float(), node_embed)  # (B,L,c_s)
        # node_embed = node_embed + contact_node_feat
        # edge_embed = edge_embed + contact_embed
        
        # 节点信息混合
        node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    contact_node_feat,
                                                    self.current_seq_embedder(seqs_t), 
                                                    self.embed_t(t,node_mask), 
                                                    self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
        node_embed = node_embed * node_mask[..., None]
        
        # 边信息混合
        edge_embed = self.edge_feat_mixer(torch.cat([edge_embed, contact_embed],dim=-1)) # 
        edge_embed = edge_embed * edge_mask[..., None] #
        
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)
        pred_contact_1 = self.contact_net(edge_embed).squeeze(-1)
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_contact_1

class GaussianDecay(nn.Module):
    def __init__(self, sigma=2.0, cutoff=15.0):
        super().__init__()
        self.sigma = sigma
        self.cutoff = cutoff
    
    def forward(self, distances):
        # f(d) = exp(-d² / (2 * sigma²))
        weights = torch.exp(-0.5 * (distances / self.sigma) ** 2)
        weights = torch.where(distances < self.cutoff, weights, torch.zeros_like(weights))
        return weights


class GAEncoderDist(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for contact condition 
        self.dist_decay = GaussianDecay(sigma=2.0, cutoff=15.0)
        self.current_dist_embedder = nn.Embedding(64, self._ipa_conf.c_z)
        self.dist_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),    
            nn.Linear(self._ipa_conf.c_z, 64)
        )
        
        # mixer
        self.res_feat_mixer = nn.Sequential(
            # nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.Linear(4 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        
        self.edge_feat_mixer = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_z, self._ipa_conf.c_z),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # if b < self._ipa_conf.num_blocks-1:
            # edge update on the last block.
            edge_in = self._ipa_conf.c_z
            self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self._ipa_conf.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self._ipa_conf.c_z,
            )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, dist_t, node_embed, edge_embed, 
                generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        # contact embedding and information fusion 
        contact_embed = self.current_dist_embedder(dist_t.long())  # (B,L,L,c_s)
        contact_node_feat = torch.bmm(self.dist_decay(dist_t.float()), node_embed)  # (B,L,c_s)
        # node_embed = node_embed + contact_node_feat
        # edge_embed = edge_embed + contact_embed
        
        # 节点信息混合
        node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    contact_node_feat, # 
                                                    self.current_seq_embedder(seqs_t), 
                                                    self.embed_t(t,node_mask), 
                                                    self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
        node_embed = node_embed * node_mask[..., None]
        
        # 边信息混合
        edge_embed = self.edge_feat_mixer(torch.cat([edge_embed, contact_embed],dim=-1)) # 
        edge_embed = edge_embed * edge_mask[..., None] # 
        
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)
        pred_dist_1 = self.dist_net(edge_embed).squeeze(-1)
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_dist_1   
    
class GAEncoderDist_Raw(nn.Module):
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf 

        # angles
        self.angles_embedder = AngularEncoding(num_funcs=12) # 25*5=120, for competitive embedding size
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 5)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, 20)
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for contact condition 
        self.dist_decay = GaussianDecay(sigma=2.0, cutoff=15.0)
        self.current_dist_embedder = nn.Embedding(64, self._ipa_conf.c_z)
        self.dist_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),nn.ReLU(),    
            nn.Linear(self._ipa_conf.c_z, 64)
        )
        
        # mixer
        self.res_feat_mixer = nn.Sequential(
            # nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.Linear(3 * self._ipa_conf.c_s + self.angles_embedder.get_out_dim(in_dim=5), self._ipa_conf.c_s),
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        self.feat_dim = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # if b < self._ipa_conf.num_blocks-1:
            # edge update on the last block.
            edge_in = self._ipa_conf.c_z
            self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self._ipa_conf.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self._ipa_conf.c_z,
            )
    
    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.feat_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def forward(self, t, rotmats_t, trans_t, angles_t, seqs_t, dist_t, node_embed, edge_embed, 
                generate_mask, res_mask):
        num_batch, num_res = seqs_t.shape

        # incorperate current seq and timesteps
        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        
        # contact embedding and information fusion 
        contact_embed = self.current_dist_embedder(dist_t.long())  # (B,L,L,c_s)
        contact_node_feat = torch.bmm(self.dist_decay(dist_t.float()), node_embed)  # (B,L,c_s)
        node_embed = node_embed + contact_node_feat
        edge_embed = edge_embed + contact_embed
        
        # 节点信息混合
        node_embed = self.res_feat_mixer(torch.cat([node_embed, 
                                                    self.current_seq_embedder(seqs_t), 
                                                    self.embed_t(t,node_mask), 
                                                    self.angles_embedder(angles_t).reshape(num_batch,num_res,-1)],dim=-1))
        node_embed = node_embed * node_mask[..., None]
        
        # # 边信息混合
        edge_embed = edge_embed * edge_mask[..., None] # 
        
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, node_mask[..., None])

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2*math.pi) # inductive bias to bound between (0,2pi)
        pred_dist_1 = self.dist_net(edge_embed).squeeze(-1)
        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob, pred_dist_1   
  
