import torch
from torch import nn

from diffpep.models_con import ipa_pytorch as ipa_pytorch
from diffpep.data import utils as du

from diffpep.models_con.utils import get_index_embedding, get_time_embedding

from diffpep.modules.protein.constants import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from diffpep.modules.common.layers import AngularEncoding

import math

class GAEncoder(nn.Module):  
    def __init__(self, ipa_conf):
        super().__init__()
        self._ipa_conf = ipa_conf

        # angles
        self.angle_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),  #128,128
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),  #128,128
            nn.Linear(self._ipa_conf.c_s, 5)   #128,5
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # for condition on current seq
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(), #128,128
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),#128,128
            nn.Linear(self._ipa_conf.c_s, 20)  #128,20
            # nn.Linear(self._ipa_conf.c_s, 22)
        )

        # mixer  将节点特征和时间步嵌入 混合起来
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(2 * self._ipa_conf.c_s, self._ipa_conf.c_s), #128,128
            nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),  #128,128
        )

        self.feat_dim = self._ipa_conf.c_s  #128

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):  #6
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)  #128
            tfmr_in = self._ipa_conf.c_s  #128
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,#128
                nhead=self._ipa_conf.seq_tfmr_num_heads,  #4
                dim_feedforward=tfmr_in,#128
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            # 序列特征
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)  #2
            # 预测头
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")   #（128,128）
            # 节点更新和刚体更新
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)  #128
            # Backbone update
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)  #128

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z  #64
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,  #128
                    edge_embed_in=edge_in,  #64
                    edge_embed_out=self._ipa_conf.c_z,  #64
                )

    def embed_t(self, timesteps, mask):  #将扩散步骤t编程一个向量，然后复制到每个残基位置
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
        node_embed = self.res_feat_mixer(torch.cat([node_embed, self.embed_t(t, node_mask)], dim=-1))  #时间编码
        node_embed = node_embed * node_mask[..., None]

        # 创建刚体表示
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        for b in range(self._ipa_conf.num_blocks):  #6个块
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
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # 预测的结果
        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs1_prob = self.seq_net(node_embed)
        pred_angles1 = self.angle_net(node_embed)
        pred_angles1 = pred_angles1 % (2 * math.pi)  # inductive bias to bound between (0,2pi)

        return pred_rotmats1, pred_trans1, pred_angles1, pred_seqs1_prob










