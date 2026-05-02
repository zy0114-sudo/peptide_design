import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from diffpep.modules.common.geometry import *
from diffpep.modules.protein.constants import *

import MDAnalysis as mda  #
from MDAnalysis.analysis import distances, contacts
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis


RESIDUE_DISTANCE_THRESHOLD = 8.0  # Å
ATOM_DISTANCE_THRESHOLD = 4.0     # Å

# 定义相互作用类型及其距离和角度阈值
INTERACTION_CUTOFFS = {
    'hydrogen_bond': {'distance': 3.5, 'angle': 120.0},  # 距离(Å), 角度(°)
    'salt_bridge': 4.0,                                  # 距离(Å)
    'hydrophobic': 5.0,                                  # 距离(Å)
    'pi_stacking': {'distance': 5.5, 'angle': 30.0},     # 距离(Å), 平面夹角(°)
    'cation_pi': 6.0,                                    # 距离(Å)
    'vdw': {'min': 3.0, 'max': 4.5}                      # 范德华距离范围(Å)
}

# 氨基酸分类
POLAR_RESIDUES = ['ARG', 'LYS', 'HIS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN']
CHARGED_RESIDUES = ['ARG', 'LYS', 'ASP', 'GLU']
HYDROPHOBIC_RESIDUES = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']
AROMATIC_RESIDUES = ['PHE', 'TYR', 'TRP', 'HIS']

# 氢键分析

def analyze_hbond_interaction(protein, binder, output_dir):
    
    results = []
    
    hbonds = HydrogenBondAnalysis(complex_universe, 
                                 donors_sel="protein",
                                 acceptors_sel="resname BINDER")
    hbonds.run()
    
    for frame, donor_idx, acceptor_idx, dist, angle in hbonds.results.hbonds:
        donor = complex_universe.atoms[donor_idx]
        acceptor = complex_universe.atoms[acceptor_idx]
        
        # 获取残基信息
        donor_res = donor.residue
        acceptor_res = acceptor.residue
        
        # 确保一个来自蛋白，一个来自binder
        if donor_res.segment_id == binder_residues[0].segment_id:
            protein_res = acceptor_res
            binder_res = donor_res
        else:
            protein_res = donor_res
            binder_res = acceptor_res
            
        # 检查是否在有效残基对中
        if (protein_res.segment_id != binder_res.segment_id and 
            protein_res.resname in protein.residues.resnames and
            binder_res.resname in binder.residues.resnames):
            
            results.append({
                'protein_res': f"{protein_res.resname}{protein_res.resid}",
                'binder_res': f"{binder_res.resname}{binder_res.resid}",
                'donor_atom': donor.name,
                'acceptor_atom': acceptor.name,
                'distance': dist,
                'angle': angle
            })
    
    return results


def vander_walls_interaction(protein, binder, output_dir):
    
    protein_residues = protein.residues
    binder_residues = binder.residues
    results = []
    
    for p_res in protein_residues:
        for b_res in binder_residues:
            p_atoms = p_res.atoms.select_atoms("not name H*")
            b_atoms = b_res.atoms.select_atoms("not name H*")
            min_dist = distances.min_dist(p_atoms.positions, b_atoms.positions)[0]
            
            if (INTERACTION_CUTOFFS['vdw']['min'] <= min_dist <= 
                INTERACTION_CUTOFFS['vdw']['max']):
                results.append({
                    'protein_res': f"{p_res.resname}{p_res.resid}",
                    'binder_res': f"{b_res.resname}{b_res.resid}",
                    'distance': min_dist
                })
    return results

def compute_distance_matrix(pos_a, pos_b):
    """
    pos_a: B, L, 3
    """
    if pos_a.dim() == 3 and pos_b.dim() == 3:
        diff = pos_a[:, None, :, :] - pos_b[:, :, None, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        return dist_matrix   #B,L,L,3
    elif pos_a.dim() == 2 and pos_b.dim() == 2:
        diff = pos_a[:, None, :] - pos_b[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        return dist_matrix

def atom_distance_map(receptor, ligand):
    receptor_heavyatom_pos = receptor['pos_heavyatom'].reshape(-1, 3)
    ligand_heavyatom_pos = ligand['pos_heavyatom'].reshape(-1, 3)
    distance_map = compute_distance_matrix(receptor_heavyatom_pos, ligand_heavyatom_pos)
    return distance_map

def atom_distance_map_complex(data, cutoff=ATOM_DISTANCE_THRESHOLD):
    pos_receptor = data['pos_heavyatom'][~data['generate_mask']]
    pos_ligand = data['pos_heavyatom'][data['generate_mask']]
    receptor_heavyatom_pos = pos_receptor.reshape(-1, 3)
    ligand_heavyatom_pos = pos_ligand.reshape(-1, 3)
    distance_map = compute_distance_matrix(receptor_heavyatom_pos, ligand_heavyatom_pos)
    return distance_map

def atom_contact_map(receptor, ligand, cutoff=ATOM_DISTANCE_THRESHOLD):
    distance_map = atom_distance_map(receptor, ligand)
    return (distance_map < cutoff).float()

def distance_map(receptor, ligand, mode='CA'):
    receptor_heavyatom_pos = receptor['pos_heavyatom']
    ligand_heavyatom_pos = ligand['pos_heavyatom']
    if mode == 'CA':
        receptor_pos = receptor_heavyatom_pos[:, BBHeavyAtom.CA, :]
        ligand_pos = ligand_heavyatom_pos[:, BBHeavyAtom.CA, :]
        return compute_distance_matrix(receptor_pos, ligand_pos)         
    elif mode == 'center':
        receptor_mask = receptor['mask_heavyatom']
        ligand_mask = ligand['mask_heavyatom']
        receptor_pos = torch.sum(receptor_heavyatom_pos * receptor_mask.unsqueeze(-1), dim=1) / (torch.sum(receptor_mask, dim=1, keepdim=True) + 1e-8)
        ligand_pos = torch.sum(ligand_heavyatom_pos * ligand_mask.unsqueeze(-1), dim=1) / (torch.sum(ligand_mask, dim=1, keepdim=True) + 1e-8)
        return compute_distance_matrix(receptor_pos, ligand_pos)
    else:
        raise ValueError(f'Unknown mode {mode}')

def distance_map_complex(data, mode):
    if mode == 'CA':
        pos_receptor = data['pos_heavyatom'][~data['generate_mask'], BBHeavyAtom.CA, :]
        pos_peptide = data['pos_heavyatom'][data['generate_mask'], BBHeavyAtom.CA, :]
        return compute_distance_matrix(pos_receptor, pos_peptide)
    elif mode == 'center':
        mask_receptor = data['mask_heavyatom'][~data['generate_mask']]
        mask_peptide = data['mask_heavyatom'][data['generate_mask']]
        pos_receptor = data['pos_heavyatom'][~data['generate_mask']]
        pos_peptide = data['pos_heavyatom'][data['generate_mask']]
        center_receptor = torch.sum(pos_receptor * mask_receptor.unsqueeze(-1), dim=1) / (torch.sum(mask_receptor, dim=1, keepdim=True) + 1e-8)
        center_peptide = torch.sum(pos_peptide * mask_peptide.unsqueeze(-1), dim=1) / (torch.sum(mask_peptide, dim=1, keepdim=True) + 1e-8)
        return compute_distance_matrix(center_receptor, center_peptide)
    elif mode == 'minimal':
        num_receptor_res = (~data['generate_mask']).sum().item()
        num_ligand_res = data['generate_mask'].sum().item()
        mask_receptor = data['mask_heavyatom'][~data['generate_mask']].reshape(-1)
        mask_peptide = data['mask_heavyatom'][data['generate_mask']].reshape(-1)
        pos_receptor = data['pos_heavyatom'][~data['generate_mask']].reshape(-1, 3)
        pos_peptide = data['pos_heavyatom'][data['generate_mask']].reshape(-1, 3)
        distance_mask = torch.logical_and(mask_receptor.unsqueeze(-1), mask_peptide.unsqueeze(0))
        atom_dist_map = compute_distance_matrix(pos_receptor, pos_peptide)
        atom_dist_map[~distance_mask] = 1e6  # large value to ignore
        atom_dist_map = atom_dist_map.reshape(num_receptor_res, max_num_heavyatoms, num_ligand_res, max_num_heavyatoms)
        min_dist_map =  torch.amin(atom_dist_map, dim=[1, 3])
        return min_dist_map
    
def contact_map(receptor, ligand, mode='CA', cutoff=RESIDUE_DISTANCE_THRESHOLD):
    distance_map = distance_map(receptor, ligand, mode=mode)
    return (distance_map < cutoff).float()

def remove_self_contacts(contact_map):
    contact_map = contact_map * (1 - torch.eye(contact_map.shape[1], device=contact_map.device)).unsqueeze(0)
    return contact_map
    
def contact_map_complex(data, mode, threshold=RESIDUE_DISTANCE_THRESHOLD):  #8
    if mode == 'CA':
        pos = data['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        distance_map = compute_distance_matrix(pos, pos)
        contact_map = (distance_map < threshold).float()
        contact_map = remove_self_contacts(contact_map)   #把 i=i 的位置强制设为 0。
        return contact_map
    
    elif mode == 'center':
        mask = data['mask_heavyatom']
        pos = data['pos_heavyatom']
        # B,L,3 / B,L,1 -> B,L,3
        center = torch.sum(pos, dim=2) / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        distance_map =  compute_distance_matrix(center, center)
        contact_map = (distance_map < threshold).float()
        contact_map = remove_self_contacts(contact_map)
        return contact_map
    
    elif mode == 'minimal':
        num_res = data['mask_heavyatom'].shape[0]
        mask = data['mask_heavyatom'].reshape(-1)
        pos = data['pos_heavyatom'].reshape(-1, 3)
        distance_mask = torch.logical_and(mask.unsqueeze(-1), mask.unsqueeze(0))
        atom_dist_map = compute_distance_matrix(pos, pos)
        atom_dist_map[~distance_mask] = 1e6  # large value to ignore
        atom_dist_map = atom_dist_map.reshape(num_res, max_num_heavyatoms, num_res, max_num_heavyatoms)
        min_dist_map =  torch.amin(atom_dist_map, dim=[1, 3])
        contact_map = (min_dist_map < threshold).float()
        contact_map = remove_self_contacts(contact_map)
        return contact_map
    
def create_unified_mask(batch, mode='complex'):
    """
    创建详细的接触掩码，包含各种接触类型
    batch: dict, 包含 'generate_mask' 键，指示配体和受体残基, N,L
    mode: str, 'complex' 表示处理复合物
    """
    
    generate_mask = batch['generate_mask'].bool()
    batch_size, N = generate_mask.shape
    
    ligand_mask = generate_mask
    receptor_mask = ~generate_mask
    
    if 'res_mask' in batch:
        res_mask = batch['res_mask'].bool()
        ligand_mask = ligand_mask & res_mask
        receptor_mask = receptor_mask & res_mask
    
    # 计算配体和受体数量（用于验证）
    n_ligand = ligand_mask.sum(dim=1)  # 每个样本的配体数
    n_receptor = receptor_mask.sum(dim=1)  # 每个样本的受体数
    
    # 1. 配体内部接触 (L-L)
    ligand_internal = ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
    
    # 2. 受体内部接触 (R-R) - 如果您需要
    receptor_internal = receptor_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
    
    # 3. 配体-受体接触 (L-R 和 R-L)
    ligand_receptor_LR = ligand_mask.unsqueeze(-1) & receptor_mask.unsqueeze(-2)
    ligand_receptor_RL = receptor_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)
    ligand_receptor = ligand_receptor_LR | ligand_receptor_RL
    
    # 4. 统一掩码（包含所有您关心的接触）
    unified_mask = ligand_internal | ligand_receptor
    
    if mode == 'complex':
        return unified_mask
    if mode == 'ligand':
        return ligand_internal
    
def distance_map_complex(data, mode):
    if mode == 'CA':
        pos = data['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        distance_map = compute_distance_matrix(pos, pos)
        return distance_map
    
    elif mode == 'center':
        mask = data['mask_heavyatom']
        pos = data['pos_heavyatom']
        # B,L,3 / B,L,1 -> B,L,3
        center = torch.sum(pos, dim=2) / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        distance_map =  compute_distance_matrix(center, center)
        return distance_map
    
    elif mode == 'minimal':
        num_res = data['mask_heavyatom'].shape[0]
        mask = data['mask_heavyatom'].reshape(-1)
        pos = data['pos_heavyatom'].reshape(-1, 3)
        distance_mask = torch.logical_and(mask.unsqueeze(-1), mask.unsqueeze(0))
        atom_dist_map = compute_distance_matrix(pos, pos)
        atom_dist_map[~distance_mask] = 1e6  # large value to ignore
        atom_dist_map = atom_dist_map.reshape(num_res, max_num_heavyatoms, num_res, max_num_heavyatoms)
        min_dist_map =  torch.amin(atom_dist_map, dim=[1, 3])
        return min_dist_map
    