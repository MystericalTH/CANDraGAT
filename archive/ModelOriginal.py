#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import os
import pandas as pd
from typing import OrderedDict
import torch
from torch.optim.lr_scheduler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
import random
from datetime import datetime
import json
import re
import errno
import sys
from deepchem.feat import ConvMolFeaturizer
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as PyGGMaxPool
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Data as PyGData
# from PyFingerprint.All_Fingerprint import get_fingerprint
from PyFingerprint.fingerprint import get_fingerprint

cuda = torch.device('cuda')
cpu = torch.device('cpu')
device = cuda

# # FraGAT Code
# ### ChemUtils

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# –––– Utils ––––


def GetSparseFromAdj(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]
    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    return torch.stack(index, dim=0), edge_attr


def graph_collate_fn(batch):
    Output = [torch.Tensor()]*len(batch[0])
    for Input in batch:
        for i, x in enumerate(Input):
            Output[i] = torch.cat((Output[i], x))
    return Output


class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer).squeeze(-1)
        label = torch.Tensor(label)
        #print("Size for RMSE")
        #print("Answer size: ", answer.size())
        #print("Label size: ", label.size())
        RMSE = F.mse_loss(answer, label, reduction='mean').sqrt()
        #SE = F.mse_loss(answer, label, reduction='none')
        #print("SE: ", SE)
        #MSE = SE.mean()
        #print("MSE: ", MSE)
        #RMSE = MSE.sqrt()
        return RMSE.item()


class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MAE = F.l1_loss(answer, label, reduction='mean')
        return MAE.item()


class MSE(object):
    def __init__(self):
        super(MSE, self).__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MSE = F.mse_loss(answer, label, reduction='mean')
        return MSE.item()


class PCC(object):
    def __init__(self):
        super(PCC, self).__init__()
        self.name = 'PCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        pcc = np.corrcoef(answer, label)
        return pcc[0][1]


class R2(object):
    def __init__(self):
        super(R2, self).__init__()
        self.name = 'R2'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        r_squared = r2_score(answer, label)
        return r_squared


class SRCC(object):
    def __init__(self):
        super(SRCC, self).__init__()
        self.name = 'Spearman Rank Cor. Coef.'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        #print("Size for MAE")
        srcc = spearmanr(answer, label)
        return srcc[0]
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def GetNeiList(mol):
    atomlist = mol.GetAtoms()
    TotalAtom = len(atomlist)
    NeiList = {}

    for atom in atomlist:
        atomIdx = atom.GetIdx()
        neighbors = atom.GetNeighbors()
        NeiList.update({"{}".format(atomIdx): []})
        for nei in neighbors:
            neiIdx = nei.GetIdx()
            NeiList["{}".format(atomIdx)].append(neiIdx)

    return NeiList


def GetAdjMat(mol):
    # Get the adjacency Matrix of the given molecule graph
    # If one node i is connected with another node j, then the element aij in the matrix is 1; 0 for otherwise.
    # The type of the bond is not shown in this matrix.

    NeiList = GetNeiList(mol)
    TotalAtom = len(NeiList)
    AdjMat = np.zeros([TotalAtom, TotalAtom])

    for idx in range(TotalAtom):
        neighbors = NeiList["{}".format(idx)]
        for nei in neighbors:
            AdjMat[idx, nei] = 1

    return AdjMat


def GetSingleBonds(mol):
    Single_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                bond_idx = bond.GetIdx()
                beginatom = bond.GetBeginAtomIdx()
                endatom = bond.GetEndAtomIdx()
                Single_bonds.append([bond_idx, beginatom, endatom])

    return Single_bonds


def GetAtomFeatures(atom):
    feature = np.zeros(72)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
    ]
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[43] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 10:
        print("atom degree larger than 10. Please check before featurizing.")
        raise RuntimeError

    feature[44 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[55] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[56] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc+57] = 1
    else:
        feature[62] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[63] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[64+hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[69] = 1
        try:
            # chi = atom.GetProp('_CIPCode')
            # ChiList = ['R','S']
            # loc = ChiList.index(chi)
            # feature[70+loc] = 1
            #print("Chirality resolving finished.")
            pass
        except:
            feature[70] = 0
            feature[71] = 0
    return feature


def GetBondFeatures(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    BondTypeList = [rdkit.Chem.rdchem.BondType.SINGLE,
                    rdkit.Chem.rdchem.BondType.DOUBLE,
                    rdkit.Chem.rdchem.BondType.TRIPLE,
                    rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in BondTypeList:
        loc = BondTypeList.index(type)
        feature[0+loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = conj

    # stereo
    stereo = bond.GetStereo()
    StereoList = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                  rdkit.Chem.rdchem.BondStereo.STEREOANY,
                  rdkit.Chem.rdchem.BondStereo.STEREOZ,
                  rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in StereoList:
        loc = StereoList.index(stereo)
        feature[6+loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature


def GetMolFeatureMat(mol):
    FeatureMat = []
    for atom in mol.GetAtoms():
        feature = GetAtomFeatures(atom)
        FeatureMat.append(feature.tolist())
    return FeatureMat


def GetMolFingerprints(SMILES, nBits):
    mol = Chem.MolFromSmiles(SMILES)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    FP = fp.ToBitString()
    FP_array = []
    for i in range(len(FP)):
        FP_value = float(FP[i])
        FP_array.append(FP_value)
    return FP_array

######################################################################


class ScaffoldGenerator(object):
    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, smiles):
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=self.include_chirality
        )


# ### AttentiveFP

# In[6]:


class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x


class AttentionCalculator(nn.Module):
    def __init__(self, FP_size, droprate):
        super(AttentionCalculator, self).__init__()
        self.FP_size = FP_size
        #self.align = nn.Linear(2*self.FP_size, 1)
        self.align = LinearBn(2*self.FP_size, 1)
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, FP_align, atom_neighbor_list):
        # size of input Tensors:
        # FP_align: [batch_size, max_atom_length, max_neighbor_length, 2*FP_size]
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length]

        batch_size, max_atom_length, max_neighbor_length, _ = FP_align.size()

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_neighbor_list.clone().to(device)
        attend_mask[attend_mask != max_atom_length - 1] = 1
        attend_mask[attend_mask == max_atom_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = atom_neighbor_list.clone()
        softmax_mask[softmax_mask != max_atom_length - 1] = 0
        softmax_mask[softmax_mask == max_atom_length - 1] = - \
            9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        # size of masks: [batch_szie, max_atom_length, max_neighbor_length, 1]

        # calculate attention value
        align_score = self.align(self.dropout(FP_align))
        align_score = F.leaky_relu(align_score)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, dim=-2)
        attention_weight = attention_weight * attend_mask
        # size: [batch_size, max_atom_length, max_neighbor_length, 1]

        return attention_weight


class ContextCalculator(nn.Module):
    def __init__(self, FP_size, droprate):
        super(ContextCalculator, self).__init__()
        #self.attend = nn.Linear(FP_size, FP_size)
        self.attend = LinearBn(FP_size, FP_size)
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, neighbor_FP, attention_score):
        # size of input Tensors:
        # neighbor_FP: [batch_size, max_atom_length, max_neighbor_length, FP_size]
        # attention_score: [batch_size, max_atom_length, max_neighbor_length, 1]

        neighbor_FP = self.dropout(neighbor_FP)
        neighbor_FP = self.attend(neighbor_FP)
        # after sum, the dim -2 disappears.
        context = torch.sum(torch.mul(attention_score, neighbor_FP), -2)
        context = F.elu(context)
        # context size: [batch_size, max_atom_length, FP_size]
        # a context vector for each atom in each molecule.
        return context


class FPTranser(nn.Module):
    def __init__(self, FP_size):
        super(FPTranser, self).__init__()
        self.FP_size = FP_size
        self.GRUCell = nn.GRUCell(self.FP_size, self.FP_size)

    def forward(self, atom_FP, context_FP, atom_neighbor_list):
        # size of input Tensors:
        # atom_FP: [batch_size, max_atom_length, FP_size]
        # context_FP: [batch_size, max_atom_length, FP_size]

        batch_size, max_atom_length, _ = atom_FP.size()

        # GRUCell cannot treat 3D Tensors.
        # flat the mol dim and atom dim.
        context_FP_reshape = context_FP.view(
            batch_size * max_atom_length, self.FP_size)
        atom_FP_reshape = atom_FP.view(
            batch_size * max_atom_length, self.FP_size)
        new_atom_FP_reshape = self.GRUCell(context_FP_reshape, atom_FP_reshape)
        new_atom_FP = new_atom_FP_reshape.view(
            batch_size, max_atom_length, self.FP_size)
        activated_new_atom_FP = F.relu(new_atom_FP)
        # size:
        # [batch_size, max_atom_length, FP_size]

        # calculate new_neighbor_FP
        new_neighbor_FP = [activated_new_atom_FP[i]
                           [atom_neighbor_list[i]] for i in range(batch_size)]
        new_neighbor_FP = torch.stack(new_neighbor_FP, dim=0)
        # size:
        # [batch_size, max_atom_length, max_neighbor_length, FP_size]

        return new_atom_FP, activated_new_atom_FP, new_neighbor_FP


class FPInitializer(nn.Module):
    def __init__(self, atom_feature_size, bond_feature_size, FP_size):
        super(FPInitializer, self).__init__()
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.FP_size = FP_size
        #self.atom_fc = nn.Linear(self.atom_feature_size, self.FP_size)
        self.atom_fc = LinearBn(self.atom_feature_size, self.FP_size)
        #self.nei_fc = nn.Linear((self.atom_feature_size + self.bond_feature_size), self.FP_size)
        self.nei_fc = LinearBn(
            (self.atom_feature_size + self.bond_feature_size), self.FP_size)

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        batch_size, max_atom_length, _ = atom_features.size()

        # generate atom_neighbor_features from atom_features by atom_neighbor_list, size is [batch, atom, neighbor, atom_feature_size]
        atom_neighbor_features = [
            atom_features[i][atom_neighbor_list[i]] for i in range(batch_size)]
        atom_neighbor_features = torch.stack(atom_neighbor_features, dim=0)

        # generate bond_neighbor_features from bond_features by bond_neighbor list, size is [batch, atom, neighbor, bond_feature_size]
        bond_neighbor_features = [
            bond_features[i][bond_neighbor_list[i]] for i in range(batch_size)]
        bond_neighbor_features = torch.stack(bond_neighbor_features, dim=0)

        # concate bond_neighbor_features and atom_neighbor_features, and then transform it from
        # [batch, atom, neighbor, atom_feature_size+bond_feature_size] to [batch, atom, neighbor, FP_size]
        neighbor_FP = torch.cat(
            [atom_neighbor_features, bond_neighbor_features], dim=-1)
        # print(neighbor_FP.size())
        neighbor_FP = self.nei_fc(neighbor_FP)
        neighbor_FP = F.leaky_relu(neighbor_FP)

        # transform atom_features from [batch, atom, atom_feature_size] to [batch, atom, FP_size]
        atom_FP = self.atom_fc(atom_features)
        atom_FP = F.leaky_relu(atom_FP)

        return atom_FP, neighbor_FP


class FPInitializerNew(nn.Module):
    # with mixture item.
    def __init__(self, atom_feature_size, bond_feature_size, FP_size, droprate):
        super(FPInitializerNew, self).__init__()
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.FP_size = FP_size
        #self.atom_fc = nn.Linear(self.atom_feature_size, self.FP_size)
        self.atom_fc = nn.Sequential(
            LinearBn(self.atom_feature_size, self.FP_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace=True)
        )
        self.bond_fc = nn.Sequential(
            LinearBn(self.bond_feature_size, self.FP_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace=True)
        )
        self.nei_fc = nn.Sequential(
            LinearBn(3*self.FP_size, self.FP_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            LinearBn(self.FP_size, self.FP_size),
            nn.ReLU(inplace=True)
        )
        #self.nei_fc = nn.Linear((self.atom_feature_size + self.bond_feature_size), self.FP_size)
        #self.nei_fc = LinearBn((self.atom_feature_size + self.bond_feature_size), self.FP_size)

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        batch_size, max_atom_length, _ = atom_features.size()

        atom_FP = self.atom_fc(atom_features)
        #[batch_size, max_atom_length, FP_size]

        bond_FP = self.bond_fc(bond_features)
        #[batch_size, max_bond_length, FP_size]

        # generate atom_neighbor_FP from atom_FP by atom_neighbor_list,
        # size is [batch, atom, neighbor, FP_size]
        atom_neighbor_FP = [atom_FP[i][atom_neighbor_list[i]]
                            for i in range(batch_size)]
        atom_neighbor_FP = torch.stack(atom_neighbor_FP, dim=0)

        # generate bond_neighbor_FP from bond_FP by bond_neighbor list,
        # size is [batch, atom, neighbor, FP_size]
        bond_neighbor_FP = [bond_FP[i][bond_neighbor_list[i]]
                            for i in range(batch_size)]
        bond_neighbor_FP = torch.stack(bond_neighbor_FP, dim=0)

        # generate mixture item
        # size is [batch, atom, neighbor, FP_size]
        mixture = atom_neighbor_FP + bond_neighbor_FP - \
            atom_neighbor_FP * bond_neighbor_FP

        # concate bond_neighbor_FP and atom_neighbor_FP and mixture item, and then transform it from
        # [batch, atom, neighbor, 3*FP_size] to [batch, atom, neighbor, FP_size]
        neighbor_FP = torch.cat(
            [atom_neighbor_FP, bond_neighbor_FP, mixture], dim=-1)
        # print(neighbor_FP.size())
        neighbor_FP = self.nei_fc(neighbor_FP)
        #neighbor_FP = F.leaky_relu(neighbor_FP)

        # transform atom_features from [batch, atom, atom_feature_size] to [batch, atom, FP_size]
        #atom_FP = self.atom_fc(atom_features)
        #atom_FP = F.leaky_relu(atom_FP)

        return atom_FP, neighbor_FP
###########################################################################################################


class AttentiveFPLayer(nn.Module):
    def __init__(self, FP_size, droprate):
        super(AttentiveFPLayer, self).__init__()
        self.FP_size = FP_size
        self.attentioncalculator = AttentionCalculator(self.FP_size, droprate)
        self.contextcalculator = ContextCalculator(self.FP_size, droprate)
        self.FPtranser = FPTranser(self.FP_size)

    def forward(self, atom_FP, neighbor_FP, atom_neighbor_list):
        # align atom FP and its neighbors' FP to generate [hv, hu]
        FP_align = self.feature_align(atom_FP, neighbor_FP)
        # FP_align: [batch_size, max_atom_length, max_neighbor_length, 2*FP_size]

        # calculate attention score evu
        attention_score = self.attentioncalculator(
            FP_align, atom_neighbor_list)
        # attention_score: [batch_size, max_atom_length, max_neighbor_length, 1]

        # calculate context FP
        context_FP = self.contextcalculator(neighbor_FP, attention_score)
        # context_FP: [batch_size, max_atom_length, FP_size)

        # transmit FPs between atoms.
        activated_new_atom_FP, new_atom_FP, neighbor_FP = self.FPtranser(
            atom_FP, context_FP, atom_neighbor_list)

        return activated_new_atom_FP, new_atom_FP, neighbor_FP

    def feature_align(self, atom_FP, neighbor_FP):
        # size of input Tensors:
        # atom_FP: [batch_size, max_atom_length, FP_size]
        # neighbor_FP: [batch_size, max_atom_length, max_neighbor_length, FP_size]

        batch_size, max_atom_length, max_neighbor_length, _ = neighbor_FP.size()

        atom_FP = atom_FP.unsqueeze(-2)
        # [batch_size, max_atom_length, 1, FP_size]
        atom_FP = atom_FP.expand(
            batch_size, max_atom_length, max_neighbor_length, self.FP_size)
        # [batch_size, max_atom_length, max_neighbor_length, FP_size]

        FP_align = torch.cat([atom_FP, neighbor_FP], dim=-1)
        # size: [batch_size, max_atom_length, max_neighborlength, 2*FP_size]

        return FP_align

###########################################################################################################


class AttentiveFP_atom(nn.Module):
    def __init__(self, atom_feature_size, bond_feature_size, FP_size, layers, droprate):
        super(AttentiveFP_atom, self).__init__()
        self.FPinitializer = FPInitializerNew(
            atom_feature_size, bond_feature_size, FP_size, droprate)
        self.AttentiveFPLayers = nn.ModuleList()
        for i in range(layers):
            self.AttentiveFPLayers.append(AttentiveFPLayer(FP_size, droprate))

    def forward(self, atom_features, bond_features, atom_neighbor_list, bond_neighbor_list):
        # size of input Tensors:
        # atom_features: [batch_size, max_atom_length, atom_feature_length], with pads in dim=1
        # bond_features: [batch_size, max_bond_length, bond_feature_length], with pads in dim=1
        # atom_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2
        # bond_neighbor_list: [batch_size, max_atom_length, max_neighbor_length], with pads in dim=1 and 2

        # atom_features and neighbor_features initializing.
        atom_FP, neighbor_FP = self.FPinitializer(
            atom_features, bond_features, atom_neighbor_list, bond_neighbor_list)

        # use attentiveFPlayers to update the atom_features and the neighbor_features.
        for layer in self.AttentiveFPLayers:
            atom_FP, _, neighbor_FP = layer(
                atom_FP, neighbor_FP, atom_neighbor_list)

        return atom_FP


class AttentiveFP_mol(nn.Module):
    def __init__(self, layers, FP_size, droprate):
        super(AttentiveFP_mol, self).__init__()
        self.layers = layers
        self.FP_size = FP_size
        #self.align = nn.Linear(2 * self.FP_size, 1)
        self.align = LinearBn(2 * self.FP_size, 1)
        self.dropout = nn.Dropout(p=droprate)
        #self.attend = nn.Linear(self.FP_size, self.FP_size)
        self.attend = LinearBn(self.FP_size, self.FP_size)
        self.mol_GRUCell = nn.GRUCell(self.FP_size, self.FP_size)

    def forward(self, atom_FP, atom_mask):
        # size of input Tensors:
        # atom_FP: [batch_size, atom_length, FP_size]
        # atom_mask: [batch_size, atom_length]

        batch_size, max_atom_length, _ = atom_FP.size()
        atom_mask = atom_mask.unsqueeze(2)
        # [batch_size, atom_length, 1]
        super_node_FP = torch.sum(atom_FP * atom_mask, dim=-2)
        # FP of super node S is set to be sum of all of the atoms' FP initially.
        # pad nodes is eliminated after atom_FP * atom_mask
        # super node FP size: [batch, FP_size]

        # generate masks to eliminate the influence of pad nodes.
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        # [batch, atom_length, 1]

        activated_super_node_FP = F.relu(super_node_FP)

        for i in range(self.layers):
            super_node_FP_expand = activated_super_node_FP.unsqueeze(-2)
            super_node_FP_expand = super_node_FP_expand.expand(
                batch_size, max_atom_length, self.FP_size)
            # [batch, max_atom_length, FP_size]

            super_node_align = torch.cat(
                [super_node_FP_expand, atom_FP], dim=-1)
            # [batch, max_atom_length, 2*FP_size]
            super_node_align_score = self.align(super_node_align)
            super_node_align_score = F.leaky_relu(super_node_align_score)
            # [batch, max_atom_length, 1]
            super_node_align_score = super_node_align_score + mol_softmax_mask
            super_node_attention_weight = F.softmax(super_node_align_score, -2)
            super_node_attention_weight = super_node_attention_weight * atom_mask
            # [batch_size, max_atom_length, 1]

            atom_FP_transform = self.attend(self.dropout(atom_FP))
            # [batch_size, max_atom_length, FP_size]
            super_node_context = torch.sum(
                torch.mul(super_node_attention_weight, atom_FP_transform), -2)
            super_node_context = F.elu(super_node_context)
            # [batch_size, FP_size]
            # the dim -2 is eliminated after sum function.
            super_node_FP = self.mol_GRUCell(super_node_context, super_node_FP)

            # do nonlinearity
            activated_super_node_FP = F.relu(super_node_FP)

        return super_node_FP, activated_super_node_FP

        ###########################################################


# ### Featurizer

# In[7]:


class BasicFeaturizer(object):
    def __init__(self):
        super(BasicFeaturizer, self).__init__()

    def featurize(self, item):
        raise NotImplementedError(
            "Molecule Featurizer not implemented.")


class AttentiveFPFeaturizer(BasicFeaturizer):
    def __init__(self, atom_feature_size, bond_feature_size, max_degree, max_frag, mode):
        super(AttentiveFPFeaturizer, self).__init__()
        self.max_atom_num = 0
        self.max_bond_num = 0
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.max_degree = max_degree
        self.mode = mode
        self.max_frag = max_frag

    def prefeaturize(self, dataset):
        entire_atom_features = []
        entire_bond_features = []
        entire_atom_neighbor_list = []
        entire_bond_neighbor_list = []
        entire_atom_mask = []
        cell_line_list = []
        SMILES_list = []

        for item in dataset:
            CellLine = item['cell_line']
            SMILES = item['SMILES']
            mol = Chem.MolFromSmiles(SMILES)

            mol_atom_feature = np.zeros(
                [self.max_atom_num, self.atom_feature_size])
            mol_bond_feature = np.zeros(
                [self.max_bond_num, self.bond_feature_size])

            mol_atom_neighbor_list = np.zeros(
                [self.max_atom_num, self.max_degree])
            mol_bond_neighbor_list = np.zeros(
                [self.max_atom_num, self.max_degree])
            mol_atom_neighbor_list.fill(self.pad_atom_idx)
            mol_bond_neighbor_list.fill(self.pad_bond_idx)

            mol_atom_mask = np.zeros([self.max_atom_num])

            #  generate five information of a molecule.

            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                mol_atom_mask[idx] = 1.0
                atom_feature = GetAtomFeatures(atom)
                mol_atom_feature[idx] = atom_feature

                neighbors = atom.GetNeighbors()
                pointer = 0
                for neighbor in neighbors:
                    nei_idx = neighbor.GetIdx()
                    mol_atom_neighbor_list[idx][pointer] = nei_idx
                    pointer += 1

            bond_pointer = np.zeros([self.max_atom_num])
            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                bond_feature = GetBondFeatures(bond)
                mol_bond_feature[idx] = bond_feature

                start_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()

                start_atom_pointer = int(bond_pointer[start_atom])
                end_atom_pointer = int(bond_pointer[end_atom])

                mol_bond_neighbor_list[start_atom][start_atom_pointer] = idx
                mol_bond_neighbor_list[end_atom][end_atom_pointer] = idx

                bond_pointer[start_atom] += 1
                bond_pointer[end_atom] += 1
            cell_line_list.append(CellLine)
            entire_atom_features.append(mol_atom_feature)
            entire_bond_features.append(mol_bond_feature)
            entire_atom_neighbor_list.append(mol_atom_neighbor_list)
            entire_bond_neighbor_list.append(mol_bond_neighbor_list)
            entire_atom_mask.append(mol_atom_mask)
            SMILES_list.append(SMILES)

        return [cell_line_list, entire_atom_features, entire_bond_features, entire_atom_neighbor_list, entire_bond_neighbor_list, entire_atom_mask, SMILES_list]

    def featurizenew(self, dataset, omicsdataset, index, mol, value, Frag):
        [cell_line_list, entire_atom_features, entire_bond_features, entire_atom_neighbor_list,
            entire_bond_neighbor_list, entire_atom_mask, SMILES_list] = dataset  # from prefeaturizer

        mutdataset = omicsdataset[0]
        geneexpdataset = omicsdataset[1]
        methdataset = omicsdataset[2]
        genecndataset = omicsdataset[3]
        cell_line = cell_line_list[index]

        mut_cell_line = torch.Tensor(
            mutdataset.loc[cell_line].values).view(1, -1)

        geneexp_cell_line = torch.Tensor(
            geneexpdataset.loc[cell_line].values).view(-1)
        meth_cell_line = torch.Tensor(
            methdataset.loc[cell_line].values).view(-1)
        genecn_cell_line = torch.Tensor(
            genecndataset.loc[cell_line].values).view(-1)

        # SMILES = SMILES_list[index]
        # maccs, pubchemfp = self.GetFingerprints(SMILES)


        mol_atom_features = entire_atom_features[index]
        mol_bond_features = entire_bond_features[index]
        mol_atom_neighbor_list = entire_atom_neighbor_list[index]
        mol_bond_neighbor_list = entire_bond_neighbor_list[index]
        mol_atom_mask = entire_atom_mask[index]

        mol_atom_features = torch.Tensor(mol_atom_features)
        mol_bond_features = torch.Tensor(mol_bond_features)
        mol_atom_neighbor_list = torch.Tensor(mol_atom_neighbor_list).long()
        mol_bond_neighbor_list = torch.Tensor(mol_bond_neighbor_list).long()
        mol_atom_mask = torch.Tensor(mol_atom_mask)

        label = []
        label.append(float(value))
        label = torch.Tensor(label)

        label.unsqueeze_(-1)

        # sizes:
        # mol_atom_features: [max_atom_length, atom_feature_size]
        # mol_bond_features: [max_bond_length, bond_feature_size]
        # mol_atom_neighbor_list: [max_atom_length, max_degree]
        # mol_bond_neighbor_list: [max_atom_length, max_degree]
        # mol_atom_mask: [max_atom_length]

        if Frag:
            if self.mode == "TRAIN":
                # create the information of one molecule.
                mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx = self.CutSingleBond(
                    mol, mol_atom_neighbor_list, mol_bond_neighbor_list)
                # No matter whether a bond has been cut, the structure of the return are the same.
                # However, if no bond is cut, the two neighbor_list_changed are the same as the original neighbor lists.
                # and the start_atom, end_atom, bond_idx are None.
                if bond_idx:
                    mask1, mask2 = self.GetComponentMasks(
                        start_atom, end_atom, mol_atom_neighbor_list_changed)
                    mask1 = torch.Tensor(mask1)
                    mask2 = torch.Tensor(mask2)
                    mol_frag_mask1 = mask1 * mol_atom_mask
                    mol_frag_mask2 = mask2 * mol_atom_mask
                    # In the atom_neighbor_list, all atoms are set to be connected with the pad node.
                    # so that the generated mask1 and mask2 are not correct because the pad nodes are set to 1.
                    # That's why we should use mask1 * mol_atom_mask to set the pad nodes to 0.

                    bond_idx = torch.Tensor([bond_idx]).long()

                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=start_atom, end_atom=end_atom, bondidx=bond_idx, frag_masks=[mol_frag_mask1, mol_frag_mask2])
                    # Return in such format: Origin Graph, Frags, Junction Tree
                    return ([
                        mut_cell_line,
                        geneexp_cell_line,
                        meth_cell_line,
                        genecn_cell_line,
                        mol_atom_features,
                        mol_bond_features,
                        mol_atom_neighbor_list,
                        mol_bond_neighbor_list,
                        mol_atom_mask,
                        mol_atom_neighbor_list_changed,
                        mol_bond_neighbor_list_changed,
                        mol_frag_mask1,
                        mol_frag_mask2,
                        bond_idx,
                        JT_bond_features,
                        JT_atom_neighbor_list,
                        JT_bond_neighbor_list,
                        JT_mask,
                        # maccs,
                        # pubchemfp
                    ],
                        label)

                else:
                    # No single bonds can be cut.
                    # Return in such format: Origin Graph, Frags, Junction Tree
                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=None, end_atom=None, bondidx=[], frag_masks=[])
                    return ([mut_cell_line,
                            geneexp_cell_line,
                            meth_cell_line,
                            genecn_cell_line,
                            mol_atom_features,
                            mol_bond_features,
                            mol_atom_neighbor_list,
                            mol_bond_neighbor_list,
                            mol_atom_mask,
                            mol_atom_neighbor_list_changed,
                            mol_bond_neighbor_list_changed,
                            mol_atom_mask,
                            torch.zeros(self.max_atom_num),
                            torch.Tensor([-1]).long(),
                            JT_bond_features,
                            JT_atom_neighbor_list,
                            JT_bond_neighbor_list,
                            JT_mask,
                            # maccs,
                            # pubchemfp
                             ],
                            label)

            elif self.mode == 'EVAL':
                # create a 'batch' of molecules
                extended_atom_features = torch.Tensor([])
                extended_bond_features = torch.Tensor([])
                extended_atom_neighbor_list = torch.Tensor([]).long()
                extended_bond_neighbor_list = torch.Tensor([]).long()
                extended_atom_mask = torch.Tensor([])

                extended_atom_neighbor_list_changed = torch.Tensor([]).long()
                extended_bond_neighbor_list_changed = torch.Tensor([]).long()
                extended_frag_mask1 = torch.Tensor([])
                extended_frag_mask2 = torch.Tensor([])
                extended_bond_idx = torch.Tensor([]).long()

                extended_JT_bond_features = torch.Tensor([])
                extended_JT_atom_neighbor_list = torch.Tensor([]).long()
                extended_JT_bond_neighbor_list = torch.Tensor([]).long()
                extended_JT_mask = torch.Tensor([])

                extended_label = torch.Tensor([])

                extended_mut_cell_line = torch.Tensor([])
                extended_geneexp_cell_line = torch.Tensor([])
                extended_meth_cell_line = torch.Tensor([])
                extended_genecn_cell_line = torch.Tensor([])
                # extended_maccs = torch.Tensor([])
                # extended_pubchemfp = torch.Tensor([])

                SingleBondList = GetSingleBonds(mol)
                #assert len(SingleBondList) > 0
                if len(SingleBondList) == 0:
                    # No bond is cut. Only one molecule information is used.
                    # Original Graph, no change
                    extended_atom_features = self.CatTensor(
                        extended_atom_features, mol_atom_features)
                    extended_bond_features = self.CatTensor(
                        extended_bond_features, mol_bond_features)
                    extended_atom_neighbor_list = self.CatTensor(
                        extended_atom_neighbor_list, mol_atom_neighbor_list)
                    extended_bond_neighbor_list = self.CatTensor(
                        extended_bond_neighbor_list, mol_bond_neighbor_list)
                    extended_atom_mask = self.CatTensor(
                        extended_atom_mask, mol_atom_mask)

                    # Frags, no change.
                    extended_atom_neighbor_list_changed = self.CatTensor(
                        extended_atom_neighbor_list_changed, mol_atom_neighbor_list)
                    extended_bond_neighbor_list_changed = self.CatTensor(
                        extended_bond_neighbor_list_changed, mol_bond_neighbor_list)
                    extended_frag_mask1 = self.CatTensor(
                        extended_frag_mask1, mol_atom_mask)
                    extended_frag_mask2 = self.CatTensor(
                        extended_frag_mask2, torch.zeros(self.max_atom_num))
                    extended_bond_idx = self.CatTensor(
                        extended_bond_idx, torch.Tensor([-1]).long())

                    # Junction Tree
                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=None,
                        end_atom=None, bondidx=[], frag_masks=[])

                    extended_JT_bond_features = self.CatTensor(
                        extended_JT_bond_features, JT_bond_features)
                    extended_JT_atom_neighbor_list = self.CatTensor(
                        extended_JT_atom_neighbor_list, JT_atom_neighbor_list)
                    extended_JT_bond_neighbor_list = self.CatTensor(
                        extended_JT_bond_neighbor_list, JT_bond_neighbor_list)
                    extended_JT_mask = self.CatTensor(
                        extended_JT_mask, JT_mask)
                    extended_label = self.CatTensor(extended_label, label)

                    extended_mut_cell_line = self.CatTensor(
                        extended_mut_cell_line, mut_cell_line)
                    extended_geneexp_cell_line = self.CatTensor(
                        extended_geneexp_cell_line, geneexp_cell_line)
                    extended_meth_cell_line = self.CatTensor(
                        extended_meth_cell_line, meth_cell_line)
                    extended_genecn_cell_line = self.CatTensor(
                        extended_genecn_cell_line, genecn_cell_line)

                    # extended_maccs = self.CatTensor(extended_maccs, maccs)
                    # extended_pubchemfp = self.CatTensor(
                    #     extended_pubchemfp, pubchemfp)

                else:
                    for bond in SingleBondList:
                        # Cut one bond
                        mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx = self.CutOneBond(
                            bond, mol_atom_neighbor_list, mol_bond_neighbor_list)
                        # if True:
                        mask1, mask2 = self.GetComponentMasks(
                            start_atom, end_atom, mol_atom_neighbor_list_changed)
                        mask1 = torch.Tensor(mask1)
                        mask2 = torch.Tensor(mask2)
                        mol_frag_mask1 = mask1 * mol_atom_mask
                        mol_frag_mask2 = mask2 * mol_atom_mask
                        bond_idx = torch.Tensor([bond_idx]).long()
                        # print(bond_idx.size())
                        # print(bond_idx)
                        JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                            mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=start_atom,
                            end_atom=end_atom, bondidx=bond_idx, frag_masks=[mol_frag_mask1, mol_frag_mask2])

                        extended_mut_cell_line = self.CatTensor(
                            extended_mut_cell_line, mut_cell_line)
                        extended_geneexp_cell_line = self.CatTensor(
                            extended_geneexp_cell_line, geneexp_cell_line)
                        extended_meth_cell_line = self.CatTensor(
                            extended_meth_cell_line, meth_cell_line)
                        extended_genecn_cell_line = self.CatTensor(
                            extended_genecn_cell_line, genecn_cell_line)

                        # extended_maccs = self.CatTensor(extended_maccs, maccs)
                        # extended_pubchemfp = self.CatTensor(
                        #     extended_pubchemfp, pubchemfp)

                        extended_atom_features = self.CatTensor(
                            extended_atom_features, mol_atom_features)
                        extended_bond_features = self.CatTensor(
                            extended_bond_features, mol_bond_features)
                        extended_atom_neighbor_list = self.CatTensor(extended_atom_neighbor_list,
                                                                     mol_atom_neighbor_list)
                        extended_bond_neighbor_list = self.CatTensor(extended_bond_neighbor_list,
                                                                     mol_bond_neighbor_list)
                        extended_atom_mask = self.CatTensor(
                            extended_atom_mask, mol_atom_mask)

                        extended_atom_neighbor_list_changed = self.CatTensor(
                            extended_atom_neighbor_list_changed, mol_atom_neighbor_list_changed)
                        extended_bond_neighbor_list_changed = self.CatTensor(
                            extended_bond_neighbor_list_changed, mol_bond_neighbor_list_changed)
                        extended_frag_mask1 = self.CatTensor(
                            extended_frag_mask1, mol_frag_mask1)
                        extended_frag_mask2 = self.CatTensor(
                            extended_frag_mask2, mol_frag_mask2)
                        extended_bond_idx = self.CatTensor(
                            extended_bond_idx, bond_idx)

                        extended_JT_bond_features = self.CatTensor(
                            extended_JT_bond_features, JT_bond_features)
                        extended_JT_atom_neighbor_list = self.CatTensor(extended_JT_atom_neighbor_list,
                                                                        JT_atom_neighbor_list)
                        extended_JT_bond_neighbor_list = self.CatTensor(extended_JT_bond_neighbor_list,
                                                                        JT_bond_neighbor_list)
                        extended_JT_mask = self.CatTensor(
                            extended_JT_mask, JT_mask)

                        extended_label = self.CatTensor(extended_label, label)

                return ([
                    extended_mut_cell_line,
                    extended_geneexp_cell_line,
                    extended_meth_cell_line,
                    extended_genecn_cell_line,
                    extended_atom_features,
                    extended_bond_features,
                    extended_atom_neighbor_list,
                    extended_bond_neighbor_list,
                    extended_atom_mask,
                    extended_atom_neighbor_list_changed,
                    extended_bond_neighbor_list_changed,
                    extended_frag_mask1,
                    extended_frag_mask2,
                    extended_bond_idx,
                    extended_JT_bond_features,
                    extended_JT_atom_neighbor_list,
                    extended_JT_bond_neighbor_list,
                    extended_JT_mask,
                    # extended_maccs,
                    # extended_pubchemfp
                ],
                    extended_label)

            else:
                print("Wrong mode.")
                raise RuntimeError

        return [mut_cell_line, geneexp_cell_line, meth_cell_line, genecn_cell_line, mol_atom_features, mol_bond_features, mol_atom_neighbor_list, mol_bond_neighbor_list, mol_atom_mask,
                # maccs, pubchemfp
                ], label

    # def GetFingerprints(self, SMILES):
    #     maccsfps = get_fingerprint(SMILES, fp_type='maccs')
    #     pubchemfps = get_fingerprint(SMILES, fp_type='pubchem')

    #     maccs = maccsfps.to_numpy().astype(int)
    #     pubchemfp = pubchemfps.to_numpy().astype(int)

    #     maccs = torch.Tensor(maccs).unsqueeze(0)
    #     pubchemfp = torch.Tensor(pubchemfp).unsqueeze(0)

    #     return maccs, pubchemfp

    def GetPad(self, dataset):
        # dataset format: [{"SMILES": smiles, "Value": value}]

        for item in dataset:
            smiles = item["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            total_atom_num = len(mol.GetAtoms())
            total_bond_num = len(mol.GetBonds())
            self.max_atom_num = max(self.max_atom_num, total_atom_num)
            self.max_bond_num = max(self.max_bond_num, total_bond_num)

        self.pad_atom_idx = self.max_atom_num
        self.pad_bond_idx = self.max_bond_num

        self.max_atom_num += 1
        self.max_bond_num += 1

    def CatTensor(self, stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = torch.cat(
            [stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

    def CutOneBond(self, bond, mol_atom_neighbor_list, mol_bond_neighbor_list):   # for eval
        _mol_atom_neighbor_list = mol_atom_neighbor_list.clone()
        _mol_bond_neighbor_list = mol_bond_neighbor_list.clone()
        # insulate.
        [bond_idx, start_atom_idx, end_atom_idx] = bond
        assert end_atom_idx in _mol_atom_neighbor_list[start_atom_idx]
        assert start_atom_idx in _mol_atom_neighbor_list[end_atom_idx]
        # print(start_atom_idx)
        # print(end_atom_idx)
        # print(bond_idx)

        loc = _mol_atom_neighbor_list[start_atom_idx].tolist().index(
            end_atom_idx)
        _mol_atom_neighbor_list[start_atom_idx][loc] = self.pad_atom_idx
        loc = _mol_atom_neighbor_list[end_atom_idx].tolist().index(
            start_atom_idx)
        _mol_atom_neighbor_list[end_atom_idx][loc] = self.pad_atom_idx

        loc = _mol_bond_neighbor_list[start_atom_idx].tolist().index(bond_idx)
        _mol_bond_neighbor_list[start_atom_idx][loc] = self.pad_bond_idx
        loc = _mol_bond_neighbor_list[end_atom_idx].tolist().index(bond_idx)
        _mol_bond_neighbor_list[end_atom_idx][loc] = self.pad_bond_idx

        return _mol_atom_neighbor_list, _mol_bond_neighbor_list, start_atom_idx, end_atom_idx, bond_idx

    def CutSingleBond(self, mol, mol_atom_neighbor_list, mol_bond_neighbor_list):   # for train
        # This function will calculate the SingleBondList and tries to cut a random one.
        # if len(SingleBondList) > 0, one single bond will be cut. The two neighbor lists will be modified.
        # the return is [mol_atom_neighbor_list_changed, mol_bond_neighbor_list_changed, start_atom, end_atom, bond_idx]
        # and if len(SingleBondList) == 0, no single bond will be cut. The two neighbor lists will not be modified.
        # the return is [mol_atom_neighbor_list, mol_bond_neighbor_list, None, None, None]
        # This function is compatible with the molecules that cannot be cut.

        # mol_atom_neighbor_list and mol_bond_neighbor_list are original neighbor lists that transmit to this function.
        # However, using neighbor_list[x,x] = xxx will exactly change the value of the original neighbor lists.
        # so ,the Tensors should be cloned first, to make sure that the Tensors outside of this function will not be changed.
        _mol_atom_neighbor_list = mol_atom_neighbor_list.clone()
        _mol_bond_neighbor_list = mol_bond_neighbor_list.clone()
        # insulate.
        SingleBondList = GetSingleBonds(mol)
        if len(SingleBondList) > 0:

            # Choose one bond to cut.
            random.shuffle(SingleBondList)
            [bond_idx, start_atom_idx, end_atom_idx] = SingleBondList[0]
            assert end_atom_idx in _mol_atom_neighbor_list[start_atom_idx]
            assert start_atom_idx in _mol_atom_neighbor_list[end_atom_idx]

            # modify the two neighbor lists based on the chosen bond.
            loc = _mol_atom_neighbor_list[start_atom_idx].tolist().index(
                end_atom_idx)
            _mol_atom_neighbor_list[start_atom_idx][loc] = self.pad_atom_idx
            loc = _mol_atom_neighbor_list[end_atom_idx].tolist().index(
                start_atom_idx)
            _mol_atom_neighbor_list[end_atom_idx][loc] = self.pad_atom_idx

            loc = _mol_bond_neighbor_list[start_atom_idx].tolist().index(
                bond_idx)
            _mol_bond_neighbor_list[start_atom_idx][loc] = self.pad_bond_idx
            loc = _mol_bond_neighbor_list[end_atom_idx].tolist().index(
                bond_idx)
            _mol_bond_neighbor_list[end_atom_idx][loc] = self.pad_bond_idx

            return _mol_atom_neighbor_list, _mol_bond_neighbor_list, start_atom_idx, end_atom_idx, bond_idx
        else:
            # no bond can be cut. _nei_list is same as the original one
            return _mol_atom_neighbor_list, _mol_bond_neighbor_list, None, None, None

    def GetComponentMasks(self, root_node1, root_node2, mol_atom_neighbor_list):
        mask1 = self.ComponentSearch(
            mol_atom_neighbor_list, self.max_atom_num, root_node1)
        mask2 = self.ComponentSearch(
            mol_atom_neighbor_list, self.max_atom_num, root_node2)
        assert len(mask1) == len(mask2)
        return mask1, mask2

    def ComponentSearch(self, mol_atom_neighbor_list, max_atom_num, root_node):
        candidate_set = []
        mask = np.zeros([max_atom_num])
        mask[root_node] = 1
        candidate_set.append(root_node)

        while len(candidate_set) > 0:
            node = candidate_set[0]
            candidate_set.pop(0)

            neighbors = mol_atom_neighbor_list[node]
            for nei in neighbors:
                if mask[nei] == 0:
                    candidate_set.append(nei)
                    mask[nei] = 1

        assert len(candidate_set) == 0
        return mask

    def CreateJunctionTree(self, mol_bond_features, atom_neighbor_list, bond_neighbor_list, start_atom, end_atom, bondidx, frag_masks):
        # For simple, in this version, we only consider that 0 or 1 bond is cut.
        # The case that multiple bonds are cut will be considered in the furture version.
        pad_bond_feature = torch.zeros(
            1, self.bond_feature_size)        # [1, bond_feature_size]
        cut_bonds_num = len(bondidx)
        if cut_bonds_num == 0:
            # [2, bond_feature_size]. max_bond_num = 2
            JT_bond_feature = torch.cat([pad_bond_feature, pad_bond_feature])

            # max_frag_num = 2, max_atom_num = 3, for 1 pad node.
            JT_atom_neighbor_list = np.zeros([3, self.max_degree])
            JT_atom_neighbor_list.fill(2)
            JT_bond_neighbor_list = np.zeros([3, self.max_degree])
            JT_bond_neighbor_list.fill(1)

            JT_atom_neighbor_list = torch.Tensor(JT_atom_neighbor_list).long()
            JT_bond_neighbor_list = torch.Tensor(JT_bond_neighbor_list).long()

            JT_mask = torch.Tensor([1.0, 0.0, 0.0])

        elif cut_bonds_num == 1:
            # [2, bond_feature_size]
            JT_bond_feature = torch.cat(
                [mol_bond_features[bondidx], pad_bond_feature])
            # max_frag_num = 2, max_atom_num = 3, for 1 pad node.
            JT_atom_neighbor_list = np.zeros([3, self.max_degree])
            JT_atom_neighbor_list.fill(2)
            JT_bond_neighbor_list = np.zeros([3, self.max_degree])
            JT_bond_neighbor_list.fill(1)

            JT_atom_neighbor_list[0, 0] = 1
            JT_atom_neighbor_list[1, 0] = 0
            JT_bond_neighbor_list[0, 0] = 0
            JT_bond_neighbor_list[1, 0] = 0

            JT_atom_neighbor_list = torch.Tensor(JT_atom_neighbor_list).long()
            JT_bond_neighbor_list = torch.Tensor(JT_bond_neighbor_list).long()

            JT_mask = torch.Tensor([1.0, 1.0, 0.0])
        # sizes:
        # mol_atom_features: [max_atom_length, atom_feature_size]
        # mol_bond_features: [max_bond_length, bond_feature_size]
        # mol_atom_neighbor_list: [max_atom_length, max_degree]
        # mol_bond_neighbor_list: [max_atom_length, max_degree]
        # mol_atom_mask: [max_atom_length]
        return JT_bond_feature, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask


class DCGraphFeaturizer(BasicFeaturizer):
    def __init__(self):
        super(DCGraphFeaturizer, self).__init__()
        self.max_atom_num = 0
        self.max_bond_num = 0
        self.atom_feature_size = 75

    def prefeaturize(self, dataset):
        cell_line_list = []
        SMILES_list = []
        entire_atom_features = []
        entire_bond_sparse = []
        conv_featurizer = ConvMolFeaturizer()
        for item in dataset:
            CellLine = item['cell_line']
            SMILES = item['SMILES']
            mol = Chem.MolFromSmiles(SMILES)
            cell_line_list.append(CellLine)

            SMILES_list.append(SMILES)
            convmol = conv_featurizer.featurize(SMILES)
            conv_feature = torch.Tensor(convmol[0].get_atom_features())

            adj_matrix = torch.Tensor(GetAdjMat(mol))
            sparse = GetSparseFromAdj(adj_matrix)[0]
            sparse = np.array(sparse.tolist())

            # mol_atoms_feature = np.zeros([self.max_atom_num, self.atom_feature_size])
            # mol_bond_sparse = np.zeros([2, 2*self.max_bond_num])

            # mol_atoms_feature[:conv_feature.shape[0],:] = conv_feature
            # mol_bond_sparse[:,:sparse.shape[1]] = sparse

            # entire_atom_features.append(mol_atoms_feature)
            # entire_bond_sparse.append(mol_bond_sparse)

            entire_atom_features.append(conv_feature)
            entire_bond_sparse.append(sparse)

        return cell_line_list, entire_atom_features, entire_bond_sparse, SMILES_list

    def featurize(self, dataset, omicsdataset, index, mol, value):
        cell_line_list, entire_atom_features, entire_bond_sparse, SMILES_list = dataset

        mutdataset = omicsdataset[0]
        geneexpdataset = omicsdataset[1]
        methdataset = omicsdataset[2]
        genecndataset = omicsdataset[3]
        cell_line = cell_line_list[index]

        mut_cell_line = torch.Tensor(
            mutdataset.loc[cell_line].values).view(1, -1)

        geneexp_cell_line = torch.Tensor(
            geneexpdataset.loc[cell_line].values).view(-1)
        meth_cell_line = torch.Tensor(
            methdataset.loc[cell_line].values).view(-1)
        genecn_cell_line = torch.Tensor(
            genecndataset.loc[cell_line].values).view(-1)

        mol_atom_features = torch.Tensor(entire_atom_features[index])
        mol_bond_sparse = torch.Tensor(entire_bond_sparse[index]).long()
        SMILES = SMILES_list[index]

        # mol_tensor = [torch.as_tensor(x) for x in conv_feature]
        label = torch.Tensor([value]).unsqueeze(-1)

        # ================================

        return [mut_cell_line, geneexp_cell_line, meth_cell_line, genecn_cell_line, PyGData(x=mol_atom_features, edge_index=mol_bond_sparse)], label

    def CatTensor(self, stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = torch.cat(
            [stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

    def GetPad(self, dataset):
        # dataset format: [{"SMILES": smiles, "Value": value}]
        for item in dataset:
            smiles = item["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            total_atom_num = len(mol.GetAtoms())
            total_bond_num = len(mol.GetBonds())
            self.max_atom_num = max(self.max_atom_num, total_atom_num)
            self.max_bond_num = max(self.max_bond_num, total_bond_num)

        self.pad_atom_idx = self.max_atom_num
        self.pad_bond_idx = self.max_bond_num

        self.max_atom_num += 1
        self.max_bond_num += 1

# ### Checker

# In[8]:


class BasicChecker(object):
    def __init__(self):
        super(BasicChecker, self).__init__()

    def check(self, dataset):
        raise NotImplementedError(
            "Dataset Checker not implemented.")


class ScaffoldSplitterChecker(BasicChecker):
    def __init__(self):
        super(ScaffoldSplitterChecker, self).__init__()

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                checked_dataset.append(item)
            else:
                discarded_dataset.append(item)
        assert len(checked_dataset) + \
            len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")

        return checked_dataset


class AttentiveFPChecker(BasicChecker):
    def __init__(self, max_atom_num, max_degree):
        super(AttentiveFPChecker, self).__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            # check
            if mol:
                # self.check_single_bonds(mol)
                self.check_degree(mol)
                self.check_max_atom_num(mol)

                if self.mol_error_flag == 0:
                    checked_dataset.append(item)
                else:
                    discarded_dataset.append(item)
                    self.mol_error_flag = 0
            else:
                discarded_dataset.append(item)
                self.mol_error_flag = 0
        assert len(checked_dataset) + \
            len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")

        return checked_dataset

    def check_degree(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetDegree() > self.max_degree:
                self.mol_error_flag = 1
                break

    def check_max_atom_num(self, mol):
        if len(mol.GetAtoms()) > self.max_atom_num:
            self.mol_error_flag = 1

    def check_single_bonds(self, mol):
        self.mol_error_flag = 1
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if not bond.IsInRing():
                    self.mol_error_flag = 0
                    break

# ### Dataset

# In[9]:


class DrugOmicsDatasetCreator(data.Dataset):
    def __init__(self):
        super(DrugOmicsDatasetCreator, self).__init__()

    def read_row(self, row):
        SMILES = row.SMILES
        CellLine = row.cell_line
        Value = row.IC50
        return {'SMILES': SMILES, 'cell_line': CellLine, 'IC50': Value}

    def createDataset(self, dataframe):
        dataset = []
        for i in range(len(dataframe)):
            row = dataframe.iloc[i]
            data = self.read_row(row)
            dataset.append(data)
        return dataset


class MolDatasetEval(data.Dataset):
    def __init__(self, dataset, modelname, omicsdata):
        super(MolDatasetEval, self).__init__()
        self.dataset = dataset
        self.Frag = True
        self.FeaturizerList = {
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=72,
                bond_feature_size=10,
                max_degree=5,
                max_frag=2,
                mode='EVAL'
            ),
            'GCN': DCGraphFeaturizer(),
            'GAT': DCGraphFeaturizer()
        }
        self.modelname = modelname
        self.featurizer = self.FeaturizerList[modelname]
        self.omicsdata = omicsdata
        # if use methods in AttentiveFP to construct dataset, some more works should be down here.

        print("Using Attentive FP. Dataset is being checked.")
        self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
        self.dataset = self.checker.check(
            self.dataset)       # screen invalid molecules
        print("Prefeaturizing molecules......")
        self.featurizer.GetPad(self.dataset)
        self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
        print("Prefeaturization complete.")

    def __getitem__(self, index):
        value = self.dataset[index]["IC50"]
        smiles = self.dataset[index]["SMILES"]
        cell_line = self.dataset[index]['cell_line']
        mol = Chem.MolFromSmiles(smiles)
        #print("Single bonds num: ", len(GetSingleBonds(mol)))
        if self.modelname == 'AttentiveFP':
            feature = self.featurizer.featurizenew(
                self.prefeaturized_dataset, self.omicsdata, index, mol, value, self.Frag)
        elif (self.modelname == 'GAT') or (self.modelname == 'GCN'):
            feature = self.featurizer.featurize(
                self.prefeaturized_dataset, self.omicsdata, index, mol, value)
        Input, label = feature[0], feature[1]
        return Input, label
        # print('Drugfeaturized shape 0 = {}, shape 1 = {}, shape 2 = {}'.format(drugdata.shape[0],drugdata.shape[1],drugdata.shape[2]))

    def __len__(self):
        return len(self.dataset)


class MolDatasetTrain(data.Dataset):
    def __init__(self, dataset, omicsdata, modelname):
        super(MolDatasetTrain, self).__init__()
        self.dataset = dataset
        self.Frag = True
        self.FeaturizerList = {
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=72,
                bond_feature_size=10,
                max_degree=5,
                max_frag=2,
                mode='TRAIN'
            ),
            'GCN': DCGraphFeaturizer(),
            'GAT': DCGraphFeaturizer()
        }
        self.modelname = modelname
        self.featurizer = self.FeaturizerList[modelname]
        self.omicsdata = omicsdata
        self.max_atom_num = 102

        # if use methods in AttentiveFP to construct dataset, some more works should be down here.

        print("Using Attentive FP. Dataset is being checked.")
        self.checker = AttentiveFPChecker(
            max_atom_num=self.max_atom_num, max_degree=5)
        self.dataset = self.checker.check(
            self.dataset)       # screen invalid molecules
        print("Prefeaturizing molecules......")
        self.featurizer.GetPad(self.dataset)
        self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
        print("Prefeaturization complete.")

    def __getitem__(self, index):
        value = self.dataset[index]["IC50"]
        smiles = self.dataset[index]["SMILES"]
        cell_line = self.dataset[index]['cell_line']
        mol = Chem.MolFromSmiles(smiles)
        if self.modelname == 'AttentiveFP':
            feature = self.featurizer.featurizenew(
                self.prefeaturized_dataset, self.omicsdata, index, mol, value, self.Frag)
        elif (self.modelname == 'GAT') or (self.modelname == 'GCN'):
            feature = self.featurizer.featurize(
                self.prefeaturized_dataset, self.omicsdata, index, mol, value)
        Input, label = feature[0], feature[1]
        return Input, label

    def __len__(self):
        return len(self.dataset)

# In[11]:


class Saver(object):
    def __init__(self,ckpt_dir, max_epoch):
        super(Saver, self).__init__()
        
        self.ckpt_dir = ckpt_dir
        if self.ckpt_dir[-1] != '/':
            self.ckpt_dir = self.ckpt_dir + '/'
        # if self.results_dir[-1] != '/':
        #     self.results_dir = self.results_dir + '/'
            
        self.ckpt_count = 1
        self.EarlyStopController = EarlyStopController()
        self.maxepoch = max_epoch
        
    def SaveModel(self, model, epoch, scores):
        # state = {'model': model, 'optimizer': optimizer, 'epoch': epoch}
        ckpt_name = os.path.join(self.ckpt_dir, 'epoch-{}'.format(epoch))
        if not os.path.exists(os.path.dirname(ckpt_name)):
            try:
                os.makedirs(os.path.dirname(ckpt_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(model, ckpt_name)

        # result_file_name = self.results_dir + str(epoch) + '.json'
        # if not os.path.exists(os.path.dirname(result_file_name)):
        #     try:
        #         os.makedirs(os.path.dirname(result_file_name))
        #     except OSError as exc: # Guard against race condition
        #         if exc.errno != errno.EEXIST:
        #             raise
                
        # with open(result_file_name, 'w') as f:
        #     json.dump(scores, f, indent=4)
        print("Model saved.")

        ShouldStop = self.EarlyStopController.ShouldStop(scores, self.ckpt_count)

        if ShouldStop:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("Early stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            # delete other models
            self.DeleteUselessCkpt(BestModelCkpt)
            return True, BestModelCkpt, BestValue
        
        elif self.ckpt_count == self.maxepoch:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("The model didn't stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            self.DeleteUselessCkpt(BestModelCkpt)
            return False, BestModelCkpt, BestValue

        else:
            self.ckpt_count += 1
            BestValue, BestModelCkpt= self.EarlyStopController.BestModel()
            return False, BestModelCkpt, BestValue

    def DeleteUselessCkpt(self, BestModelCkpt):
        file_names = os.listdir(self.ckpt_dir)
        for file_name in file_names:
            ckpt_idx = re.split('epoch-',file_name)[-1]
            if int(ckpt_idx) != BestModelCkpt:
                exact_file_path = self.ckpt_dir + file_name
                os.remove(exact_file_path)

    def LoadModel(self, ckpt=None):
        dir_files = os.listdir(self.ckpt_dir)  # list of the checkpoint files
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.ckpt_dir, x)))
            last_model_ckpt = dir_files[-1]   # find the latest checkpoint file.
            model = torch.load(os.path.join(self.ckpt_dir, last_model_ckpt))

            self.ckpt_count = int(re.split('epoch-',last_model_ckpt)[-1]) + 1  # update the ckpt_count, get rid of overwriting the existed checkpoint files.
            return model
        else:
            return None, None, None


class EarlyStopController(object):
    def __init__(self):
        super(EarlyStopController, self).__init__()
        self.MaxResult = 9e8
        self.MaxResultModelIdx = None
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.LowerThanMaxLimit = 5
        self.DecreasingLimit = 3
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx,metricname):
        MainScore = score[metricname]
        if self.MaxResult > MainScore:
            self.MaxResult = MainScore
            self.MaxResultModelIdx = ckpt_idx
            self.LowerThanMaxNum = 0
            self.DecreasingNum = 0
            # all set to 0.
        else:
            # decreasing, start to count.
            self.LowerThanMaxNum += 1
            if MainScore > self.LastResult:
                # decreasing consistently.
                self.DecreasingNum += 1
            else:
                self.DecreasingNum = 0
        self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        # print(self.MaxResultModelIdx)
        # print(self.TestResult)
        return self.MaxResult, self.MaxResultModelIdx

# In[12]:


def Validation(validloader, model, metrics, mode=None):
    model.eval()
    print("Validating..")
    All_answer = []
    All_label = []
    # [tasknum, ]
    no_compound = len(validloader)
    for ii, Data in enumerate(validloader):

        # one molecule input, but batch is not 1. Different Frags of one molecule consist of a batch.
        ValidInput, ValidLabel = Data
        if model.modelname == 'AttentiveFP':
            ValidInput = list(map(lambda x: x.squeeze(0), ValidInput))

        # Label size: [wrongbatch, batch(mol), task, 1]
        # [wrongbatch, batch(mol), task]
        ValidLabel = ValidLabel.squeeze(-1).to(device)
        ValidLabel = ValidLabel.squeeze(0)  # [batch(mol), task]

        ValidOutput = model(ValidInput)
        ValidOutputMean = ValidOutput.mean(
            dim=0, keepdims=True)  # [1, output_size]

        All_answer.append(ValidOutputMean.item())
        All_label.append(ValidLabel[0].item())

    scores = {}

    assert len(All_label) == len(All_answer)
    for metric in metrics:
        result = metric.compute(All_answer, All_label)
        scores.update({metric.name: result})
        print(metric.name, ': ', result)

    torch.cuda.empty_cache()
    model.train()

    if not mode:
        return scores
    elif mode is 'GetAnswerLabel':
        return scores, All_answer, All_label

    # return trainlosslist, testlosslist


def model_check(model, trainsetloader, criterion=nn.MSELoss()):
    model.eval()

    Data = next(iter(trainsetloader))

    loss = 0.0

    Input, Label = Data
    Label = Label.squeeze(-1).to(device)    # [batch, task]

    output = model(Input, mode='check')   # [batch, output_size]

    loss += criterion(output, Label)

    return None


class DNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, use_conv):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LayerList = nn.ModuleList()
        self.use_conv = use_conv
        self.layer_sizes = layer_sizes
        self.Drop = nn.Dropout(p=0.2)

        if not use_conv:
            if len(layer_sizes) == 0:
                self.FC = nn.Linear(input_size, output_size)
            else:
                for i in range(len(layer_sizes)):
                    if i == 0:
                        self.LayerList.append(
                            nn.Linear(input_size, layer_sizes[i]))
                        self.LayerList.append(nn.Tanh())
                    else:
                        self.LayerList.append(
                            nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                        self.LayerList.append(nn.ReLU())
                self.Output = nn.Linear(layer_sizes[-1], output_size)
        else:
            for i in range(len(self.layer_sizes)):
                if i == 0:
                    self.LayerList.extend([
                        nn.Conv1d(1, self.layer_sizes[i], 2),
                        nn.Tanh(),
                        nn.BatchNorm1d(self.layer_sizes[i]),
                        nn.MaxPool1d(3)
                    ])
                elif i == len(self.layer_sizes)-1:
                    self.LayerList.extend([
                        nn.Conv1d(self.layer_sizes[i-1],
                                  self.layer_sizes[i], 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.layer_sizes[i])
                    ])
                else:
                    self.LayerList.extend([
                        nn.Conv1d(self.layer_sizes[i-1],
                                  self.layer_sizes[i], 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.layer_sizes[i]),
                        nn.MaxPool1d(2)
                    ])

    def globalavgpool(self, Input):
        output = nn.AdaptiveAvgPool1d(1)(Input)
        output = nn.Flatten()(output)
        return output

    def globalmaxpool(self, Input):
        output = nn.AdaptiveMaxPool1d(1)(Input)
        output = nn.Flatten()(output)
        return output

    def forward(self, x, mode):
        # print(x.size())
        if mode == 'check':
            print('before cnn: ', x.shape)
        if len(self.layer_sizes) == 0:
            x = self.FC(x)
        else:
            for num, layer in enumerate(self.LayerList):
                x = layer(x)
                if (mode == 'check') & self.use_conv:
                    print('Drug_conv{}:'.format(num), x.size())
                elif (mode == 'check') & (not self.use_conv):
                    print('Drug_fc{}:'.format(num), x.size())
            x = self.Drop(x)
            if self.use_conv:
                x = self.globalavgpool(x)
                return x
            else:
                x = self.Output(x)
                return x

# ### Fragment Model

class MolPredFragFPv8(nn.Module):
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 drop_rate,
                 use_conv,
                 mode='default'
                 ):
        super(MolPredFragFPv8, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size=atom_feature_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=drop_rate
        )   # For Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=drop_rate
        )  # MolEmbedding module can be used repeatedly
        self.Classifier = DNN(
            input_size=4*FP_size,
            output_size=output_size,
            layer_sizes=DNN_layers,
            use_conv=use_conv
        )
        self.AtomEmbeddingHigher = AttentiveFP_atom(
            atom_feature_size=FP_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=drop_rate
        )  # For Junction Tree
        # self.InformationFuser =
        self.mode = mode

    def forward(self, Input, mode):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = self.Input_cuda(Input)

        # layer origin
        atom_FP_origin = self.AtomEmbedding(
            atom_features, bond_features, atom_neighbor_list_origin, bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # layer Frag:
        atom_FP = self.AtomEmbedding(
            atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are used to input the DNN module.
        # activated_mol_FP1 and activated_mol_FP2 are used to calculate the mol_FP
        # size: [batch_size, FP_size]
        ##################################################################################
        # Junction Tree Construction
        # construct a higher level graph: Junction Tree

        # Construct atom features of JT:
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = torch.zeros(batch_size, FP_size).to(device)
        JT_atom_features = torch.stack(
            [activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim=1)

        # Junction Tree Construction complete.
        ##################################################################################
        # layer Junction Tree: calculate information of the junction tree of Frags

        atom_FP_super = self.AtomEmbeddingHigher(JT_atom_features,
                                                 JT_bond_features,
                                                 JT_atom_neighbor_list,
                                                 JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)
        entire_FP = torch.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)

        if self.mode == 'get_entire_FP':
            return entire_FP
        else:
            if self.Classifier.use_conv:
                entire_FP = torch.unsqueeze(entire_FP, 1)
            prediction = self.Classifier(entire_FP, mode)
            return prediction

    def Input_cuda(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input

        atom_features = atom_features.to(device)
        bond_features = bond_features.to(device)
        atom_neighbor_list_origin = atom_neighbor_list_origin.to(device)
        bond_neighbor_list_origin = bond_neighbor_list_origin.to(device)
        atom_mask_origin = atom_mask_origin.to(device)

        atom_neighbor_list_changed = atom_neighbor_list_changed.to(device)
        bond_neighbor_list_changed = bond_neighbor_list_changed.to(device)
        frag_mask1 = frag_mask1.to(device)
        frag_mask2 = frag_mask2.to(device)
        bond_index = bond_index.to(device)

        JT_bond_features = JT_bond_features.to(device)
        JT_atom_neighbor_list = JT_atom_neighbor_list.to(device)
        JT_bond_neighbor_list = JT_bond_neighbor_list.to(device)
        JT_mask = JT_mask.to(device)

        return [atom_features,
                bond_features,
                atom_neighbor_list_origin,
                bond_neighbor_list_origin,
                atom_mask_origin,
                atom_neighbor_list_changed,
                bond_neighbor_list_changed,
                frag_mask1,
                frag_mask2,
                bond_index,
                JT_bond_features,
                JT_atom_neighbor_list,
                JT_bond_neighbor_list,
                JT_mask]


class GCNNet(torch.nn.Module):
    def __init__(self, n_output=100, n_filters=32, embed_dim=128, num_features_xd=75, num_features_xt=25, output_dim=128, dropout=0.5):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(
            in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(
            in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(
            in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        # target = data.target
        # target = target[:,None,:]

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = PyGGMaxPool(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        return x


class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=75, n_output=1, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd,
                            heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(
            in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(
            in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(
            in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = PyGGMaxPool(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x


class MODPredIC50(nn.Module):
    def __init__(self,
                 omicsdata,
                 modelname,
                 FP_size,         # 'FP_size': 150
                 DNN_layers,   # 'DNNLayers':[512]
                 drug_output_size,
                 omics_output_size,
                 drop_rate,
                 use_mut,
                 use_gexp,
                 use_meth,
                 use_genecn,
                 atom_feature_size=72,  # 'atom_feature_size': 39
                 bond_feature_size=10,  # 'bond_feature_size': 10
                 atom_layers=3,     # 'atom_layers':3
                 mol_layers=2,      # 'mol_layers':2
                 use_conv=True,
                 classify=False,
                 ):
        super(MODPredIC50, self).__init__()
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_meth = use_meth
        self.use_genecn = use_genecn
        self.use_conv = use_conv
        self.omicsdata = omicsdata
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.FP_size = FP_size
        self.atom_layers = atom_layers
        self.mol_layers = mol_layers
        self.DNN_layers = DNN_layers
        self.omics_output_size = omics_output_size
        self.drug_output_size = drug_output_size
        self.drop_rate = drop_rate
        self.classify = classify
        self.modelname = modelname
        self.mut_nn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1, 50, 700, stride=5)),
            ('actvfunc1', nn.Tanh()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('conv2', nn.Conv1d(50, 30, 5, stride=2)),
            ('actvfunc2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(10)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(2700, self.omics_output_size)),  # DeepCDR = 2010, 2700
            ('actvfunc3', nn.ReLU()),
            ('drop', nn.Dropout(drop_rate))
        ]))
        self.gexp_nn = nn.Sequential(
            nn.Linear(omicsdata[1].shape[1], 256),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.omics_output_size),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        self.meth_nn = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(omicsdata[2].shape[1], 256)),
            ('actvfunc1', nn.Tanh()),
            ('bn1', nn.BatchNorm1d(256)),
            ('drop1', nn.Dropout(p=0.1)),
            ('fc2', nn.Linear(256, self.omics_output_size)),
            ('actvfunc2', nn.ReLU()),
            ('drop2', nn.Dropout(drop_rate))
        ]))
        self.genecn_nn = nn.Sequential(
            nn.Linear(omicsdata[3].shape[1], 256),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, self.omics_output_size),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        drugmodel_list = {
            'GCN': GCNNet(
                num_features_xd=75,
                output_dim=drug_output_size
            ),
            'GAT': GATNet(
                num_features_xd=75,
                output_dim=drug_output_size
            ),
            'AttentiveFP': MolPredFragFPv8(
                atom_feature_size=atom_feature_size,  # 'atom_feature_size': 39
                bond_feature_size=bond_feature_size,  # 'bond_feature_size': 10
                FP_size=FP_size,         # 'FP_size': 150
                atom_layers=atom_layers,     # 'atom_layers':3
                mol_layers=mol_layers,      # 'mol_layers':2
                DNN_layers=DNN_layers,   # 'DNNLayers':[512]
                output_size=drug_output_size,
                drop_rate=drop_rate,    # 'drop_rate':0.2
                use_conv=use_conv,
            )          # drugmodel = molpredfragfpv8
        }
        self.drug_nn = drugmodel_list[modelname]
        if modelname == 'AttentiveFP':
            if self.use_conv:
                layer_before_size = (use_mut + use_gexp + use_meth + use_genecn) * \
                    omics_output_size + DNN_layers[-1]
            else:
                layer_before_size = (use_mut + use_gexp + use_meth + use_genecn) * \
                    omics_output_size + self.drug_output_size        
        else:
            layer_before_size = (use_mut + use_gexp + use_meth + use_genecn)*omics_output_size + self.drug_output_size
        layer_after_conv1 = (800-150) // +1
        layer_after_maxpool1 = layer_after_conv1//2
        layer_after_conv2 = (layer_after_maxpool1-5)//1 + 1
        layer_after_maxpool2 = layer_after_conv2//3
        layer_after_conv3 = (layer_after_maxpool2-5)//1 + 1
        layer_after_maxpool3 = layer_after_conv3//3
        final_size = 32*layer_after_maxpool3

        self.drugomics_fc = nn.Sequential(
            nn.Linear(layer_before_size, 800),
            nn.ReLU()
        )
        self.drugomics_conv = nn.Sequential(
            nn.Conv1d(1, 64, 150, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, 5, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 32, 5, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(3),
            # nn.Dropout(p=0.1),
            nn.Flatten(),
            # nn.ReLU(),
            # nn.Dropout(p=drop_rate)
        )
        # self.maccs_nn = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv1d(1, 20, 5, stride=1)),
        #     ('actvfunc1', nn.ReLU()),
        #     ('pool1', nn.MaxPool1d(3)),
        #     ('conv2', nn.Conv1d(20, 10, 5, stride=1)),
        #     ('actvfunc2', nn.ReLU()),
        #     ('pool2', nn.MaxPool1d(2)),
        #     ('flatten', nn.Flatten()),
        #     ('fc', nn.Linear(250, 100)),
        #     # ('drop',nn.Dropout(0.1))
        # ]))
        # self.pubchem_nn = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv1d(1, 20, 20)),
        #     ('actvfunc1', nn.ReLU()),
        #     ('pool1', nn.MaxPool1d(3)),
        #     ('conv2', nn.Conv1d(20, 10, 5)),
        #     ('actvfunc2', nn.ReLU()),
        #     ('pool2', nn.MaxPool1d(2)),
        #     ('flatten', nn.Flatten()),
        #     ('fc', nn.Linear(1410, 300)),
        #     ('actvfunc3', nn.ReLU()),
        #     ('fc2', nn.Linear(300, 100)),
        #     # ('drop',nn.Dropout(0.1))
        # ]))
        self.out = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(final_size, 1024)),
            ('relu', nn.ReLU()),
            # ('drop',nn.Dropout(0.1)),
            ('fc2', nn.Linear(1024, 128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, 1)),
        ]))

    def forward(self, Input, mode='default'):
        MultiOmics_layer = torch.Tensor().to(device)
        Mut_layer = Input[0].to(device)
        GeneExp_layer = Input[1].to(device)
        Meth_layer = Input[2].to(device)
        GeneCN_layer = Input[3].to(device)

        if self.use_mut:
            Mut_layer = self.mut_nn(Mut_layer)
            MultiOmics_layer = torch.cat((MultiOmics_layer, Mut_layer), 1)

        if self.use_gexp:
            GeneExp_layer = self.gexp_nn(GeneExp_layer)
            MultiOmics_layer = torch.cat((MultiOmics_layer, GeneExp_layer), 1)

        if self.use_meth:
            Meth_layer = self.meth_nn(Meth_layer)
            MultiOmics_layer = torch.cat((MultiOmics_layer, Meth_layer), 1)

        if self.use_genecn:
            GeneCN_layer = self.genecn_nn(GeneCN_layer)
            MultiOmics_layer = torch.cat((MultiOmics_layer, GeneCN_layer), 1)

        if self.modelname == 'AttentiveFP':
            Drug_layer = self.drug_nn(Input[4:18], mode)
            # maccs = self.maccs_nn(torch.Tensor(Input[18]).to(device))
            # pubchem = self.pubchem_nn(torch.Tensor(Input[19]).to(device))
            OmicsDrug_layer = torch.cat(
                (MultiOmics_layer, Drug_layer, 
                # maccs, pubchem
                ), 1)
        else:
            druggraph = Input[4].to(device)
            Drug_layer = self.drug_nn(druggraph)
            OmicsDrug_layer = torch.cat((MultiOmics_layer, Drug_layer), 1)

        OmicsDrug_layer = self.drugomics_fc(OmicsDrug_layer)
        OmicsDrug_layer = OmicsDrug_layer.unsqueeze(1)
        OmicsDrug_layer = self.drugomics_conv(OmicsDrug_layer)
        prediction = self.out(OmicsDrug_layer)

        if self.classify :
            prediction = torch.sigmoid(prediction)

        return prediction