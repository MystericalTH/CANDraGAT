import pandas as pd
import numpy as np
from datetime import datetime, date
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import torch
from torch import optim, nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.optim.lr_scheduler import *
# from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as PyGGMaxPool
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from typing import Sequence, List, Dict, Optional, OrderedDict, Generator
import csv
import os
import shutil
import argparse
import arrow
import optuna
import warnings
import random
import json
import re
import errno
from itertools import product
from deepchem.feat import ConvMolFeaturizer
import pickle as pkl
from Metrics import *

cuda = torch.device('cuda')
cpu = torch.device('cpu')
device = cuda

# # FraGAT Code
# ### ChemUtils

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)
warnings.filterwarnings('ignore')

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


    # X = pd.DataFrame(columns=pd.MultiIndex.from_product(([x.name for x in report_metrics],['mean','interval'])))
    # X.index.name='experiment_time'
    # X.insert(0,'mutation',np.nan)
    # X.insert(1,'gene_expression',np.nan)
    # X.insert(2,'methylation',np.nan)
    # X.insert(3,'copy_number_variation', np.nan)
    # X.insert(0,'experiment_name',np.nan)
    # X.insert(1,'start_time',np.nan)
    # X.insert(2,'end_time',np.nan)
    # X.insert(3,'elapsed_time',np.nan)
    # X.to_csv(summaryfile,index=False)

def GetSparseFromAdj(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]
    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    return torch.stack(index, dim=0), edge_attr




class BasicCriterion(object):

    def __init__(self):
        super().__init__()
        self.name = None

    def compute(self, answer, label):
        raise NotImplementedError
    
    def __len__(self):
        return 1

class RMSE(BasicCriterion):
    def __init__(self):
        super().__init__()
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


class MAE(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MAE = F.l1_loss(answer, label, reduction='mean')
        return MAE.item()


class MSE(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'MSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = torch.Tensor(answer)
        label = torch.Tensor(label)
        #print("Size for MAE")
        MSE = F.mse_loss(answer, label, reduction='mean')
        return MSE.item()


class PCC(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'PCC'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        pcc = np.corrcoef(answer, label)
        return pcc[0][1]


class R2(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'R2'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = np.array(answer)
        label = np.array(label)
        #print("Size for MAE")
        r_squared = r2_score(answer, label)
        return r_squared


class SRCC(BasicCriterion):
    def __init__(self):
        super().__init__()
        self.name = 'SpearmanR'

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

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super().__init__()
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
        super().__init__()
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
        attend_mask = atom_neighbor_list.clone() #.to(device)
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        mixture = atom_neighbor_FP + bond_neighbor_FP - atom_neighbor_FP * bond_neighbor_FP

        # concate bond_neighbor_FP and atom_neighbor_FP and mixture item, and then transform it from
        # [batch, atom, neighbor, 3*FP_size] to [batch, atom, neighbor, FP_size]
        neighbor_FP = torch.cat([atom_neighbor_FP, bond_neighbor_FP, mixture], dim=-1)
        # print(neighbor_FP.size())
        neighbor_FP = self.nei_fc(neighbor_FP)
        #neighbor_FP = F.leaky_relu(neighbor_FP)

        # transform atom_features from [batch, atom, atom_feature_size] to [batch, atom, FP_size]
        #atom_FP = self.atom_fc(atom_features)
        #atom_FP = F.leaky_relu(atom_FP)

        return atom_FP, neighbor_FP
###########################################################################################################






# ### Checker

class BasicChecker(object):
    def __init__(self):
        super().__init__()

    def check(self, dataset):
        raise NotImplementedError(
            "Dataset Checker not implemented.")


class ScaffoldSplitterChecker(BasicChecker):
    def __init__(self):
        super().__init__()

    def check(self, smiles_list):
        origin_dataset = smiles_list
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                checked_dataset.append(item)
            else:
                discarded_dataset.append(item)
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")

        return checked_dataset


class AttentiveFPChecker(BasicChecker):
    def __init__(self, max_atom_num, max_degree,log=True):
        super().__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0
        self.log = log

    def check(self, id2smiles:dict):
        origin_dataset = id2smiles
        checked_dataset = {}
        discarded_dataset = {}
        for index in origin_dataset:
            smiles = id2smiles[index]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                self.check_degree(mol)
                self.check_max_atom_num(mol)

                if self.mol_error_flag == 0:
                    checked_dataset[index] = (smiles)
                else:
                    discarded_dataset[index] = (smiles)
                    self.mol_error_flag = 0
            else:
                discarded_dataset[index] = (smiles)
                self.mol_error_flag = 0
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)

        if self.log:
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

class OmicsMolDatasetCreator(data.Dataset):
    def __init__(self):
        super().__init__()

    def createDataset(self, dataframe):
        smiles_list = dataframe['SMILES'].unique().tolist()
        cl_list = dataframe['cell_line'].unique().tolist()

        id2smiles = dict(zip(list(range(len(smiles_list))),smiles_list))
        id2cl = dict(zip(list(range(len(cl_list))),cl_list))

        smiles2id = {id2smiles[k]:k for k in id2smiles}
        cl2id =  {id2cl[k]:k for k in id2cl}
        encoded_dataframe = dataframe.copy()
        encoded_dataframe['SMILES'] = encoded_dataframe['SMILES'].map(smiles2id)
        encoded_dataframe['cell_line'] = encoded_dataframe['cell_line'].map(cl2id)

        # dataset = []
        # for i in range(len(dataframe)):
        #     row = dataframe.iloc[i]
        #     data = self.read_row(row)
        #     dataset.append(data)
        # return dataset
        assert type(encoded_dataframe) == pd.DataFrame
        return encoded_dataframe, id2smiles, id2cl


class OmicsMolDataset(data.Dataset):
    def __init__(self, encoded_dataframe: pd.DataFrame, omicsdata: Sequence[pd.DataFrame], modelname: str, id2smiles: dict, id2cl: dict, mode: str = None, log = False):
        super().__init__()
        self.encoded_dataframe = encoded_dataframe
        self.Frag = True
        self.id2smiles = id2smiles
        self.id2cl = id2cl
        self.MolFeaturizerList = {
            'FragAttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=72,
                bond_feature_size=10,
                max_degree=5,
                max_frag=2,
                mode=mode
            ),
            'GCN': DCGraphFeaturizer(),
            'GAT': DCGraphFeaturizer()
        }
        self.modelname = modelname
        self.omics_featurizer = OmicsFeaturizer(modelname=modelname, mode=mode, log=log)
        self.label_featurizer = LabelFeaturizer(modelname=modelname, mode=mode)
        self.mol_featurizer = self.MolFeaturizerList[modelname]
        self.omicsdata = omicsdata
        self.max_atom_num = 102
        self.log = log
        # if use methods in AttentiveFP to construct dataset, some more works should be down here.

        if self.log:("Using Attentive FP. Dataset is being checked.")
        self.checker = AttentiveFPChecker(
            max_atom_num=self.max_atom_num, max_degree=5,log=log)
        self.id2smiles = self.checker.check(
            self.id2smiles)       # screen invalid molecules
        if self.log:
            print("Prefeaturizing molecules......")
        if self.modelname in ('AttentiveFP','FragAttentiveFP'):
            self.mol_featurizer.GetPad(self.id2smiles)

        self.prefeaturized_mol_dataset = self.mol_featurizer.prefeaturize(self.id2smiles)
        self.prefeaturized_omics_dataset = self.omics_featurizer.prefeaturize(self.id2cl,self.omicsdata)
        self.prefeaturized_label_dataset = self.label_featurizer.prefeaturize(self.encoded_dataframe)

        if self.log:
            print("Prefeaturization complete.")

    def __getitem__(self, index):
        
        id_smiles = self.encoded_dataframe["SMILES"].iloc[index]
        id_cl = self.encoded_dataframe['cell_line'].iloc[index]
        smiles = self.id2smiles[id_smiles]
        mol = Chem.MolFromSmiles(smiles)

        # mol = Chem.MolFromSmiles(smiles)
        if self.modelname in ('AttentiveFP','FragAttentiveFP'):
            drug_feature = self.mol_featurizer.featurizenew(self.prefeaturized_mol_dataset, id_smiles, mol, self.Frag)
        elif self.modelname in ('GAT','GCN'):
            drug_feature = self.mol_featurizer.featurize(self.prefeaturized_mol_dataset, id_smiles)
        
        omics_feature = self.omics_featurizer.featurize(self.prefeaturized_omics_dataset,id_cl,mol)
        label = self.label_featurizer.featurize(self.prefeaturized_label_dataset,index,mol)

        return omics_feature, drug_feature, label

    def __len__(self):
        return len(self.encoded_dataframe)




def Validation(validloader, model, metrics):
    model.eval()
    print("Validating..")
    All_answer = []
    All_label = []
    # [tasknum, ]
    for ii, Data in enumerate(validloader):

        ValidOmicsInput, ValidDrugInput, ValidLabel = Data
        if model.drug_nn.name in ('AttentiveFP','FragAttentiveFP'):
            # ValidOmicsInput = list(map(lambda x: x.squeeze(0), ValidOmicsInput))
            # ValidDrugInput = list(map(lambda x: x.squeeze(0), ValidDrugInput))
            ValidOmicsInput = [x.squeeze(0) for x in ValidOmicsInput]
            ValidDrugInput = [x.squeeze(0) for x in ValidDrugInput]

        ValidLabel = ValidLabel.squeeze(-1).to(device)
        ValidLabel = ValidLabel.squeeze(0)  # [batch(mol), task]

        ValidOutput = model(ValidOmicsInput,ValidDrugInput)
        ValidOutputMean = ValidOutput.mean(
            dim=0, keepdims=True)  # [1, output_size]

        All_answer.append(ValidOutputMean.item())
        All_label.append(ValidLabel[0].item())

    scores = {}

    assert len(All_label) == len(All_answer)
    if len(metrics) != 1:
        for metric in metrics:
            result = metric.compute(All_answer, All_label)
            scores.update({metric.name: result})
            print(metric.name, ': ', result)
    elif len(metrics) == 1:
        result = metrics.compute(All_answer, All_label)
        scores.update({metrics.name: result})
        print(metrics.name, ': ', result)

    torch.cuda.empty_cache()
    model.train()

    return scores, All_answer, All_label

    # return trainlosslist, testlosslist

class DNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LayerList = nn.ModuleList()

        self.layer_sizes = layer_sizes
        self.Drop = nn.Dropout(p=0.2)

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

    def forward(self, x):
        if len(self.layer_sizes) == 0:
            x = self.FC(x)
        else:
            for num, layer in enumerate(self.LayerList):
                x = layer(x)
            x = self.Drop(x)
            x = self.Output(x)
        return x

# ### Fragment Model

class MolPredFragFPv8(BasicDrugModule):
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 drop_rate,
                 ):
        super().__init__()
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
        )
        self.AtomEmbeddingHigher = AttentiveFP_atom(
            atom_feature_size=FP_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=drop_rate
        )  # For Junction Tree
        # self.InformationFuser =
        self._name = 'FragAttentiveFP'

    def forward(self, Input):
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
         JT_mask] = self.InputCUDA(Input)

        # layer origin
        atom_FP_origin = self.AtomEmbedding(atom_features,
                                            bond_features, 
                                            atom_neighbor_list_origin, 
                                            bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # layer Frag:
        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
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
        JT_atom_features = torch.stack([activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim=1)

        # Junction Tree Construction complete.
        ##################################################################################
        # layer Junction Tree: calculate information of the junction tree of Frags

        atom_FP_super = self.AtomEmbeddingHigher(JT_atom_features,
                                                 JT_bond_features,
                                                 JT_atom_neighbor_list,
                                                 JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)
        entire_FP = torch.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)

        prediction = self.Classifier(entire_FP)
        return prediction

class GCNNet(BasicDrugModule):
    
    def __init__(self, n_output=100, n_filters=32, embed_dim=128, num_features_xd=75, num_features_xt=25, output_dim=128, dropout=0.5):

        super().__init__()
        self._name = 'GCN'
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
        x, edge_index, batch = self.InputCUDA([data.x, data.edge_index, data.batch])

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

class GATNet(BasicDrugModule):
    def __init__(self, num_features_xd=75, n_output=1, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super().__init__()
        self._name = 'GAT'
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
        x, edge_index, batch = self.InputCUDA([data.x, data.edge_index, data.batch])

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = PyGGMaxPool(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        return x




class MutNet(nn.Module):
    def __init__(self,drop_rate):
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(1, 50, 700, stride=5)
        # self.actvfunc1 = nn.Tanh()),
        self.maxpool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(50, 30, 5, stride=2)
        # self.actvfunc2 = nn.ReLU()),
        self.maxpool2 = nn.MaxPool1d(10)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2700, 100) # DeepCDR = 2010, 2700
        # self.actvfunc3 = nn.ReLU()),
        self.drop = nn.Dropout(drop_rate)

    def forward(self, Input):
        x = self.conv1(Input)
        x = self.tanh(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.drop(x)
        return x

class ExprNet(nn.Module):
    def __init__(self,input_size,drop_rate):
        super().__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 100)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self,Input):
        x = self.fc1(Input)
        x = self.tanh(x)
        x = self.bn(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)
        return x

class CNVNet(nn.Module):
    def __init__(self,input_size,drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(256, 100)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self,Input):
        x = self.fc1(Input)
        x = self.tanh(x)
        x = self.bn(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)
        return x

class MethNet(nn.Module):
    def __init__(self,input_size,drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256) #input_size = omicsdata[2].shape[1]
        self.actvfunc1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(256, 100)
        self.actvfunc2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop_rate)
    def forward(self,Input):
        x = self.fc1(Input)
        x = self.actvfunc1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.actvfunc2(x)
        x = self.drop2(x)
        return x

class MultiOmicsMolNet(nn.Module):

    def __init__(self,
                 drug_nn,
                 drug_output_size,
                 omics_output_size,
                 drop_rate,
                 classify=False,
                 ):
        super().__init__()
        
        self.omics_output_size = omics_output_size
        self.drug_output_size = drug_output_size
        self.classify = classify
        self.mut_nn = MutNet(drop_rate=drop_rate)
        self.expr_nn = ExprNet(input_size=MultiOmicsMolNet.expr_input_shape,drop_rate=drop_rate)
        self.meth_nn = MethNet(input_size=MultiOmicsMolNet.meth_input_shape, drop_rate=drop_rate)
        self.cnv_nn = CNVNet(input_size=MultiOmicsMolNet.cnv_input_shape, drop_rate=drop_rate)
        self.drug_nn = drug_nn

        self.omics_fc = nn.Sequential(
            nn.Linear(400, self.omics_output_size),
            nn.ReLU()
        )
        
        print('omics_output_size',self.omics_output_size)
        print('drug_output_size',self.drug_output_size)

        self.drugomics_fc = nn.Sequential(
            nn.Linear(self.omics_output_size+self.drug_output_size, self.omics_output_size+self.drug_output_size),
            nn.LeakyReLU()
        )

        self.drugomics_conv = nn.Sequential(
            nn.Conv1d(1, 64, 150, stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, 5, stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 32, 5, stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
        )
        
        self.out = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.calculate_cnn_shape(), 128)),
            ('relu', nn.ELU()),
            ('fc2', nn.Linear(128, 1)),
            # ('relu2', nn.ReLU()),
            # ('fc3', nn.Linear(128, 1)),
        ]))

    def forward(self, OmiscInput, DrugInput):
        Mut,Expr,Meth,CNV = self.InputCUDA(OmiscInput)

        Mut_layer = self.mut_nn(Mut)
        Expr_layer = self.expr_nn(Expr)
        Meth_layer = self.meth_nn(Meth)
        CNV_layer = self.cnv_nn(CNV)
        Drug_layer = self.drug_nn(DrugInput)
        Omics_layer = torch.cat((Mut_layer, Expr_layer,Meth_layer,CNV_layer), dim=1)
        Omics_layer = self.omics_fc(Omics_layer)
        OmicsDrug_layer = torch.cat((Omics_layer,Drug_layer), dim=1)
        OmicsDrug_layer = self.drugomics_fc(OmicsDrug_layer)
        OmicsDrug_layer = OmicsDrug_layer.unsqueeze(1)
        OmicsDrug_layer = self.drugomics_conv(OmicsDrug_layer)
        prediction = self.out(OmicsDrug_layer)

        if self.classify :
            prediction = torch.sigmoid(prediction)

        return prediction

    @classmethod
    def set_omics_input_shape(cls,list_omics_df):
        cls.expr_input_shape = list_omics_df[1].shape[1]
        cls.meth_input_shape = list_omics_df[2].shape[1]
        cls.cnv_input_shape = list_omics_df[3].shape[1]

    @staticmethod
    def InputCUDA(listTensor:Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [d.to(device) for d in listTensor]

    def calculate_cnn_shape(self):
        layer_after_conv1 = (self.omics_output_size+self.drug_output_size-150)//1 +1
        layer_after_maxpool1 = layer_after_conv1//2
        layer_after_conv2 = (layer_after_maxpool1-5)//1 + 1
        layer_after_maxpool2 = layer_after_conv2//3
        layer_after_conv3 = (layer_after_maxpool2-5)//1 + 1
        layer_after_maxpool3 = layer_after_conv3//3
        final_size = 32*layer_after_maxpool3
        return final_size

# ----- Hyperparameter ------

def get_best_trial(hyper_dir):
    with open(f'{hyper_dir}.json','r') as jsonfile:
        best_trial_param = json.load(jsonfile)
    return best_trial_param

def run_hyper_study(study_func, n_trials, hyperexpfilename,study_name=None):
    study = optuna.create_study(direction="minimize", study_name=study_name, load_if_exists=True)
    study.optimize(study_func, n_trials=n_trials, gc_after_trial=True)
    # df = study.trials_dataframe()
    trial = study.best_trial
    best_trial_param = dict()
    for key, value in trial.params.items():
        best_trial_param[key]=value
    with open(f'{hyperexpfilename}.json','w') as jsonfile:
        json.dump(best_trial_param,jsonfile,indent=4)
    return study

def get_trial_config(trial):
    return {
        # 'batchsize': 2**(trial.suggest_int('batchsize', 2,5)),
        'drop_rate': trial.suggest_float('drop_rate',0.1,0.9,step=0.05),
        'lr': trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        'WeightDecay': trial.suggest_uniform('WeightDecay',1e-7, 1e-5),
        'omics_output_size': trial.suggest_int('omics_output_size',100,300,step=25),
        'drug_output_size': trial.suggest_int('drug_output_size',100,300,step=25)
    }

def candragat_tuning(trial, drugsens_tv, omics_dataset, cl_list, smiles_list, max_tuning_epoch: int = 5):

    criterion = nn.MSELoss()
    pt_param = get_trial_config(trial)
    drug_model = get_drug_model(modelname,pt_param)
    validmseloss = []

    for Trainset, Validset in df_kfold_split(TVset):

        model = MultiOmicsMolNet(
                    drug_model,
                    drug_output_size=pt_param['drug_output_size'],
                    omics_output_size=pt_param['omics_output_size'],
                    drop_rate=pt_param['drop_rate'],
                    classify=classify
        ).to(device)

        model.train()
        model.zero_grad()
        optimizer = optim.Adam(model.parameters(),lr=pt_param['lr'],weight_decay=pt_param['WeightDecay'])
        optimizer.zero_grad()

        DatasetTrain = OmicsMolDataset(Trainset, omicsdata=OmicsData, modelname=modelname,id2smiles=id2smiles_tv,id2cl=id2cl_tv,mode='TRAIN')
        DatasetValid = OmicsMolDataset(Validset, omicsdata=OmicsData, modelname=modelname,id2smiles=id2smiles_tv,id2cl=id2cl_tv,mode='EVAL')

        if modelname in ("AttentiveFP","FragAttentiveFP"):
            trainloader = data.DataLoader(DatasetTrain, batch_size=256, num_workers=2,
                                        drop_last=True, worker_init_fn=np.random.seed(), pin_memory=True)
            validloader = data.DataLoader(DatasetValid, batch_size=1, num_workers=0,
                                        drop_last=True, worker_init_fn=np.random.seed(), pin_memory=True)

        elif modelname in ("GAT", "GCN"):
            trainloader = PyGDataLoader(DatasetTrain, batch_size=256, num_workers=2, drop_last=True,
                                    worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)
            validloader = PyGDataLoader(DatasetValid, batch_size=1, num_workers=0, drop_last=True,
                                    worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)

        for epoch in range(max_tuning_epoch):

            cum_loss = 0.0
            printloss = 0.0

            for Data in trainloader:
                
                loss = 0.0

                OmicsInput, DrugInput, Label = Data
                Label = Label.squeeze(-1).to(device)    # [batch, task]
                #Label = Label.t()            # [task, batch]
                output = model(OmicsInput,DrugInput)   # [batch, output_size]

                loss += criterion(output,Label).float()

                cum_loss += loss.detach()
                printloss += loss.detach()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # trainmseloss = (cum_loss/len(trainloader)).item()
        metric = BCE() if model.classify else MSE()
        results = Validation(validloader, model, metric)[0]
        validmseloss.append(results[metric.name])
    
    status.update({'trial': status['trial'] + 1})
    return np.mean(validmseloss)

