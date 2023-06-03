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

# –––– Utils ––––

def set_base_seed(seed=None):        
    random.seed(seed)

def set_seed(seed=None):
    if seed is None:
        seed = random.randrange(1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed

def df_kfold_split(df, n_splits=5, seed: int = None) -> Generator[pd.DataFrame, pd.DataFrame, None]:
    assert type(df) == pd.DataFrame
    kfold = KFold(n_splits=n_splits,random_state=seed)
    list_idx = list(range(len(df)))
    for train_idx, test_idx in kfold.split(list_idx):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        print('len df train, test: ',len(df_train),len(df_test))
        yield df_train, df_test
    
def df_train_test_split(df: pd.DataFrame, train_size=0.8) -> Sequence[pd.DataFrame]:
    list_idx = list(range(len(df)))
    train_idx, test_idx = train_test_split(list_idx,train_size=train_size)
    return df.iloc[train_idx], df.iloc[test_idx]

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def write_result_files(report_metrics,summaryfile):
    col = ['experiment_name','start_time','end_time','elapsed_time','mutation', 'gene_expression', 'methylation', 'copy_number_variation']
    writer = csv.writer(summaryfile)
    multiheader = [(x,None) for x in col] + list(product([x.name for x in report_metrics],['mean','interval']))
    for i in range(2):
        writer.writerow([n[i] for n in multiheader])
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


def graph_collate_fn(batch):
    Output = [torch.Tensor()]*len(batch[0])
    for Input in batch:
        for i, x in enumerate(Input):
            Output[i] = torch.cat((Output[i], x))
    return Output

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

        return atom_FP, 
        
###########################################################################################################


class AttentiveFPLayer(nn.Module):
    def __init__(self, FP_size, droprate):
        super().__init__()
        self.FP_size = FP_size
        self.attentioncalculator = AttentionCalculator(self.FP_size, droprate)
        self.contextcalculator = ContextCalculator(self.FP_size, droprate)
        self.FPtranser = FPTranser(self.FP_size)

    def forward(self, atom_FP, neighbor_FP, atom_neighbor_list):
        # align atom FP and its neighbors' FP to generate [hv, hu]
        FP_align = self.feature_align(atom_FP, neighbor_FP)
        # FP_align: [batch_size, max_atom_length, max_neighbor_length, 2*FP_size]

        # calculate attention score evu
        attention_score = self.attentioncalculator(FP_align, atom_neighbor_list)
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
        super().__init__()
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
        super().__init__()
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

            super_node_align = torch.cat([super_node_FP_expand, atom_FP], dim=-1)
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

class BasicFeaturizer(object):
    def __init__(self):
        super().__init__()

    def featurize(self, *args):
        raise NotImplementedError(
            "Molecule Featurizer not implemented.")

class OmicsFeaturizer(BasicFeaturizer):
    def __init__(self, modelname, mode=None,log=True):
        super().__init__()
        self.modelname = modelname
        self.mode = mode
        self.log = log

    def prefeaturize(self, id2cl: dict, omicsdata: Sequence[pd.DataFrame]):
        mutdataset, exprdataset, methdataset, cnvdataset = omicsdata
        id2mut = self.create_id2data(id2cl,mutdataset)
        id2expr = self.create_id2data(id2cl,exprdataset)
        id2meth = self.create_id2data(id2cl,methdataset)
        id2cnv = self.create_id2data(id2cl,cnvdataset)
        assert len(id2mut) == len(id2expr) == len(id2meth) == len(id2cnv)
        if self.log:
            print(f'Loading {len(id2mut)} cell lines')
        return [id2mut, id2expr, id2meth, id2cnv]



    def featurize(self, dataset,index, mol) -> List[torch.Tensor]: 
        [id2mut, id2expr, id2meth, id2cnv] = dataset
        
        mut_cell_line = torch.Tensor(id2mut[index]).view(1, -1)
        expr_cell_line = torch.Tensor(id2expr[index]).view(-1) 
        meth_cell_line = torch.Tensor(id2meth[index]).view(-1) 
        cnv_cell_line = torch.Tensor(id2cnv[index]).view(-1)

        if self.modelname in ('AttentiveFP', 'FragAttentiveFP'):

            if self.mode == 'TRAIN':
                return [mut_cell_line, expr_cell_line, meth_cell_line, cnv_cell_line]
                
            elif self.mode == 'EVAL':
                extended_mut_cell_line = torch.Tensor([])
                extended_expr_cell_line = torch.Tensor([])
                extended_meth_cell_line = torch.Tensor([])
                extended_cnv_cell_line = torch.Tensor([])
                SingleBondList = GetSingleBonds(mol)
                if len(SingleBondList) == 0:
                    extended_mut_cell_line = self.CatTensor(extended_mut_cell_line, mut_cell_line)
                    extended_expr_cell_line = self.CatTensor(extended_expr_cell_line, expr_cell_line)
                    extended_meth_cell_line = self.CatTensor(extended_meth_cell_line, meth_cell_line)
                    extended_cnv_cell_line = self.CatTensor(extended_cnv_cell_line, cnv_cell_line)
                else:
                    for bond in SingleBondList:
                        extended_mut_cell_line = self.CatTensor(extended_mut_cell_line, mut_cell_line)
                        extended_expr_cell_line = self.CatTensor(extended_expr_cell_line, expr_cell_line)
                        extended_meth_cell_line = self.CatTensor(extended_meth_cell_line, meth_cell_line)
                        extended_cnv_cell_line = self.CatTensor(extended_cnv_cell_line, cnv_cell_line)
                
                return [extended_mut_cell_line, extended_expr_cell_line, extended_meth_cell_line, extended_cnv_cell_line]
            else:
                raise RuntimeError('Configs unavailable')

        elif self.modelname in ('GAT','GCN'):
            return [mut_cell_line, expr_cell_line, meth_cell_line, cnv_cell_line]
        else:
            raise RuntimeError('Model not found')
    
    @staticmethod
    def CatTensor(stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = torch.cat(
            [stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

    @staticmethod
    def create_id2data(id2cl,df):
        dict_ = {}
        for id_ in id2cl:
            dict_[id_] = df.loc[id2cl[id_]].values
        return dict_

class LabelFeaturizer(BasicFeaturizer):
    def __init__(self, modelname, mode=None):
        super().__init__()
        self.modelname = modelname
        self.mode = mode


    def prefeaturize(self, encoded_dataframe):
        values_list = encoded_dataframe['IC50'].tolist()
        return values_list
        
    def featurize(self, values_list,index, mol):
        
        label = torch.Tensor([values_list[index]]).unsqueeze(-1)

        if self.modelname in ('AttentiveFP', 'FragAttentiveFP'):

            if self.mode == 'TRAIN':
                return label
                
            elif self.mode == 'EVAL':
                extended_label = torch.Tensor([])
                SingleBondList = GetSingleBonds(mol)
                if len(SingleBondList) == 0:
                    extended_label = self.CatTensor(extended_label, label)

                else:
                    for bond in SingleBondList:
                        extended_label = self.CatTensor(extended_label, label)
                return extended_label
            else:
                raise RuntimeError('Invalid configs')

        elif self.modelname in ('GAT','GCN'):
            return label
        else:
            raise RuntimeError('Model not found')

    @staticmethod
    def CatTensor(stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = torch.cat(
            [stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

class AttentiveFPFeaturizer(BasicFeaturizer):
    def __init__(self, atom_feature_size, bond_feature_size, max_degree, max_frag, mode):
        super().__init__()
        self.max_atom_num = 0
        self.max_bond_num = 0
        self.atom_feature_size = atom_feature_size
        self.bond_feature_size = bond_feature_size
        self.max_degree = max_degree
        self.mode = mode
        self.max_frag = max_frag

    def prefeaturize(self, id2smiles:dict):
        entire_atom_features = []
        entire_bond_features = []
        entire_atom_neighbor_list = []
        entire_bond_neighbor_list = []
        entire_atom_mask = []

        for index in id2smiles:
            SMILES = id2smiles[index]
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
            entire_atom_features.append(mol_atom_feature)
            entire_bond_features.append(mol_bond_feature)
            entire_atom_neighbor_list.append(mol_atom_neighbor_list)
            entire_bond_neighbor_list.append(mol_bond_neighbor_list)
            entire_atom_mask.append(mol_atom_mask)

        return [entire_atom_features, entire_bond_features, entire_atom_neighbor_list, entire_bond_neighbor_list, entire_atom_mask]

    def featurizenew(self, dataset, index, mol, Frag):
        [entire_atom_features, entire_bond_features, entire_atom_neighbor_list,entire_bond_neighbor_list, entire_atom_mask] = dataset  # from prefeaturizer

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
                    return [
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
                        JT_mask
                    ]

                else:
                    # No single bonds can be cut.
                    # Return in such format: Origin Graph, Frags, Junction Tree
                    JT_bond_features, JT_atom_neighbor_list, JT_bond_neighbor_list, JT_mask = self.CreateJunctionTree(
                        mol_bond_features, atom_neighbor_list=None, bond_neighbor_list=None, start_atom=None, end_atom=None, bondidx=[], frag_masks=[])

                    return [
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
                        JT_mask
                    ]

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
                    # extended_label = self.CatTensor(extended_label, label)

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


                return [
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
                ]
            else:
                print("Wrong mode.")
                raise RuntimeError

        return [mol_atom_features, mol_bond_features, mol_atom_neighbor_list, mol_bond_neighbor_list, mol_atom_mask]


    def GetPad(self, id2smiles):
        # dataset format: [{"SMILES": smiles, "Value": value}]

        for index in id2smiles:
            smiles = id2smiles[index]
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
        pad_bond_feature = torch.zeros(1, self.bond_feature_size)        # [1, bond_feature_size]
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
        super().__init__()
        self.max_atom_num = 0
        self.max_bond_num = 0
        self.atom_feature_size = 75

    def prefeaturize(self, id2smiles):
        SMILES_list = []
        entire_atom_features = []
        entire_bond_sparse = []
        conv_featurizer = ConvMolFeaturizer()

        for index in id2smiles:
            SMILES = id2smiles[index]
            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.MolFromSmiles(SMILES)

            SMILES_list.append(SMILES)
            convmol = conv_featurizer.featurize(SMILES)
            conv_feature = torch.Tensor(convmol[0].get_atom_features())

            adj_matrix = torch.Tensor(GetAdjMat(mol))
            sparse = GetSparseFromAdj(adj_matrix)[0]
            sparse = np.array(sparse.tolist())

            entire_atom_features.append(conv_feature)
            entire_bond_sparse.append(sparse)

        return entire_atom_features, entire_bond_sparse

    def featurize(self, dataset, index):
        entire_atom_features, entire_bond_sparse = dataset

        mol_atom_features = torch.Tensor(entire_atom_features[index])
        mol_bond_sparse = torch.Tensor(entire_bond_sparse[index]).long()

        return PyGData(x=mol_atom_features, edge_index=mol_bond_sparse)

    def CatTensor(self, stacked_tensor, new_tensor):
        extended_new_tensor = new_tensor.unsqueeze(dim=0)
        new_stacked_tensor = torch.cat(
            [stacked_tensor, extended_new_tensor], dim=0)
        return new_stacked_tensor

    def GetPad(self, id2smiles):

        for index in id2smiles:
            SMILES = id2smiles[index]
            mol = Chem.MolFromSmiles(SMILES)
            total_atom_num = len(mol.GetAtoms())
            total_bond_num = len(mol.GetBonds())
            self.max_atom_num = max(self.max_atom_num, total_atom_num)
            self.max_bond_num = max(self.max_bond_num, total_bond_num)

        self.pad_atom_idx = self.max_atom_num
        self.pad_bond_idx = self.max_bond_num

        self.max_atom_num += 1
        self.max_bond_num += 1

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

        if self.log:
            ("Using Attentive FP. Dataset is being checked.")
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


class Saver(object):
    def __init__(self,fold_dir, max_epoch):
        super().__init__()
        
        if fold_dir[-1] != '/':
            fold_dir = fold_dir + '/'

        self.ckpt_dir = fold_dir+ 'checkpoints/'
        self.optim_dir = fold_dir+ 'optim/'

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            os.mkdir(self.optim_dir)
        self.ckpt_count = 1
        self.EarlyStopController = EarlyStopController()
        self.maxepoch = max_epoch

    def SaveModel(self, model, optimizer, epoch, scores, mainmetric):
        # state = {'model': model, 'optimizer': optimizer, 'epoch': epoch}
        ckpt_name = os.path.join(self.ckpt_dir, f'epoch{epoch}.pt')
        optim_ckpt_name = os.path.join(self.optim_dir, f'epoch{epoch}.pt')
        if not os.path.exists(os.path.dirname(ckpt_name)):
            try:
                os.makedirs(os.path.dirname(ckpt_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        torch.save(model, ckpt_name)
        torch.save(optimizer, optim_ckpt_name)

        print("Model saved.")

        ShouldStop = self.EarlyStopController.ShouldStop(scores, self.ckpt_count,mainmetric.name)

        if ShouldStop:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("Early stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            # delete other models
            self.DeleteUselessCkpt(BestModelCkpt,remove_optim=True)
            return True, BestModelCkpt, BestValue
        
        elif self.ckpt_count == self.maxepoch:
            BestValue, BestModelCkpt = self.EarlyStopController.BestModel()
            print("The model didn't stop.")
            print("The Best model's ckpt idx is: ", BestModelCkpt)
            print("The Best Valid Value is: ", BestValue)
            self.DeleteUselessCkpt(BestModelCkpt,remove_optim=False)
            return False, BestModelCkpt, BestValue

        else:
            self.ckpt_count += 1
            BestValue, BestModelCkpt= self.EarlyStopController.BestModel()
            return False, BestModelCkpt, BestValue

    def DeleteUselessCkpt(self, BestModelCkpt,remove_optim):
        file_names = os.listdir(self.ckpt_dir)
        for file_name in file_names:
            ckpt_idx = int(re.findall('\d+',file_name)[-1])
            if ckpt_idx != BestModelCkpt:
                exact_file_path = self.ckpt_dir + file_name
                os.remove(exact_file_path)
        if remove_optim:
            shutil.rmtree(self.optim_dir)
        else:
            for file_name in os.listdir(self.optim_dir):
                ckpt_idx = int(re.findall('\d+',file_name)[-1])
                if ckpt_idx != BestModelCkpt:
                    exact_file_path = self.optim_dir + file_name
                    os.remove(exact_file_path)

    def LoadModel(self, ckpt=None, load_all=False):
        dir_files = os.listdir(self.ckpt_dir)  # list of the checkpoint files
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.ckpt_dir, x)))

            last_model_ckpt = dir_files[-1]   # find the latest checkpoint file.
            model = torch.load(os.path.join(self.ckpt_dir, last_model_ckpt))
            current_epoch = int(re.findall('\d+',last_model_ckpt)[-1])
            self.ckpt_count = current_epoch + 1  # update the ckpt_count, get rid of overwriting the existed checkpoint files.
            
            if load_all:
                optimizer = torch.load(os.path.join(self.optim_dir, f'epoch{current_epoch}.pt'))
                return model,optimizer
            else:
                return model
        else:
            return None, None if load_all else None
                
class EarlyStopController(object):
    def __init__(self):
        super().__init__()
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
        else:
            self.LowerThanMaxNum += 1
            if MainScore > self.LastResult:
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
        return self.MaxResult, self.MaxResultModelIdx

class StatusReport(object):

    def __init__(self,hyperpath,hypertune_stop_flag=False, trial=0, repeat=0, fold=0, epoch=0, run_dir=None):
        self._run_dir = run_dir
        self._status = {
            'hypertune_stop_flag':hypertune_stop_flag,
            'trial':trial,
            'repeat':repeat,
            'fold':fold,
            'epoch':epoch,
            'hyperpath': hyperpath, # YYYY-MM-DD_HyperRunNo.
        }

    def set_run_dir(self,run_dir):
        self._run_dir = run_dir
        with open(f'{self._run_dir}/status.json','w') as status_file:
            json.dump(self._status,status_file,indent=4)

    @classmethod
    def resume_run(cls,run_dir):
        with open(f'{run_dir}/status.json','r') as status_file:
            status = json.load(status_file)
        return cls(status['hyperpath'],status['hypertune_stop_flag'], status['repeat'], status['fold'], status['epoch'],run_dir=run_dir)

    def update(self,data):
        assert all(key in self._status.keys() for key in data)
        self._status.update(data)
        with open(f'{self._run_dir}/status.json','w') as status_file:
            json.dump(self._status,status_file,indent=4)
    
    def get_status(self):
        return self._status.values()

    def __getitem__(self,item):
        return self._status[item]

    def __call__(self):
        return self._status

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
            ValidOmicsInput = [tensor.squeeze(0) for tensor in ValidOmicsInput]
            ValidDrugInput = [tensor.squeeze(0) for tensor in ValidDrugInput]

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

class BasicDrugModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = None

    def forward(self,*args,**kwargs):
        raise NotImplementedError('This is the base module for drug neural network.')
    
    @property
    def name(self):
        return self._name

    @staticmethod
    def InputCUDA(listTensor:List[torch.Tensor]) -> List[torch.Tensor]:
        return [d.to(device) for d in listTensor]

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


def get_drug_model(modelname:str,config:dict):
    
    if modelname == 'GCN':
        return  GCNNet(
                num_features_xd=75,
                output_dim=config['drug_output_size']
            )
    if modelname == 'GAT':
        return  GATNet(
                num_features_xd=75,
                output_dim=config['drug_output_size']
            )
    if modelname == 'FragAttentiveFP':
        return  MolPredFragFPv8(
                atom_feature_size=72,  # 'atom_feature_size': 39
                bond_feature_size=10,  # 'bond_feature_size': 10
                FP_size=150,         # 'FP_size': 150
                atom_layers=3,     # 'atom_layers':3
                mol_layers=2,      # 'mol_layers':2
                DNN_layers=[256],   # 'DNNLayers':[512]
                output_size=config['drug_output_size'],
                drop_rate=config['drop_rate'],    # 'drop_rate':0.2
            )          # drugmodel = molpredfragfpv8

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
    _class_nickname = 'FragAttentiveFP'
    # mut_input_shape = None
    expr_input_shape = None
    meth_input_shape = None
    cnv_input_shape = None

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

def candragat_tuning(trial, OmicsData: Sequence[pd.DataFrame], TVset: pd.DataFrame, id2smiles_tv: dict, id2cl_tv: dict, max_tuning_epoch: int = 5):

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

def main():

    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')
    print('Start time:',start_time_formatted)

    num_devices = torch.cuda.device_count()

    set_base_seed(42)

    global ckpt_dir, max_epoch, max_tuning_epoch, device
    global use_mut, use_expr, use_meth, use_cnv, modelname, classify
    global status
    model_name = ["AttentiveFP", "GAT", "GCN","FragAttentiveFP"]
    parser = argparse.ArgumentParser(description='ablation_analysis')
    parser.add_argument('--modelname', dest='modelname', action='store', default = "AttentiveFP",
                        choices=model_name, help="AttentiveFP or GAT or GCN")
    parser.add_argument('--no_mut', dest='no_mut', default=False,
                        action="store_true", help='use gene mutation or not')
    parser.add_argument('--no_expr', dest='no_expr', default=False,
                        action="store_true", help='use gene expression or not')
    parser.add_argument('--no_methy', dest='no_methy', default=False,
                        action="store_true", help='use methylation or not')
    parser.add_argument('--no_cnv', dest='no_cnv', default=False,
                        action="store_true", help='use copy number variation or not')
    parser.add_argument('-c','--classify', dest='classify', default=False,
                        action="store_true", help='use copy number variation or not')
    parser.add_argument('-rs', '--resume', dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    parser.add_argument('-l', '--load_hyper', required=False,nargs=2, dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')

    args = parser.parse_args()

    classify = args.classify
    modelname = args.modelname

    device = torch.device('cuda')

    # ------ Loading Dataset  ------

    dataset_dir = "./data/"
    Mut_df = pd.read_csv(dataset_dir+'Mut_prep.csv', index_col=0, header=0)
    Expr_df = pd.read_csv(dataset_dir+'GeneExp_prep.csv', index_col=0, header=0)
    Meth_df = pd.read_csv(dataset_dir+'Meth_prep.csv', index_col=0, header=0)
    CNV_df = pd.read_csv(dataset_dir+'GeneCN_prep.csv', index_col=0, header=0)
    print('Omics data loaded.')
    OmicsData = [Mut_df, Expr_df, Meth_df, CNV_df]
    MultiOmicsMolNet.set_omics_input_shape(OmicsData)

    DrugOmics_df = pd.read_csv(dataset_dir+"DrugSens_train_CDG.csv", index_col = 0, header = 0)
    DrugOmics_df_test = pd.read_csv(dataset_dir+"DrugSens_test_CDG.csv", index_col = 0, header = 0)


    # ------ Debug Mode ------

    if args.debug:
        print('-- DEBUG MODE --')
        n_trials = 1
        max_tuning_epoch = 1
        max_epoch = 3
        folds=2
        DrugOmics_df = DrugOmics_df.iloc[:100]
        DrugOmics_df_test = DrugOmics_df_test.iloc[:100]
        batchsize = 16
    else:
        n_trials = 50
        max_tuning_epoch = 2
        max_epoch = 30
        folds=5
        batchsize = 256

    # ------ Ablation ------
    
    use_mut, use_expr, use_meth, use_cnv = True, True, True, True
    outtext_list = []
    if args.no_mut:
        use_mut = False
        Mut_df= pd.DataFrame(0, index=np.arange(len(Mut_df)), columns=Mut_df.columns)
        outtext_list.append(0)
    else:
        outtext_list.append(1)

    if args.no_expr:
        use_expr = False
        Expr_df = pd.DataFrame(0, index=np.arange(len(Expr_df)), columns=Expr_df.columns)
        outtext_list.append(0)
    else:
        outtext_list.append(1)

    if args.no_methy:
        use_meth = False
        Meth_df = pd.DataFrame(0, index=np.arange(len(Meth_df)), columns=Meth_df.columns)
        outtext_list.append(0)
    else:
        outtext_list.append(1)

    if args.no_cnv:
        use_cnv = False
        CNV_df = pd.DataFrame(0, index=np.arange(len(CNV_df)), columns=CNV_df.columns)
        outtext_list.append(0)
    else:
        outtext_list.append(1)

    # ----- Setup ------
    mainmetric, report_metrics, prediction_task = (BCE(),[BCE(),AUC(),AUC_PR()],'clas') if classify else (MSE(),[MSE(),RMSE(),PCC(),R2(),SRCC()],'regr')
    criterion = nn.MSELoss()
    print('Loading dataset...')
    TVset, id2smiles_tv, id2cl_tv = OmicsMolDatasetCreator().createDataset(DrugOmics_df) 
    Testset, id2smiles_test, id2cl_test = OmicsMolDatasetCreator().createDataset(DrugOmics_df_test)
    # Trainset, Validset = train_test_split(DrugOmicsDataset, test_size=0.1, random_state = 42)
    print('-- TEST SET --')
    
    DatasetTest = OmicsMolDataset(Testset,OmicsData,modelname,id2smiles_test,id2cl_test,mode='EVAL',log=True)

    # ------- Storage Path -------
    today_date = date.today().strftime('%Y-%m-%d')

    resultfolder = 'CANDraGAT_June2022/results' if not args.debug else 'CANDraGAT_June2022/test'
    hypertune_stop_flag = False
    if args.hyperpath is None:
        hyperexpdir = f'{resultfolder}/{modelname}/hyperparameters/'
        os.makedirs(hyperexpdir,exist_ok=True)
        num_hyperrun = 1
        while os.path.exists(hyperexpdir+f'{today_date}_HyperRun{num_hyperrun}.json'):
            num_hyperrun+=1
        else:
            hyperexpname = f'{today_date}_HyperRun{num_hyperrun}' 
            json.dump({},open(hyperexpdir+hyperexpname+'.json','w')) # placeholder file
        RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun1'
        os.makedirs(RUN_DIR)
        status = StatusReport(hyperexpname,hypertune_stop_flag)
        status.set_run_dir(RUN_DIR)
        print(f'Your run directory is "{RUN_DIR}"')

        print('Start hyperparameters optimization')
        def candragat_tuning_simplified(trial):
            return candragat_tuning(trial, OmicsData, TVset, id2smiles_tv, id2cl_tv, max_tuning_epoch)
        run_hyper_study(study_func=candragat_tuning_simplified, n_trials=n_trials,hyperexpfilename=hyperexpdir+hyperexpname)
        pt_param = get_best_trial(hyperexpdir+hyperexpname)
        exp_name=f'{hyperexpname}_TestRun1'
        hypertune_stop_flag = True
        status.update({'hypertune_stop_flag':True})
        print('Hyperparameters optimization done')

    else: 
        hypertune_stop_flag = True
        hyper_modelname,hyperexpname= args.hyperpath
        status = StatusReport(hyperexpname,hypertune_stop_flag)
        status.set_run_dir(RUN_DIR)
        hyper_jsondir = f'{resultfolder}/{hyper_modelname}/hyperparameters/{hyperexpname}.json'
        pt_param = get_best_trial(hyper_jsondir)
        num_run = 1
        while os.path.exists(f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'):
            num_run+=1
        RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'
        os.makedirs(RUN_DIR)
        exp_name=f'{hyperexpname}_TestRun{num_run}'
        print(f'Your run directory is "{RUN_DIR}"')
    
    outtext_list.insert(0,exp_name)
    
    drop_rate = pt_param['drop_rate']
    lr = pt_param['lr']
    weight_decay = pt_param['WeightDecay']
    omics_output_size = pt_param['omics_output_size']
    drug_output_size = pt_param['drug_output_size']
    report_metrics_name = [metric.name for metric in report_metrics]
    resultsdf = pd.DataFrame(columns=report_metrics_name,index=list(range(folds)))

    #############################
    
    for fold, (Trainset, _) in enumerate(df_kfold_split(TVset),start=0):
        print(f'\n=============== Fold {fold+1}/{folds} ===============\n')

        seed = set_seed()
        print(f'-- TRAIN SET {fold+1} --')

        DatasetTrain = OmicsMolDataset(Trainset, omicsdata=OmicsData, modelname=modelname, id2smiles=id2smiles_tv, id2cl=id2cl_tv, mode='TRAIN',log=True)
        print('DatasetTrain len =',len(DatasetTrain))
        print('Trainset len =',len(Trainset))
        if modelname in ("AttentiveFP","FragAttentiveFP"):
            trainloader = data.DataLoader(DatasetTrain, batch_size=batchsize, num_workers=2,
                                        drop_last=True, worker_init_fn=np.random.seed(seed), pin_memory=True)
            testloader = data.DataLoader(DatasetTest, batch_size=1, shuffle=False, num_workers=0,
                                        drop_last=True, worker_init_fn=np.random.seed(seed), pin_memory=True)

        elif modelname in ("GAT", "GCN"):
            trainloader = PyGDataLoader(DatasetTrain, batch_size=batchsize, num_workers=2, drop_last=True,
                                        worker_init_fn=np.random.seed(seed), pin_memory=True, collate_fn=graph_collate_fn)
            testloader = PyGDataLoader(DatasetTest, batch_size=1, shuffle=False, num_workers=0, drop_last=True,
                                    worker_init_fn=np.random.seed(seed), pin_memory=True, collate_fn=graph_collate_fn)

        ckpt_dir = os.path.join(RUN_DIR, f'fold_{fold}-seed_{seed}/')
        saver = Saver(ckpt_dir, max_epoch)
        model, optimizer = saver.LoadModel(load_all=True)

        if model is None:
            drug_model = get_drug_model(modelname,pt_param)
            model = MultiOmicsMolNet(
                drug_nn=drug_model,
                drug_output_size=drug_output_size,
                omics_output_size=omics_output_size,
                drop_rate=drop_rate,
                classify=classify
            )
            if num_devices > 1:
                model = nn.parallel.DistributedDataParallel(model)
            model.to(device)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

        torch.cuda.empty_cache()
        validloss = []
        trainloss = []
        printrate = 20
        model.train()

        optimizer.zero_grad()
        model.zero_grad()

        print("Start Training...")
        stop_flag = False

        for num_epoch in range(1,max_epoch+1):

            time_start = datetime.now()
            cum_loss = 0.0
            printloss = 0.0
            print(f'Epoch:{num_epoch}/{max_epoch}')
            status.update({
                    # 'repeat':repeat,
                    'fold':fold,
                    'epoch':num_epoch
                })
            if stop_flag:
                break

            start_iter = datetime.now()
            print(f'Trainloader size = {len(trainloader)}')
            for ii, Data in enumerate(trainloader): 

                if stop_flag:
                    break

                OmicsInput, DrugInput, Label = Data

                Label = Label.squeeze(-1).to(device)    # [batch, task]
                output = model(OmicsInput, DrugInput)    # [batch, output_size]

                loss = criterion(output, Label)
                cum_loss += loss.detach()
                printloss += loss.detach()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (ii+1) % 20 == 0:
                    torch.cuda.empty_cache()

                if (ii+1) % printrate == 0:
                    duration_iter = (datetime.now()-start_iter).total_seconds()
                    print("Iteration {}/{}".format(ii+1, len(trainloader)))
                    print("Duration = ", duration_iter,'seconds, loss = {:.4f}'.format((printloss/printrate)))
                    printloss = 0.0
                    start_iter = datetime.now()

            time_end = datetime.now()
            duration_epoch = (time_end-time_start).total_seconds()
            # scheduler.step()
            trainmeanloss = (cum_loss/len(trainloader)).item()
            print(f"Epoch duration = {duration_epoch} seconds")
            print(f"Train loss on Epoch {num_epoch} = {trainmeanloss}")

            testresult = Validation(testloader, model, report_metrics)[0]
            testmeanloss = testresult[mainmetric.name]

            print('===========================================================\n')
            stop_flag, _, _ = saver.SaveModel(model, optimizer, num_epoch, testresult, mainmetric)
            validloss.append(testmeanloss)
            trainloss.append(trainmeanloss)

        torch.cuda.empty_cache()

        bestmodel = saver.LoadModel()
        score, predIC50, labeledIC50 = Validation(testloader, bestmodel, report_metrics)
        for metric in score:
            resultsdf.loc[fold,metric] = score[metric]

        modelpath3 = f'{RUN_DIR}/fold_{fold}-seed_{seed}/'
        traintestloss = np.array([trainloss, validloss])
        np.savetxt(modelpath3+'traintestloss.csv', traintestloss, delimiter=',', fmt='%.5f')

        # create text file for test
        testpredlabel = np.array([predIC50, labeledIC50]).T
        np.savetxt(modelpath3+'testpred_label.csv', testpredlabel, delimiter=',', fmt='%.5f')

    for col in resultsdf.columns:
        mean, interval = compute_confidence_interval(resultsdf[col])
        outtext_list.extend((mean,interval))

    end_time = arrow.now()
    end_time_formatted = end_time.format('DD/MM/YYYY HH:mm:ss')
    print('\n===========================================================\n')
    print('Finish time:',end_time_formatted)
    elapsed_time = end_time - start_time

    print('Writing Output...')

    resultsdf.to_csv(f'{RUN_DIR}/ExperimentSummary.csv')

    summaryfile = f'CANDraGAT_June2022/{resultfolder}/{modelname}/{prediction_task}/ResultSummarySheet.csv'
    if not os.path.exists(summaryfile):
        write_result_files(report_metrics,summaryfile)
    with open(summaryfile, 'a') as outfile:
        outtext_list.insert(1,start_time_formatted)
        outtext_list.insert(2,end_time_formatted)
        outtext_list.insert(3,str(elapsed_time).split('.')[0])
        output_writer = csv.writer(outfile,delimiter = ',')
        output_writer.writerow(outtext_list)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()

