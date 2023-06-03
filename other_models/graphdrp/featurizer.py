import pandas as pd
import torch
from rdkit import Chem
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List

from candragat.chemutils import GetAtomFeatures
import numpy as np

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
    HybridizationList = [Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2]
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
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R','S']
            loc = ChiList.index(chi)
            feature[70+loc] = 1
            print("Chirality resolving finished.")
            pass
        except:
            feature[70] = 0
            feature[71] = 0
    return feature
    
def smiles_to_graph(smiles, torchTensor: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    
    c_size = mol.GetNumAtoms()
    
    mol_atom_features = []
    for atom in mol.GetAtoms():
        atom_features = GetAtomFeatures(atom)
        mol_atom_features.append(atom_features)
        # features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    if torchTensor == True:
        c_size = torch.Tensor(c_size)
        mol_atom_features = torch.Tensor(mol_atom_features)
        edge_index = torch.Tensor(edge_index)

    return c_size, mol_atom_features, edge_index

# class GraphDRP(BaseNet):
def get_dataset(drug_response_path:str, gene_exp_path:str) -> List[Data]:

    """
    Args

    1. ```drug_response_path```
    2. ```gene_exp_path```
    """
    drug_response_df = pd.read_csv(drug_response_path)
    gene_exp_df = pd.read_csv(gene_exp_path, index_col=0)
    graphs = [smiles_to_graph(smiles, torchTensor=True) for smiles in drug_response_df['smiles']]
    gene_exp_features = [torch.Tensor(gene_exp_df.loc[cell_line]) for cell_line in drug_response_df['cell_line']]
    dataset = [[*graph, gene_exp] for graph, gene_exp in zip(graphs, gene_exp_features)]
    return dataset
    # def
