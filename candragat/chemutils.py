import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch

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

def GetSparseFromAdj(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]
    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    return torch.stack(index, dim=0), edge_attr
    
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
