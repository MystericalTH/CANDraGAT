import torch
import numpy as np
from torch.utils import data 
from candragat.featurizer import *
from torch_geometric.loader import DataLoader as PyGDataLoader
from candragat.utils import graph_collate_fn
from candragat.models import MultiOmicsMolNet
import pandas as pd
import os
import shutil
from torch_geometric.data import OnDiskDataset
import gc

def DrugSensTransform(drugsens_df:pd.DataFrame):
    smiles_list = drugsens_df['SMILES'].unique().tolist()
    cl_list = drugsens_df['cell_line'].unique().tolist()
    drugsens_new_df = drugsens_df.copy()
    drugsens_new_df['SMILES'] = drugsens_df['SMILES'].apply(lambda x: smiles_list.index(x)).astype(int)
    drugsens_new_df['cell_line'] = drugsens_df['cell_line'].apply(lambda x: cl_list.index(x)).astype(int)
    drugsens_new_df.drop(columns=['gdsc_name'], inplace=True)
    drugsens_new_df = drugsens_new_df[["cell_line","SMILES","IC50"]]
    return smiles_list, cl_list, drugsens_new_df


class OmicsDataset(data.Dataset):
    def __init__(self, cl_list, **kwargs):
        super().__init__()
        try:
            assert len(kwargs) > 0
        except AssertionError:
            raise RuntimeError('Data not found.')

        self.num_samples = len(cl_list)
        self.cl_list = cl_list
        self.mut = torch.FloatTensor(kwargs['mut'].reindex(cl_list).to_numpy()).unsqueeze(1) if 'mut' in kwargs else torch.zeros(self.num_samples)
        self.expr = torch.FloatTensor(kwargs['expr'].reindex(cl_list).to_numpy()) if 'expr' in kwargs else torch.zeros(self.num_samples)
        self.meth = torch.FloatTensor(kwargs['meth'].reindex(cl_list).to_numpy()) if 'meth' in kwargs else torch.zeros(self.num_samples)
        self.cnv = torch.FloatTensor(kwargs['cnv'].reindex(cl_list).to_numpy()) if 'cnv' in kwargs else torch.zeros(self.num_samples)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return [self.mut[index], self.expr[index], self.meth[index], self.cnv[index]]
    
    @property
    def input_size(self):
        return [self.mut.shape[-1], self.expr.shape[-1], self.meth.shape[-1], self.cnv.shape[-1]]

class TestDrugTensorDataset(OnDiskDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.schema = torch.Tensor
    
    def serialize(self, data):
        name = f'{len(self)}.pt'
        path = os.path.join(self.root, name)
        torch.save(data, path)
        return name

    def deserialize(self, name):
        path = os.path.join(self.root, name)
        tensors = torch.load(path)
        return tensors
        
class DrugOmicsDataset(data.Dataset):

    def __init__(self, drugsens, omics_dataset, smiles_list, drug_mode: str, EVAL = False, **kwargs):
        super().__init__()
        self.eval = EVAL
        self.drugsens = drugsens
        self.cl_list = omics_dataset.cl_list
        self.smiles_list = smiles_list
        self.__drug_mode = drug_mode
        self.drug_featurizer = self.get_featurizer(drug_mode, EVAL)
        self.drug_dataset = self.drug_featurizer.prefeaturize(self.smiles_list)
        self.omics_dataset = omics_dataset
        if EVAL:
            assert kwargs['root'] is not None
            self.test_drug_tensor_dataset = TestDrugTensorDataset(root = kwargs['root'])

    def __getitem__(self, index):
        cl_idx, drug_idx, _label = self.drugsens.iloc[index]
        label = torch.FloatTensor([_label])
        cl_idx = int(cl_idx)
        drug_idx = int(drug_idx)
        omics_data = self.omics_dataset[cl_idx]
        
        if self.eval:
            drug_data = self.test_drug_tensor_dataset.get(drug_idx)
        else:
            drug_data = self.drug_featurizer.featurize(self.drug_dataset, drug_idx)
        return [omics_data, drug_data], label

    def __len__(self):
        return len(self.drugsens)

    def get_featurizer(self, drug_featurizer, EVAL):
        MolFeaturizerList = {
            'FragAttentiveFP': AttentiveFPFeaturizer(Frag=True, EVAL=EVAL),
            'AttentiveFP': AttentiveFPFeaturizer(Frag=False, EVAL=EVAL),
            'GCN': DCGraphFeaturizer(),
            'GAT': DCGraphFeaturizer()
        }
        if drug_featurizer not in MolFeaturizerList.keys():
            raise ValueError("Given featurizer does not exist.")
        else:
            return MolFeaturizerList[drug_featurizer]

    def precalculate_drug_tensor(self, model: MultiOmicsMolNet):
        print("Precalculating drug tensors...", flush=True)
        os.makedirs(self.test_drug_tensor_dataset.root, exist_ok=True)
        model.eval().cpu()
        for drug_idx in self.drugsens['SMILES'].unique():
            drug_data = self.drug_featurizer.featurize(self.drug_dataset, int(drug_idx))
            drug_tensor = model.drug_nn(drug_data).mean(dim=0)
            self.test_drug_tensor_dataset.append(drug_tensor)

    def clear_drug_tensor(self):
        shutil.rmtree(self.test_drug_tensor_dataset.root, ignore_errors=True)
        self.test_drug_tensor_dataset = TestDrugTensorDataset(root = self.test_drug_tensor_dataset.root)
        gc.collect()
    
def get_dataloader(Dataset,modelname, batch_size = None):
    if Dataset.eval:
        batch_size = batch_size
        num_workers = 0
        drop_last = False
    else: 
        assert batch_size is not None
        batch_size = batch_size
        num_workers = 1
        drop_last = True
    if modelname in ("AttentiveFP","FragAttentiveFP"):
        return data.DataLoader(Dataset, batch_size=batch_size, num_workers=num_workers,
                                    drop_last=drop_last, worker_init_fn=np.random.seed(), pin_memory=True)
    elif modelname in ("GAT", "GCN"):
        return PyGDataLoader(Dataset, batch_size=batch_size, num_workers=num_workers,
                                worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)
    else:
        raise NotImplementedError('Invalid model name / not yet implemented')
