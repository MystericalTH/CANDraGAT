import torch
import numpy as np
from torch.utils import data 
from candragat.featurizer import *
from torch_geometric.loader import DataLoader as PyGDataLoader
from candragat.utils import graph_collate_fn
import pandas as pd

def DrugsensTransform(drugsens_df:pd.DataFrame):
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

class DrugOmicsDataset(data.Dataset):

    def __init__(self, drugsens, omics_dataset, smiles_list, drug_mode: str, EVAL = False):
        super().__init__()
        self.eval = EVAL
        self.drugsens = drugsens
        self.cl_list = omics_dataset.cl_list
        self.smiles_list = smiles_list
        self.__drug_mode = drug_mode
        self.drug_featurizer = self.get_featurizer(drug_mode, EVAL)
        self.drug_dataset = self.drug_featurizer.prefeaturize(self.smiles_list)
        self.omics_dataset = omics_dataset

    def __getitem__(self, index):
        cl_idx, drug_idx, _label = self.drugsens.iloc[index]
        label = torch.FloatTensor([_label])
        cl_idx = int(cl_idx)
        drug_idx = int(drug_idx)
        # print("cl:",cl_idx, "- drug:",drug_idx)
        omics_data = self.omics_dataset[cl_idx]
        drug_data = self.drug_featurizer.featurize(self.drug_dataset, drug_idx)
        if self.eval and self.__drug_mode == 'FragAttentiveFP':
            num_samples = drug_data[0].shape[0]
            omics_data = [torch.stack([feature] * num_samples) for feature in omics_data]
            # print('EVAL omics_data:', [x.shape for x in omics_data])
            label = torch.stack([label] * num_samples)
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

def get_dataloader(Dataset,modelname, batch_size = None):
    if Dataset.eval:
        batch_size = 1
        num_workers = 0
        drop_last = False
    else: 
        assert batch_size is not None
        batch_size = batch_size
        num_workers = 2
        drop_last = True
    if modelname in ("AttentiveFP","FragAttentiveFP"):
        return data.DataLoader(Dataset, batch_size=batch_size, num_workers=num_workers,
                                    drop_last=drop_last, worker_init_fn=np.random.seed(), pin_memory=True)
    elif modelname in ("GAT", "GCN"):
        return PyGDataLoader(Dataset, batch_size=batch_size, num_workers=num_workers,
                                worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)
    else:
        raise NotImplementedError('Invalid model name / not yet implemented')
