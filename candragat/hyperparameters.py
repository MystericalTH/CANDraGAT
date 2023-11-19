import pandas as pd
import torch
from torch import nn, optim
from torch.utils import data
from torch_geometric.loader import DataLoader as PyGDataLoader
from candragat.models import MultiOmicsMolNet, get_drug_model
from candragat.dataset import DrugOmicsDataset, get_dataloader
from candragat.utils import *
import numpy as np
import optuna
from optuna.storages import RetryFailedTrialCallback
import json
from candragat.validation import Validation
from candragat.metrics import BCE, MSE
from typing import Dict, Union

CPU = torch.device('cpu')
# trial_id = 0


def get_best_trial(hyper_dir):
    with open(f'{hyper_dir}.json','r') as jsonfile:
        best_trial_param = json.load(jsonfile)
    return best_trial_param

def run_hyper_study(study_func, n_trials, hyperexpfilename, study_name):
    global N_TRIALS
    N_TRIALS = n_trials
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{hyperexpfilename}.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
        engine_kwargs={"connect_args": {"timeout": 10}}
    )
    study = optuna.create_study(storage=storage, direction="minimize", study_name=study_name, load_if_exists=True)
    print(study_name + " is running. Trials saved to " + f"sqlite:///{hyperexpfilename}.db")
    study.optimize(study_func, n_trials=n_trials, gc_after_trial=True)
    trial = study.best_trial
    best_trial_param = dict()
    for key, value in trial.params.items():
        best_trial_param[key]=value
    with open(f'{hyperexpfilename}.json','w') as jsonfile:
        json.dump(best_trial_param,jsonfile,indent=4)
    return study

def get_trial_config(trial: optuna.Trial) -> Dict[str, Union[float, int]]:
    """
    Return configuration dictionary with trial suggestion value as an object.
    """
    return {
        # 'batchsize': 2**(trial.suggest_int('batchsize', 2,5)),
        'drop_rate': trial.suggest_float('drop_rate',0.1,0.9,step=0.05),
        'lr': trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-5, log=True),
        'omics_output_size': trial.suggest_int('omics_output_size',100,300,step=25),
        'drug_output_size': trial.suggest_int('drug_output_size',100,300,step=25)
    }

def candragat_tuning(trial:optuna.Trial, drugsens_df, omics_dataset, smiles_list, modelname, status,
        batch_size, mainlogger, pbarlogger, args, max_tuning_epoch: int = 5, DEVICE = torch.device("cuda:0")):
    # global trial_id
    # trial_id += 1
    mainlogger.info(f"Trial {trial.number}/{N_TRIALS}")
    criterion = nn.MSELoss()
    pt_param = get_trial_config(trial)
    drug_model = get_drug_model(modelname,pt_param)
    validmseloss = []

    for Trainset, Validset in df_kfold_split(drugsens_df):

        model = MultiOmicsMolNet(
                    drug_model,
                    drug_output_size=pt_param['drug_output_size'],
                    omics_output_size=pt_param['omics_output_size'],
                    drop_rate=pt_param['drop_rate'],
                    input_size_list=omics_dataset.input_size,
                    args=args
        ).to(DEVICE)

        model.train()
        model.zero_grad()
        optimizer = optim.Adam(model.parameters(),lr=pt_param['lr'],weight_decay=pt_param['weight_decay'])
        optimizer.zero_grad()

        DatasetTrain = DrugOmicsDataset(Trainset, omics_dataset, smiles_list, modelname, EVAL=False)
        DatasetValid = DrugOmicsDataset(Validset, omics_dataset, smiles_list, modelname, EVAL=True)
        trainloader = get_dataloader(DatasetTrain, modelname, batch_size=batch_size)
        validloader = get_dataloader(DatasetValid, modelname, batch_size=batch_size)

        for epoch in range(max_tuning_epoch):

            cum_loss = 0.0
            pbar = tqdm.tqdm(trainloader, total=len(trainloader), file=TqdmToLogger(pbarlogger), mininterval=10, desc='Hyperparameter Tuning - Training')
            for Data in pbar:

                [OmicsInput, DrugInput], Label = Data
                DrugInput = DrugInputToDevice(DrugInput, modelname, DEVICE)
                OmicsInput = [tensor.to(DEVICE) for tensor in OmicsInput]
                Label = Label.squeeze(-1).to(DEVICE)    # [batch, task]
                #Label = Label.t()            # [task, batch]
                output = model([OmicsInput,DrugInput])   # [batch, output_size]
                loss = criterion(output,Label).float()
                cum_loss += loss.detach()
                pbar.set_postfix({'loss': loss.item()}, refresh=False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            trainmseloss = (cum_loss/len(trainloader)).item()
            mainlogger.info(f"Epoch {epoch+1}/{max_tuning_epoch} - Training MSE Loss: {trainmseloss}")
        metric = BCE() if args['task']=='clas' else MSE()
        results = Validation(validloader, model, metric, modelname, mainlogger, pbarlogger, CPU)[0]
        validmseloss.append(results[metric.name])
    
    status.update({'trial': status['trial'] + 1})
    return np.mean(validmseloss)
