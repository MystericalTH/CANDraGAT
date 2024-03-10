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

def get_best_trial(study_name, result_folder: str = "results"):
    db_path = f"sqlite:///{result_folder}/hyperparameter-tuning.db"
    storage = optuna.storages.RDBStorage(
        db_path,
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
        engine_kwargs={"connect_args": {"timeout": 10}}
    )
    study = optuna.load_study(storage=storage, study_name=study_name)
    best_trial_param = {k:v for k, v in study.best_trial.params.items()}
    return best_trial_param

def run_hyper_study(study_func, N_TRIALS, hyperfilename, study_name, study_attrs, result_folder: str = "results"):
    global GLOBAL_N_TRIALS
    GLOBAL_N_TRIALS = N_TRIALS
    db_path = f"sqlite:///{result_folder}/hyperparameter-tuning.db"
    storage = optuna.storages.RDBStorage(
        db_path,
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
        engine_kwargs={"connect_args": {"timeout": 10}}
    )
    study = optuna.create_study(storage=storage, direction="minimize", study_name=study_name, load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    print(study_name + " is running. Trials saved to " + db_path)
    for k, v in study_attrs.items():
        study.set_user_attr(k, v)
    n_trials = max(0, N_TRIALS-len(study.trials))
    
    study.optimize(study_func, n_trials=n_trials, gc_after_trial=True)
    trial = study.best_trial
    best_trial_param = {k:v for k, v in trial.params.items()}
    with open(hyperfilename,'w') as jsonfile:
        json.dump(best_trial_param, jsonfile, indent=4)
    return study

def delete_study(study_name, result_folder: str = "results"):
    db_path = f"sqlite:///{result_folder}/hyperparameter-tuning.db"
    storage = optuna.storages.RDBStorage(
        db_path,
        heartbeat_interval=1,
        engine_kwargs={"connect_args": {"timeout": 10}}
    )
    optuna.delete_study(storage=storage, study_name=study_name)
    print("Successfully deleted study " + study_name)
def get_trial_config(trial: optuna.Trial) -> Dict[str, Union[float, int]]:
    """
    Return configuration dictionary with trial suggestion value as an object.
    """
    return {
        'drop_rate': trial.suggest_float('drop_rate',0.1,0.9,step=0.05),
        'lr': trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
        'omics_output_size': trial.suggest_int('omics_output_size',150,500,step=50),
        'drug_output_size': trial.suggest_int('drug_output_size',150,500,step=50)
    }

def candragat_tuning(trial:optuna.Trial, Trainset, Validset, omics_dataset, smiles_list, modelname, status,
        batch_size, mainlogger, pbarlogger, args, RUN_DIR, criterion, max_tuning_epoch: int = 5, DEVICE = torch.device("cuda:0"), weight=None):

    mainlogger.info(f"=============== Trial {trial.number+1}/{GLOBAL_N_TRIALS} ===============")
    pt_param = get_trial_config(trial)
    drug_model = get_drug_model(modelname,pt_param)

    model = MultiOmicsMolNet(
                drug_model,
                drug_output_size=pt_param['drug_output_size'],
                omics_output_size=pt_param['omics_output_size'],
                drop_rate=pt_param['drop_rate'],
                input_size_list=omics_dataset.input_size,
                args=args
    )

    model.zero_grad()
    optimizer = optim.Adam(model.parameters(),lr=pt_param['lr'])
    optimizer.zero_grad()

    DatasetTrain = DrugOmicsDataset(Trainset, omics_dataset, smiles_list, modelname, EVAL=False, weight=weight)
    DatasetValid = DrugOmicsDataset(Validset, omics_dataset, smiles_list, modelname, EVAL=True, root = os.path.join(RUN_DIR, 'drug-tensors'))
    trainloader = get_dataloader(DatasetTrain, modelname, batch_size=batch_size)
    validloader = get_dataloader(DatasetValid, modelname, batch_size=1)

    earlystopper = EarlyStopController()
    
    for epoch in range(max_tuning_epoch):
        model.train().to(DEVICE)
        cum_loss = 0.0
        pbar = tqdm.tqdm(trainloader, total=len(trainloader), file=TqdmToLogger(pbarlogger), mininterval=10, desc='Hyperparameter Tuning - Training')
        for Data in pbar:

            [OmicsInput, DrugInput], Label = Data[:2]
            DrugInput = DrugInputToDevice(DrugInput, modelname, DEVICE)
            OmicsInput = [tensor.to(DEVICE) for tensor in OmicsInput]
            # Label = Label.to(DEVICE)    # [batch, task]
            #Label = Label.t()            # [task, batch]
            output = model([OmicsInput,DrugInput])    # [batch, output_size]
            if weight is None:
                loss = criterion(output.to(CPU), Label, requires_backward = True)
            else:
                loss = criterion(output.to(CPU), Label, weight=Data[2], requires_backward = True)
                print(Data[2])
            cum_loss += loss.detach()
            pbar.set_postfix({'loss': loss.item()}, refresh=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        trainmseloss = (cum_loss/len(trainloader)).item()
        mainlogger.info(f"Epoch {epoch+1}/{max_tuning_epoch} - Training MSE Loss: {trainmseloss}")
        metric = BCE() if args['task']=='clas' else MSE()
        score = Validation(validloader, model, metric, modelname, mainlogger, pbarlogger, CPU)[0]
        mainlogger.info(f"Epoch {epoch+1}/{max_tuning_epoch} - Validation {metric.name}: {score}")
        trial.report(score[metric.name], epoch)
        
        if earlystopper.ShouldStop(score, epoch, metric.name):
            break
            
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    status.update({'trial': status['trial'] + 1})
    # validmseloss.append(results[metric.name])
    # return np.mean(validmseloss)
    return earlystopper.MaxResult
