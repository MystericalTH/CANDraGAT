import numpy as np
import pandas as pd
from matplotlib import cm as cm
import optuna
import json
import os
import argparse
from datetime import date
import csv
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import deepchem as dc
from deepchem.data import DiskDataset
from deepchem.splits import RandomStratifiedSplitter
import xgboost as xgb
from optuna.storages import RetryFailedTrialCallback
import torch
import torch.nn.functional as F
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse_score,  mean_absolute_error as mae_score
from sklearn.metrics import r2_score, roc_auc_score,f1_score,precision_score,recall_score
from scipy.stats import spearmanr
from itertools import product
import pandas as pd
import pickle as pkl
from candragat import *
import gc
import arrow
import shutil
import logging

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return round(m, 4), round(pm, 4)

def set_seed(seed=None):
    if seed == None:
        seed = random.randrange(100)
    # random.seed(seed)
    np.random.seed(seed)
    return seed

def set_split_seed(new_seed=None):
    if new_seed is None:
        new_seed = random.randrange(1000)
    return new_seed

    
def df_to_dmatrix(data):
    return xgb.DMatrix(data)
    # –––––––––––––– Utils –––––––––––––––




class StatusReport(object):

    def __init__(self,hyperpath,database,hypertune_stop_flag=False, repeat=0, fold=0, run_dir=None):
        self._run_dir = run_dir
        self._status = {
            'hypertune_stop_flag':hypertune_stop_flag,
            'repeat':repeat,
            'fold':fold,
            'hyperpath': hyperpath, # YYYY-MM-DD_HyperRunNo.
            'database': database
        }

    def set_run_dir(self,run_dir):
        self._run_dir = run_dir
        with open(f'{self._run_dir}/status.json','w') as status_file:
            json.dump(self._status,status_file,indent=4)

    @classmethod
    def resume_run(cls,run_dir):
        with open(f'{run_dir}/status.json','r') as status_file:
            status = json.load(status_file)
        return cls(status['hyperpath'],status['database'],status['hypertune_stop_flag'], status['repeat'], status['fold'],run_dir=run_dir)

    def update(self,data): # must be called after run_dir is specified
        assert all(key in self._status.keys() for key in data)
        self._status.update(data)
        with open(f'{self._run_dir}/status.json','w') as status_file:
            json.dump(self._status,status_file,indent=4)
    
    def get_status(self):
        return self._status.values()

    def __call__(self):
        return self._status

def calculate_score(pred,label,criterion):
    squeeze_total_pred=np.array([])
    squeeze_total_label=np.array([])
    score_list = []
    for i in range(pred.shape[1]):
        task_label = label[:,i]
        task_pred = pred[:,i]
        task_masks = ~np.isnan(task_label)
        masked_label = np.squeeze(task_label[task_masks])
        masked_pred = np.squeeze(task_pred[task_masks])
        squeeze_total_label = np.concatenate((squeeze_total_label,masked_label))
        squeeze_total_pred = np.concatenate((squeeze_total_pred,masked_pred))
        score_list.append(float(criterion(masked_label,masked_pred)))
    if n_tasks > 1:
        score_list.insert(0,float(criterion(squeeze_total_label,squeeze_total_pred)))
    return score_list

class ResultsReport(object):
    def __init__(self, target, metrics, run_dir: str):
        assert all(param is not None for param in (target, metrics)) 
        self._metrics = metrics
        self._metrics_name = [criterion.name for criterion in metrics]
        self._target = target
        self._run_dir = run_dir
        self._resultsdf_col = [i for i in target]
        # if n_tasks > 1:
        #     self._resultsdf_col.insert(0,'Overall')
        results_summary_dir= f'{run_dir}/ResultSummary.csv'
        if os.path.exists(results_summary_dir):
            self._resultsdf = pd.read_csv(results_summary_dir,header=[0,1],index_col=[0,1])
        else:
            index = pd.MultiIndex.from_product([list(range(REPEATS)),list(range(FOLDS))])
            index = index.set_names(['repeat','fold'])
            self._resultsdf = pd.DataFrame(columns=pd.MultiIndex.from_product((self._resultsdf_col,self._metrics_name)),index=index)

    def report_score(self,test_pred, test_label, repeat, fold):

        for criterion in self._metrics:
            score = calculate_score(test_pred, test_label, criterion)
            # resultsdf.xs(criterion.name, axis=1, level=1, drop_level=False).iloc[fold] = score
            # print((repeat,fold))
            self._resultsdf.loc[(repeat,fold), pd.IndexSlice[:,criterion.name]] = score
        self._resultsdf.to_csv(self._run_dir+'/ResultSummary.csv',float_format='%.4f')
        return self._resultsdf

    def report_by_target(self):
        outtext_by_target = []
        for col in self._resultsdf.columns:
            mean, interval = compute_confidence_interval(self._resultsdf[col])
            outtext_by_target.extend((mean,interval))
        resultsdf_by_target = pd.DataFrame(columns=pd.MultiIndex.from_product((self._resultsdf_col,self._metrics_name,['mean','interval'])),index=[0])
        resultsdf_by_target.iloc[0] = outtext_by_target
        resultsdf_by_target = resultsdf_by_target.stack(0).droplevel(0) #.swaplevel(0,1)
        resultsdf_by_target = resultsdf_by_target.reindex(columns=product(self._metrics_name,['mean','interval']),index=self._resultsdf_col)
        resultsdf_by_target.to_csv(self._run_dir+'/ResultSummary_ByTarget.csv',float_format='%.4f')
        return resultsdf_by_target

    def get_dataframe(self):
        return self._resultsdf

def get_model(modelname,hyperparam_dict):

    if modelname == 'RandomForest':
        return RandomForestRegressor(
            bootstrap=hyperparam_dict['bootstrap'],
            max_depth=hyperparam_dict['max_depth'],
            min_samples_leaf=hyperparam_dict['min_samples_leaf'],
            min_samples_split=hyperparam_dict['min_samples_split'],
            n_estimators=hyperparam_dict['n_estimators'],
            max_features=hyperparam_dict['max_features'],
            n_jobs=-1
        )

    elif modelname == 'XGBoost':
        return xgb.XGBRegressor(
            predictor='gpu_predictor',
            verbosity=1,
            n_estimators=hyperparam_dict['n_estimators'], 
            learning_rate=hyperparam_dict['learning_rate'], 
            max_depth=hyperparam_dict['max_depth'], 
        )
    else:
        raise KeyError('Model name not found.')

def get_trial_config(trial,modelname):
    if modelname == 'RandomForest':
        return {
            'bootstrap': trial.suggest_categorical('bootstrap',[True, False]),
            'max_depth': trial.suggest_categorical('max_depth',[None, 20, 40, 60, 80, 100, 200]),
            'max_features': trial.suggest_categorical('max_features',['sqrt','auto']),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf',[1, 2, 5, 10, 20, 50]),
            'min_samples_split': trial.suggest_int('min_samples_split',2,50),
            'n_estimators': trial.suggest_categorical('n_estimators',[200, 400, 800, 1600, 3200, 6400]),
        }
    elif modelname =='XGBoost':
        return {
            'n_estimators':trial.suggest_int('n_estimators', 1, 400),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.5),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
        }

    else:
        raise KeyError('Model name not found.')

def create_kfold_dataset(tvset,folds,splitter,seed):
    os.makedirs(f'{RUN_DIR}/.temp/',exist_ok=True)
    directories = []
    for fold in range(folds):
        directories.extend([f'{RUN_DIR}/.temp/{fold}-train',f'{RUN_DIR}/.temp/{fold}-test'])
    for fold, (trainset, validset) in enumerate(splitter.k_fold_split(tvset,k=folds,seed=seed,directories=directories)):
        pass
def get_kfold_dataset(fold):
    trainset = DiskDataset(data_dir=f'{RUN_DIR}/.temp/{fold}-train')
    testset = DiskDataset(data_dir=f'{RUN_DIR}/.temp/{fold}-test')
    return trainset, testset

def del_tmpfolder():
    shutil.rmtree(f'{RUN_DIR}/.temp/')
class ModelWrapper(object):
    def __init__(self, model):
        self.model=model

    def fit(self,trainset):
        self.model.fit(trainset.X, trainset.y)

    def predict(self,testset):
        pred = self.model.predict(testset.X)
        pred = np.reshape(pred,(-1,1))
        label = np.reshape(testset.y, (-1,1))
        return pred, label

def get_best_trial(hyperrun_dir):
    with open(hyperrun_dir+'/besthyperparam.json','r') as jsonfile:
        best_trial_param = json.load(jsonfile)
    return best_trial_param
    
def model_tuning(trial,modelname,tvset,resume_flag=False):

    config = get_trial_config(trial,modelname)

    model = get_model(modelname,config)
    wrapper = ModelWrapper(model)
    splitter = RandomStratifiedSplitter()
    scores = []
    if not resume_flag:
        create_kfold_dataset(tvset,folds=FOLDS,seed=42,splitter=splitter)        
    for fold in range(FOLDS):
        trainset,validset = get_kfold_dataset(fold)
        wrapper.fit(trainset)
        valid_pred,valid_label = wrapper.predict(validset)
        valid_loss = calculate_score(valid_pred,valid_label, MSE())
        scores.append(valid_loss[0])
    mean_loss=np.mean(scores)
    del trainset, validset
    gc.collect()
    return mean_loss

def run_hyper_study(study_func, max_trials, study_name, hyperrun_dir):
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{hyperrun_dir}/hypertune.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(storage=storage, direction="minimize",study_name=study_name,load_if_exists=True)
    n_trials = max_trials - len(study.trials_dataframe())
    if n_trials != 0:
        study.optimize(study_func, n_trials=n_trials)
    trial = study.best_trial
    best_trial_param = dict()
    for key, value in trial.params.items():
        best_trial_param[key]=value
    with open(hyperrun_dir+'/besthyperparam.json','w') as jsonfile:
        json.dump(best_trial_param,jsonfile,indent=4)
    return study

class DrugOmicsNumpyDataset(data.Dataset):

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

    # def __getitem__(self, index):
        # cl_idx, drug_idx, _label = self.drugsens.iloc[index]
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

def main():

    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')
    print('Start time:',start_time_formatted)

    global RUN_DIR
    global fingfeaf, fingfeac
    global tvset
    global FOLDS, REPEATS
    global args, target, n_tasks
    global use_mut, use_expr, use_meth, use_cnv, modelname, classify


    model_names = ['XGBoost','RandomForest']
    parser = argparse.ArgumentParser(description='ablation_analysis')
    parser.add_argument('--modelname', dest='modelname', action='store', default = "AttentiveFP",
                        choices=model_names, help="AttentiveFP or GAT or GCN")
    parser.add_argument('--no_mut', dest='use_mut', default=True,
                        action="store_false", help='use gene mutation or not')
    parser.add_argument('--no_expr', dest='use_expr', default=True,
                        action="store_false", help='use gene expression or not')
    parser.add_argument('--no_methy', dest='use_meth', default=True,
                        action="store_false", help='use methylation or not')
    parser.add_argument('--no_cnv', dest='use_cnv', default=True,
                        action="store_false", help='use copy number variation or not')
    parser.add_argument('--no_drug', dest='use_drug', default=True,
                        action="store_false", help='use drug feature or not')
    parser.add_argument('-c','--classify', dest='classify', default=False,
                        action="store_true", help='flag this to perform classification task')
    parser.add_argument('-rs', '--resume', dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    parser.add_argument('-l', '--load_hyper', required=False,nargs=2, dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')

    args = parser.parse_args()

    classify = args.classify
    modelname = args.modelname
    print('modelname:',modelname)
    random.seed = 42
    resultfolder = './results' if not args.debug else './test'

    hyperpath = args.hyperpath
    hypertune_stop_flag = False if args.hyperpath is None else True
    testsplitname = r'random'
    hypersplitname = r'random'

    dataset_dir = "./data/"
    Mut_np = pd.read_csv(dataset_dir+'Mut_prep.csv', index_col=0, header=0).to_numpy()
    Expr_np = pd.read_csv(dataset_dir+'GeneExp_prep.csv', index_col=0, header=0).to_numpy()
    Meth_np = pd.read_csv(dataset_dir+'Meth_prep.csv', index_col=0, header=0).to_numpy()
    CNV_np = pd.read_csv(dataset_dir+'GeneCN_prep.csv', index_col=0, header=0).to_numpy()

    Omics_np = np.array([[]])

    if use_mut:
        Omics_np = np.concatenate(Omics_np, Mut_np, axis=1)
    if use_expr:
        Omics_np = np.concatenate(Omics_np, Expr_np, axis=1)
    if use_meth:
        Omics_np = np.concatenate(Omics_np, Meth_np, axis=1)
    if use_cnv:
        Omics_np = np.concatenate(Omics_np, CNV_np, axis=1)

    if classify:
         _drugsens_tv = pd.read_csv(dataset_dir + "DrugSens_train_CDG_binary.csv", index_col = 0, header = 0)
         _drugsens_test = pd.read_csv(dataset_dir + "DrugSens_test_CDG_binary.csv", index_col = 0, header = 0)
    else:
        _drugsens_tv = pd.read_csv(dataset_dir + "DrugSens_train_CDG.csv", index_col = 0, header = 0)
        _drugsens_test = pd.read_csv(dataset_dir + "DrugSens_test_CDG.csv", index_col = 0, header = 0)

    smiles_list, cl_list, drugsens = DrugsensTransform(pd.concat([_drugsens_tv, _drugsens_test], axis=0))
    drugsens_tv = drugsens.iloc[:len(_drugsens_tv)]
    drugsens_test = drugsens.iloc[len(_drugsens_tv):]

    model_dir = target_dir+'/'+modelname
    split_dir = model_dir+'/'+testsplitname
    os.makedirs(split_dir,exist_ok=True)
    hyperparamsplit_dir = model_dir+'/hyperparameters/'+hypersplitname

    if args.resume_run_dir is None:
        resume_flag = False
        database = args.database
        
        # Init new status
        status = StatusReport(hyperpath,database,hypertune_stop_flag)
        hypertune_stop_flag, ck_repeat, ck_fold, hyperpath, database = status.get_status()

    else:
        original_run = f'{split_dir}/{args.resume_run_dir}'
        print(f'Reload experiment: {original_run}')
        # SYS_LOGGER.info(f'RESUME RUN: {original_run}')
        resume_flag = True
        status = StatusReport.resume_run(original_run)
        hypertune_stop_flag, ck_repeat, ck_fold, hyperpath, database = status.get_status()

    if args.debug:
        max_trials = 1
        FOLDS=2
    else:
        max_trials = 30
        FOLDS=5
    splitter = RandomStratifiedSplitter()

    main_metric, report_metrics, prediction_task = (BCE(),[BCE(),AUROC(),AUCPR()],'clas') if classify \
                                             else (MSE(),[MSE(),RMSE(),PCC(),R2(),SRCC()],'regr')
    outtext_list = [database]

    # ------- Load Feature -------

    dataset = dc.data.NumpyDataset(X=X,y=y)
    tvset = dc.data.NumpyDataset(X=,y=y)

    tvset,testset = splitter.train_test_split(dataset,frac_train=0.8,seed=42)
    today_date = date.today().strftime('%Y-%m-%d')

    if hypertune_stop_flag == False:
            
        print('Optimizing hyperparameters...')

        if not resume_flag:
            num_hyperrun = 1
            hyperpath = f'{today_date}_HyperRun{num_hyperrun}'
            hyperrun_dir = os.path.join(hyperparamsplit_dir,hyperpath)
            while os.path.exists(hyperrun_dir):
                num_hyperrun+=1
                hyperpath = f'{today_date}_HyperRun{num_hyperrun}'
                hyperrun_dir = os.path.join(hyperparamsplit_dir,hyperpath)
            os.makedirs(hyperrun_dir)        
            RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{hyperpath}_TestRun1'
            os.makedirs(RUN_DIR)
                            # create status
            print('Create new experiment')
            status.set_run_dir(RUN_DIR)
            status.update({'hyperpath': hyperpath})
        else:
            RUN_DIR = original_run
        print(f'Your run directory is "{RUN_DIR}"')

        print('Start hyperparameters optimization')

        def model_tuning_simplified(trial):
            return model_tuning(trial=trial, tvset=tvset, modelname=modelname, resume_flag=resume_flag)
        study_name=f'{modelname}-{prediction_task}'
        hyperrun_dir = os.path.join(hyperparamsplit_dir,hyperpath)
        run_hyper_study(study_func=model_tuning_simplified,max_trials=max_trials,study_name=study_name,hyperrun_dir=hyperrun_dir)
        hyperparam = get_best_trial(hyperrun_dir)
        hypertune_stop_flag = True
        status.update({'hypertune_stop_flag':True})

    else: 

        print('Loading hyperparameters...')
        if not resume_flag:
            num_run = 1
            RUN_DIR = f'{split_dir}/{hyperpath}-{hypersplitname}_TestRun{num_run}'
            while os.path.exists(RUN_DIR):
                num_run+=1
                RUN_DIR = f'{split_dir}/{hyperpath}-{hypersplitname}_TestRun{num_run}'
            os.makedirs(RUN_DIR)
            # create status
            status.set_run_dir(RUN_DIR)
        else: 
            RUN_DIR = original_run
        print(f'Your run directory is "{RUN_DIR}"')
        hyperrun_dir = os.path.join(hyperparamsplit_dir,hyperpath)
        hyperparam = get_best_trial(hyperrun_dir)

    logging.basicConfig(level=5, filename=f'{RUN_DIR}/.log',filemode='a')

    print('- Done!')

    with open(RUN_DIR+'/hyperparameters.json','w') as jsonfile:
        json.dump(hyperparam,jsonfile,indent=4)
    outtext_list.insert(0,os.path.basename(RUN_DIR))

    results_report = ResultsReport(target, report_metrics, run_dir = RUN_DIR)

    # ------- Run Experiment -------

    for repeat in range(ck_repeat,REPEATS):
        print('====================')
        print(f'Repeat:{repeat+1}/{REPEATS}')
        print('====================')
        split_seed = set_split_seed()

        if not resume_flag:
            create_kfold_dataset(tvset,folds=FOLDS,seed=split_seed,splitter=splitter)

        for fold in range(ck_fold,FOLDS):

            status.update({
                'repeat':repeat,
                'fold':fold
            })

            trainset,_ = get_kfold_dataset(fold)

            print(f'Fold:{fold+1}/{FOLDS}')
            print('====================')
            seed = set_seed()
            
            round_dir = f'{RUN_DIR}/repeat_{repeat}-fold_{fold}-splitseed_{split_seed}'

            os.makedirs(round_dir)

            model = get_model(modelname,hyperparam)
            wrapper = ModelWrapper(model)

            wrapper.fit(trainset)
            train_pred,train_label = wrapper.predict(trainset)
            test_pred,test_label = wrapper.predict(testset)
            train_loss = calculate_score(train_pred, train_label, main_metric)
            test_loss = calculate_score(test_pred, test_label, main_metric)
            print(f'train loss: {train_loss[0]}, test loss: {test_loss[0]}')
            pt_dict = {
                'train_p':train_pred,
                'train_t':train_label,
                'test_p':test_pred,
                'test_t':test_label,
            }

            pkl.dump(pt_dict,open(round_dir+"/pred_true.pkl", "wb"))

            results_report.report_score(test_pred, test_label, repeat, fold)
            del trainset
            gc.collect()

        ck_fold=0

    resultsdf = results_report.get_dataframe()

    for col in resultsdf.columns:
        mean, interval = compute_confidence_interval(resultsdf[col])
        outtext_list.extend((mean,interval))

    # del_tmpfolder()

    end_time = arrow.now()
    end_time_formatted = end_time.format('DD/MM/YYYY HH:mm:ss')
    print('Finish time:',end_time_formatted)
    elapsed_time = end_time - start_time

    print('Writing output..')
    summaryfilepath = f'{resultfolder}/{modelname}/{prediction_task}/ResultSummarySheet.csv'
    if not os.path.exists(summaryfilepath):
        with open(summaryfilepath, 'w') as summaryfile:
            write_result_files(report_metrics,summaryfile)
    with open(summaryfilepath, 'a') as outfile:
        outtext_list.insert(1,start_time_formatted)
        outtext_list.insert(2,end_time_formatted)
        outtext_list.insert(3,str(elapsed_time).split('.')[0])
        output_writer = csv.writer(outfile,delimiter = ',')
        output_writer.writerow(outtext_list)
       
if __name__ == "__main__":
    main()
