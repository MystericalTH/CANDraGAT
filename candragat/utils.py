import torch
import random
import pandas as pd
from typing import Generator, Sequence
from sklearn.model_selection import train_test_split, KFold
import csv
import numpy as np
from itertools import product
import json
import os
import shutil
import errno
import re
import logging
import io
import tqdm
import time
import scipy.stats as st

def set_base_seed(seed=None):        
    random.seed(seed)

def set_seed(seed=None):
    if seed is None:
        seed = random.randrange(1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return seed

def df_kfold_split(df, n_splits=5, seed: int = None) -> Generator[pd.DataFrame, pd.DataFrame, None]:
    assert type(df) == pd.DataFrame
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    list_idx = list(range(len(df)))
    for i, (train_idx, test_idx) in enumerate(kfold.split(list_idx)):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        print('len df train, test: ',len(df_train),len(df_test))
        yield df_train, df_test
    
def df_train_test_split(df: pd.DataFrame, train_size=0.8) -> Sequence[pd.DataFrame]:
    list_idx = list(range(len(df)))
    train_idx, test_idx = train_test_split(list_idx,train_size=train_size)
    return df.iloc[train_idx], df.iloc[test_idx]

def compute_confidence_interval(data, confidence=0.95):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    confidence = confidence/2+0.5
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = round(st.norm.ppf(confidence),4) * (std / np.sqrt(len(a)))
    return m, pm

def write_result_files(report_metrics,summaryfile):
    col = ['experiment_name','study_name','data_name','note','start_time','end_time','elapsed_time','mutation', 'gene_expression', 'methylation', 'copy_number_variation','drug']
    writer = csv.writer(summaryfile)
    multiheader = [(x,None) for x in col] + list(product([x.name for x in report_metrics],['mean','interval']))
    for i in range(2):
        writer.writerow([n[i] for n in multiheader])

def graph_collate_fn(batch):
    Output = [torch.Tensor()]*len(batch[0])
    for Input in batch:
        for i, x in enumerate(Input):
            Output[i] = torch.cat((Output[i], x))
    return Output

class StatusReport(object):

    def __init__(self,hypertune_stop_flag=False, trial=0, repeat=0, fold=0, epoch=0, run_dir=None):
        self._run_dir = run_dir
        self._status = {
            'hypertune_stop_flag':hypertune_stop_flag,
            'trial':trial,
            'repeat':repeat,
            'fold':fold,
            'epoch':epoch,
        }

    def set_run_dir(self,run_dir):
        self._run_dir = run_dir
        with open(f'{self._run_dir}/status.json','w') as status_file:
            json.dump(self._status,status_file,indent=4)

    @classmethod
    def resume_run(cls,run_dir):
        with open(f'{run_dir}/status.json','r') as status_file:
            status = json.load(status_file)
        return cls(status['hypertune_stop_flag'], status['repeat'], status['fold'], status['epoch'],run_dir=run_dir)

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
        self.LowerThanMaxLimit = 10
        self.DecreasingLimit = 5
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

def DrugInputToDevice(DrugInput, modelname,DEVICE):
    if modelname in ("AttentiveFP","FragAttentiveFP"):
        return [tensor.to(DEVICE) for tensor in DrugInput]
    elif modelname in ("GAT", "GCN"):
        return DrugInput.to(DEVICE)

class MyLogging(object):
    filename = 'output.log'
    logger_names = []
    logging_level = logging.INFO
    group_id = ""
    def __init__(self) -> None:
        pass
    @classmethod
    def getLogger(cls, name, filename: str = None, logging_level: int = None):
        if filename is None:
            filename = cls.filename
        if logging_level is None:
            logging_level = cls.logging_level
            
        if name[:4] != MyLogging.group_id:
            name = MyLogging.group_id + '.' + name
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger = logging.getLogger(name)
        if name not in cls.logger_names:
            logger.setLevel(logging_level)

            fhdlr = logging.FileHandler(filename)
            formatter = logging.Formatter("%(asctime)-15s.%(msecs)03d %(name)-20s %(levelname)-8s %(message)s",
                                        "%Y-%m-%d %H:%M:%S")
            fhdlr.setFormatter(formatter)
            logger.addHandler(fhdlr)
            cls.logger_names.append(name)
        return logger

    @classmethod
    def set_args(cls, args):
        cls.filename = args['log']
        cls.logging_level = args['logging_level']
        
    @staticmethod
    def log_task(message, *out_args, **out_kwargs):
        def inner(func):
            def wrapper(*args, **kwargs):
                logger = args[0].logger
                logger.info(message)
                output = func(*args, **kwargs)
                logger.info(message + f" - DONE!")
                return output
            return wrapper
        return inner

    @classmethod
    def setGroupID(cls, group_id):
        cls.group_id = group_id
        
class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.\n
        From https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
    """
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip()
    def flush(self):
        self.logger.log(self.level, self.buf)
