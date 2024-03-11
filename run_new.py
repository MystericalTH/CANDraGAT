from candragat import *

import arrow
import torch
from torch import nn, optim
from datetime import datetime, date
import csv
import pandas as pd
import numpy as np
import json
import os
from argparse import ArgumentParser
import pickle as pkl
MyLogging.setGroupID('candragat')

def parse_args():
    model_names = ["AttentiveFP", "GAT", "GCN","FragAttentiveFP"]
    parser = ArgumentParser(description='ablation_analysis')
    parser.add_argument('configs', help="JSON configs file under comfigs/ folder")
    parser.add_argument('--modelname', dest='modelname', action='store', default = "FragAttentiveFP",
                        choices=model_names, help="AttentiveFP or GAT or GCN")
    parser.add_argument('--task', dest='task', default='regr',
                        action="store", help='classification or regression (default: \'regr\')', choices=['clas','regr'])
    parser.add_argument('--data', dest='data', default='normal',
                        action="store", help='normal, blind cell, or blind drug data (default: \'normal\')', choices=['normal','blind-cell','blind-drug'])
    parser.add_argument('--enable', dest='enable', default="mxdv", 
                        help='feature(s) to enable; \nm = mutation, x = gene expression, d = DNA methylation, \n' + 
                             'v = copy number variation')
    parser.add_argument('--disable_drug', dest='enable_drug', default=True,
                        action="store_false", help='enable drug feature or not')
    parser.add_argument('--note', dest='note', default='',
                        action="store", help='a string of note to display in summary file')
    parser.add_argument('-rs', '--resume', dest='hyperpath', nargs=2 , help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument("--job-id", dest="job_id", type=int, help="job id")
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    parser.add_argument('-l', '--load_hyper', required=False,nargs=2, dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument('-skip', '--skip_hyper', required=False, dest='skip_hyper', help='skip hyperparameter tuning', default=False, action="store_true")

    args = vars(parser.parse_args())

    return args

def main():
    args = parse_args()
    configs = json.load(open(os.path.join("configs", args['configs']),'r'))
    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')

    # set_base_seed(42)
    torch.backends.cudnn.deterministic = True
    
    task = args['task']
    modelname = args['modelname']
    dataname = args['data']
    note = args['note']

    CPU = torch.device('cpu')
    CUDA = torch.device('cuda')
    
    resultfolder = './results' if not args['debug'] else './test'
    hypertune_stop_flag = False
    
    # ---------- Setup ----------
    mainmetric, report_metrics, prediction_task = (BCE(),[BCE(),AUROC(),AUCPR(),ACC(),BACC(),SPEC(),PREC(),REC(),F1(),KAPPA(),MCC(),],task) if task == 'clas' \
                                             else (MSE(),[MSE(),RMSE(),MAE(),PCC(),R2(),SRCC()],task)
    criterion = mainmetric # NOTE must alert (changed from nn.MSELoss to mainmetric)

    # ------- Storage Path -------
    today_date = date.today().strftime('%Y-%m-%d')
    pt_param = None
    if args['hyperpath'] is not None:
        hypertune_stop_flag = True
        hyper_modelname,hyperexpname= args['hyperpath']
        hyper_jsondir = f'{resultfolder}/{hyper_modelname}/hyperparameters/{hyperexpname}.json'
        pt_param = get_best_trial(hyper_jsondir)
        num_run = 1
        while os.path.exists(f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'):
            num_run+=1
        exp_name=f'{hyperexpname}_TestRun{num_run}'
    elif "hp" in configs:
        hyperexpdir = f'{resultfolder}/{modelname}/hyperparameters/'
        os.makedirs(hyperexpdir,exist_ok=True)
        pt_param = configs['hp']
        num_hyperrun = 1
        while os.path.exists(hyperexpdir+f'{today_date}_HyperRun{num_hyperrun}.json'):
            num_hyperrun+=1
        hyperexpname = f'{today_date}_HyperRun{num_hyperrun}' 
        exp_name = f'{hyperexpname}_TestRun1'
        json.dump(pt_param,open(hyperexpdir+hyperexpname+'.json','w'), indent=2) # placeholder file
    else: 
        hyperexpdir = f'{resultfolder}/{modelname}/hyperparameters/'
        os.makedirs(hyperexpdir,exist_ok=True)
        num_hyperrun = 1
        while os.path.exists(hyperexpdir+f'{today_date}_HyperRun{num_hyperrun}.json'):
            num_hyperrun+=1
        else:
            hyperexpname = f'{today_date}_HyperRun{num_hyperrun}' 
            json.dump({},open(hyperexpdir+hyperexpname+'.json','w'), indent=2) # placeholder file
        exp_name=f'{hyperexpname}_TestRun1'
        
    status = StatusReport(hyperexpname,hypertune_stop_flag)
    RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{exp_name}'
    mainlogger = MyLogging.getLogger("main", filename=f'{RUN_DIR}/main.log') 
    pbarlogger = MyLogging.getLogger('pbar', filename=f'{RUN_DIR}/pbar.log')
    os.makedirs(RUN_DIR, exist_ok=True)
    status.set_run_dir(RUN_DIR)
    mainlogger.info(f'Your run directory is "{RUN_DIR}"')
    configs["note"] = args["note"]
    with open(os.path.join(RUN_DIR, "configs.json"), "w") as file:
        json.dump(configs, file, indent=2)
    
    mainlogger.info(f"SLURM ID: {args['job_id']}")
    mainlogger.info(f'Start time: {start_time_formatted}')

    try: 
        
        # ------ Loading Dataset  ------
        mainlogger.info('Loading dataset...')

        mainlogger.info('-- TEST SET --')
        enable_mut, enable_expr, enable_meth, enable_cnv = args['enable_mut'], args['enable_expr'], args['enable_meth'], args['enable_cnv'] = [char in args['enable'] for char in 'mxdv']
        enable_drug = args['enable_drug']
        features_dir = "./data/datasets/features"
        drugsens_dir = "./data/datasets/sensitivity/stack"
        Mut_df = pd.read_csv(features_dir + '/' + 'Mut.csv', index_col=0, header=0) 
        Expr_df = pd.read_csv(features_dir + '/' + 'GeneExp.csv', index_col=0, header=0) 
        Meth_df = pd.read_csv(features_dir + '/' + 'Meth.csv', index_col=0, header=0) 
        CNV_df = pd.read_csv(features_dir + '/' + 'GeneCN.csv', index_col=0, header=0)
        
        if dataname == 'normal':
            _hyper_trainset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-Trainhyper-Subsampling.csv', header=0)#.iloc[:, :3] 
            _hyper_validset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-Validhyper-Subsampling.csv', header=0)#.iloc[:, :3] 
        elif dataname == 'blind-cell':
            _hyper_trainset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-BlindCellLine-Trainhyper-Subsampling.csv', header=0)#.iloc[:, 3:] 
            _hyper_validset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-BlindCellLine-Validhyper-Subsampling.csv', header=0)#.iloc[:, 3:]
        else:
            _hyper_trainset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-BlindDrug-Trainhyper-Subsampling.csv', header=0)#.iloc[:, 3:] 
            _hyper_validset = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + 'DrugSens-BlindDrug-Validhyper-Subsampling.csv', header=0)#.iloc[:, 3:]
        mainlogger.info('Omics data loaded.')
        
        if dataname == 'normal':
            _drugsens_tv = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-Train.csv", header = 0)
            _drugsens_test = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-Test.csv", header = 0)
        elif dataname == 'blind-cell':
            _drugsens_tv = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-BlindCellLine-Train.csv", header = 0)
            _drugsens_test = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-BlindCellLine-Test.csv", header = 0)
        else:
            _drugsens_tv = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-BlindDrug-Train.csv", header = 0)
            _drugsens_test = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-BlindDrug-Test.csv", header = 0)
        
        smiles_list, cl_list, drugsens = DrugSensTransform(pd.concat([_drugsens_tv, _drugsens_test], axis=0))
        drugsens_tv = drugsens.iloc[:len(_drugsens_tv)]
        drugsens_test = drugsens.iloc[len(_drugsens_tv):]
        
        hp_smiles_list, hp_cl_list, hp_drugsens = DrugSensTransform(pd.concat([_hyper_trainset, _hyper_validset], axis=0))
        hyper_trainset = hp_drugsens.iloc[:len(_hyper_trainset)]
        hyper_validset = hp_drugsens.iloc[len(_hyper_trainset):]

        hp_omics_dataset = OmicsDataset(hp_cl_list, mut = Mut_df, expr = Expr_df, meth = Meth_df, cnv = CNV_df)
        omics_dataset = OmicsDataset(cl_list, mut = Mut_df, expr = Expr_df, meth = Meth_df, cnv = CNV_df)
        outtext_list = [enable_mut, enable_expr, enable_meth, enable_cnv, enable_drug]

        study_attrs = {}
        n_trials = configs['n_trials']
        max_tuning_epoch = configs['max_tuning_epoch']
        max_epoch = configs['max_epoch']
        folds = configs['folds']
        batch_size = configs['batch_size']
        
        if configs['dataset_debug']:
            mainlogger.info('-- DEBUG MODE --')
            drugsens_tv = drugsens_tv.iloc[:configs['drugsens_tv_num_items']]
            drugsens_test = drugsens_test.iloc[:configs['drugsens_test_num_items']]
            hyper_trainset = hyper_trainset.iloc[:200]
            hyper_validset = hyper_validset.iloc[:200]
            mainlogger.info(f"drugsens_tv size: {len(drugsens_tv)}")
            mainlogger.info(f"drugsens_test size: {len(drugsens_test)}")
            mainlogger.info(f"hyper_trainset size: {len(hyper_trainset)}")
            mainlogger.info(f"hyper_validset size: {len(hyper_validset)}")
        
        if enable_drug:
            mainlogger.info('Use Drug')
        else:
            mainlogger.info('No Drug')
        if enable_expr:
            mainlogger.info('Use Gene Exp')
        else:
            mainlogger.info('No Gene Exp')
        if enable_mut:
            mainlogger.info('Use Gene Mut')
        else:
            mainlogger.info('No Gene Mut')
        if enable_cnv:
            mainlogger.info('Use CNV')
        else:
            mainlogger.info('No CNV')
        if enable_meth:
            mainlogger.info('Use DNA Methy')
        else:
            mainlogger.info('No DNA Methy')
        
        if args['task'] == 'clas':
            weight_dict = pkl.load(open("data/datasets/weight_dict.pkl", "rb"))
        else:
            weight_dict = None
        feat_prefix = ''.join([char for char, enable in zip('mxdvD', [enable_mut, enable_expr, enable_meth, enable_cnv, enable_drug]) if enable])
        study_name=f'{modelname}:{task}:{feat_prefix}:{dataname}'
        data_name=f'{dataname}'
        mainlogger.info(f'Study name: "{study_name}"')
        mainlogger.info(f'Data name: "{data_name}"')

        DatasetTest = DrugOmicsDataset(drugsens_test, omics_dataset, smiles_list, modelname, EVAL = True, root = os.path.join(RUN_DIR, 'drug-tensors-test'))
        testloader = get_dataloader(DatasetTest, modelname, batch_size=8)
        
        mainlogger.info('Hyperparameters optimization')
        study_attrs['model'] = modelname
        study_attrs['task'] = dict(zip(['clas','regr'], ['classification', 'regression']))[prediction_task]
        for key, feature in zip(['mutation', 'gene expression', 'DNA methylation', 'copy number variation', 'drug'], 
                                ['mut','expr','meth','cnv','drug']):
            study_attrs[key] = args['enable_'+feature]
        if pt_param is None:
            if not args['skip_hyper']:
                def candragat_tuning_simplified(trial):
                    return candragat_tuning(trial, hyper_trainset, hyper_validset, hp_omics_dataset, hp_smiles_list, modelname, status, batch_size, mainlogger, pbarlogger, args, RUN_DIR, criterion, max_tuning_epoch, weight=weight_dict)
                run_hyper_study(study_func=candragat_tuning_simplified, N_TRIALS=n_trials,hyperexpfilename=hyperexpdir+hyperexpname, study_name=study_name, study_attrs=study_attrs,result_folder=resultfolder)
            pt_param = get_best_trial(study_name, result_folder=resultfolder)
            json.dump(pt_param,open(RUN_DIR+'/hp.json','w'), indent=2)
        hypertune_stop_flag = True
        status.update({'hypertune_stop_flag':True})

        outtext_list.insert(0,exp_name)
        
        drop_rate = pt_param['drop_rate']
        lr = pt_param['lr']
        omics_output_size = pt_param['omics_output_size']
        drug_output_size = pt_param['drug_output_size']
        report_metrics_name = [metric.name for metric in report_metrics]
        resultsdf = pd.DataFrame(columns=report_metrics_name,index=list(range(folds)))

        #############################
        
        
        for fold, (Trainset, Validset) in enumerate(df_kfold_split(drugsens_tv, n_splits=folds, seed=42),start=0): 
            mainlogger.info(f'=============== Fold {fold+1}/{folds} ===============')

            seed = set_seed()
            mainlogger.info(f'-- TRAIN SET {fold+1} --')

            fold_dir = os.path.join(RUN_DIR, f'fold_{fold}-seed_{seed}')
            os.makedirs(f'{fold_dir}/.debug/', exist_ok=True)
                
            DatasetTrain = DrugOmicsDataset(Trainset, omics_dataset, smiles_list, modelname, EVAL = False, weight=weight_dict)
            DatasetValid = DrugOmicsDataset(Validset, omics_dataset, smiles_list, modelname, EVAL = True, root = os.path.join(fold_dir, '.drug-tensors'))
            trainloader = get_dataloader(DatasetTrain, modelname, batch_size=batch_size)
            validloader = get_dataloader(DatasetValid, modelname, batch_size=8)

            saver = Saver(fold_dir, max_epoch)
            model, optimizer = saver.LoadModel(load_all=True)

            if model is None:
                drug_model = get_drug_model(modelname,pt_param)
                mainlogger.info(f'{drug_model}')
                model = MultiOmicsMolNet(
                    drug_nn=drug_model,
                    drug_output_size=drug_output_size,
                    omics_output_size=omics_output_size,
                    drop_rate=drop_rate,
                    input_size_list=omics_dataset.input_size,
                    args = args,
                )
                
            if optimizer is None:
                optimizer = optim.Adam(model.parameters(), lr=lr)

            torch.cuda.empty_cache()
            validloss = []
            trainloss = []

            optimizer.zero_grad()

            mainlogger.info("Start Training...")
            stop_flag = False

            for num_epoch in range(1,max_epoch+1):
                model.train().to(CUDA)
                time_start = datetime.now()
                cum_loss = 0.0
                mainlogger.info(f'Epoch:{num_epoch}/{max_epoch}')
                status.update({
                        'fold':fold,
                        'epoch':num_epoch
                    })
                if stop_flag:
                    break
                
                with tqdm.tqdm(trainloader, total=len(trainloader), file=TqdmToLogger(pbarlogger,level=logging.INFO), 
                               mininterval=1, desc=f'Epoch {num_epoch}/{max_epoch} - Train') as pbar:
                    All_answer = []
                    All_label = []
                    for ii, Data in enumerate(pbar): 
                        if stop_flag:
                            break
                        
                        [OmicsInput, DrugInput], Label = Data[:2]
                            
                        DrugInput = DrugInputToDevice(DrugInput, modelname, CUDA)
                        OmicsInput = [tensor.to(CUDA) for tensor in OmicsInput]
                            # Label = Label.squeeze(-1).to(CUDA)    # [batch, task]
                            
                        output = model([OmicsInput, DrugInput])    # [batch, output_size]
                        
                        if weight_dict is None:
                            loss = criterion(output.to(CPU), Label.to(CPU), requires_backward = True)
                        else:
                            loss = criterion(output.to(CPU), Label.to(CPU), weight=Data[2], requires_backward = True)
                        
                        cum_loss += loss.detach()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if (ii+1) % 20 == 0:
                            torch.cuda.empty_cache()
                        pbar.set_postfix({'loss':loss.item()}, refresh=False)
                        All_answer.append(output.detach().cpu())
                        All_label.append(Label.detach())
                    All_answer = torch.cat(All_answer, dim=0).squeeze()
                    All_label = torch.cat(All_label, dim=0).squeeze()
                    All_answer = All_answer.tolist()
                    All_label = All_label.tolist()
                    np.savez(f'{fold_dir}/.debug/epoch{num_epoch}-train-IC50.npz', predIC50=All_answer, labeledIC50=All_label)

                time_end = datetime.now()
                duration_epoch = (time_end-time_start).total_seconds()
                trainmeanloss = (cum_loss/len(trainloader)).item()
                mainlogger.info(f"Epoch duration = {duration_epoch} seconds")
                mainlogger.info(f"Train loss on Epoch {num_epoch} = {trainmeanloss}")

                validresult, predIC50, labeledIC50 = Validation(validloader, model, report_metrics, modelname, mainlogger, pbarlogger, CUDA)
                np.savez(f'{fold_dir}/.debug/epoch{num_epoch}-valid-IC50.npz', predIC50=predIC50, labeledIC50=labeledIC50)
                validmeanloss = validresult[mainmetric.name]

                mainlogger.info('===========================================================')
                stop_flag, _, _ = saver.SaveModel(model, optimizer, num_epoch, validresult, mainmetric)
                validloss.append(validmeanloss)
                trainloss.append(trainmeanloss)

            torch.cuda.empty_cache()

            bestmodel = saver.LoadModel()
            mainlogger.info('TEST SET')
            score, predIC50, labeledIC50 = Validation(testloader, bestmodel, report_metrics, modelname, mainlogger, pbarlogger, CUDA)
                
            for metric in score:
                resultsdf.loc[fold,metric] = score[metric]

            traintestloss = np.array([trainloss, validloss])
            np.savetxt(fold_dir+'/traintestloss.csv', traintestloss, delimiter=',', fmt='%.5f')

            # create text file for test
            testpredlabel = np.array([predIC50, labeledIC50]).T
            np.savetxt(f'{fold_dir}/testIC50.csv', testpredlabel, delimiter=',', fmt='%.5f')
            np.savez(f'{fold_dir}/testIC50.npz', predIC50=predIC50, labeledIC50=labeledIC50)

        for col in resultsdf.columns:
            mean, interval = compute_confidence_interval(resultsdf[col])
            outtext_list.extend((round(mean, 4), round(interval, 4)))

        end_time = arrow.now()
        end_time_formatted = end_time.format('DD/MM/YYYY HH:mm:ss')
        mainlogger.info('===========================================================')
        mainlogger.info(f'Finish time: {end_time_formatted}')
        elapsed_time = end_time - start_time

        mainlogger.info('Writing Output...')

        resultsdf.to_csv(f'{RUN_DIR}/ExperimentSummary.csv', index=False)

        summaryfilepath = f'{resultfolder}/{modelname}/{prediction_task}/ResultSummarySheet.csv'
        if not os.path.exists(summaryfilepath):
            with open(summaryfilepath, 'w') as summaryfile:
                write_result_files(report_metrics,summaryfile)
        with open(summaryfilepath, 'a') as outfile:
            outtext_list.insert(1,study_name)
            outtext_list.insert(2,data_name)
            outtext_list.insert(3,note)
            outtext_list.insert(4,start_time_formatted)
            outtext_list.insert(5,end_time_formatted)
            outtext_list.insert(6,str(elapsed_time).split('.')[0])
            output_writer = csv.writer(outfile,delimiter = ',')
            output_writer.writerow(outtext_list)
        # if args['debug']:
        #     delete_study(study_name, result_folder=resultfolder)
    except:
        mainlogger.exception('Error occured', exc_info=True)
    mainlogger.info("Exiting at {}".format(arrow.now().format('DD/MM/YYYY HH:mm:ss')))
        
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

