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

MyLogging.setGroupID('candragat')

def parse_args():
    model_names = ["AttentiveFP", "GAT", "GCN","FragAttentiveFP"]
    parser = ArgumentParser(description='ablation_analysis')
    parser.add_argument('--modelname', dest='modelname', action='store', default = "FragAttentiveFP",
                        choices=model_names, help="AttentiveFP or GAT or GCN")
    parser.add_argument('--task', dest='task', default='regr',
                        action="store", help='classification or regression (default: \'regr\')', choices=['clas','regr'])
    parser.add_argument('--enable', dest='enable', default="mxdv", 
                        help='feature(s) to enable; \nm = mutation, x = gene expression, d = DNA methylation, \n' + 
                             'v = copy number variation')
    parser.add_argument('--disable_drug', dest='enable_drug', default=True,
                        action="store_false", help='enable drug feature or not')
    parser.add_argument('--note', dest='note', default='',
                        action="store", help='a string of note to display in summary file')
    parser.add_argument('-rs', '--resume', dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument("--job-id", dest="job_id", type=int, help="job id")
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    parser.add_argument('-l', '--load_hyper', required=False,nargs=2, dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')

    args = vars(parser.parse_args())

    return args


def main():
    args = parse_args()
    

    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')

    set_base_seed(42)
    task = args['task']
    modelname = args['modelname']
    note = args['note']

    CPU = torch.device('cpu')
    CUDA = torch.device('cuda')
    
    resultfolder = './results' if not args['debug'] else './test'
    hypertune_stop_flag = False
    
    # ----- Setup ------
    mainmetric, report_metrics, prediction_task = (BCE(),[BCE(),AUROC(),AUCPR()],task) if task == 'clas' \
                                             else (MSE(),[MSE(),RMSE(),PCC(),R2(),SRCC()],task)
    criterion = mainmetric # NOTE must alert (changed from nn.MSELoss to mainmetric)

    # ------- Storage Path -------
    today_date = date.today().strftime('%Y-%m-%d')
    if args['hyperpath'] is None:
        hyperexpdir = f'{resultfolder}/{modelname}/hyperparameters/'
        os.makedirs(hyperexpdir,exist_ok=True)
        num_hyperrun = 1
        while os.path.exists(hyperexpdir+f'{today_date}_HyperRun{num_hyperrun}.json'):
            num_hyperrun+=1
        else:
            hyperexpname = f'{today_date}_HyperRun{num_hyperrun}' 
            json.dump({},open(hyperexpdir+hyperexpname+'.json','w')) # placeholder file
        RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun1'
        exp_name=f'{hyperexpname}_TestRun1'
        
    else: 
        hypertune_stop_flag = True
        hyper_modelname,hyperexpname= args['hyperpath']
        hyper_jsondir = f'{resultfolder}/{hyper_modelname}/hyperparameters/{hyperexpname}.json'
        pt_param = get_best_trial(hyper_jsondir)
        num_run = 1
        while os.path.exists(f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'):
            num_run+=1
        RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'
        exp_name=f'{hyperexpname}_TestRun{num_run}'
        
    mainlogger = MyLogging.getLogger("main", filename=f'{RUN_DIR}/main.log') 
    pbarlogger = MyLogging.getLogger('pbar', filename=f'{RUN_DIR}/pbar.log')
    
    mainlogger.info(f"SLURM ID: {args['job_id']}")
    mainlogger.info(f'Start time: {start_time_formatted}')

    try: 
        os.makedirs(RUN_DIR, exist_ok=True)
        status = StatusReport(hyperexpname,hypertune_stop_flag)
        status.set_run_dir(RUN_DIR)
        mainlogger.info(f'Your run directory is "{RUN_DIR}"')
        
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
        mainlogger.info('Omics data loaded.')
        
        _drugsens_tv = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-Train.csv", header = 0)
        _drugsens_test = pd.read_csv(drugsens_dir + '/' + args['task'] + '/' + "DrugSens-Test.csv", header = 0)
        smiles_list, cl_list, drugsens = DrugSensTransform(pd.concat([_drugsens_tv, _drugsens_test], axis=0))
        drugsens_tv = drugsens.iloc[:len(_drugsens_tv)]
        drugsens_test = drugsens.iloc[len(_drugsens_tv):]

        omics_dataset = OmicsDataset(cl_list, mut = Mut_df, expr = Expr_df, meth = Meth_df, cnv = CNV_df)
        outtext_list = [enable_mut, enable_expr, enable_meth, enable_cnv, enable_drug]

        if args['debug']:
            mainlogger.info('-- DEBUG MODE --')
            n_trials = 1
            max_tuning_epoch = 1
            max_epoch = 3
            folds=2
            drugsens_tv = drugsens_tv.iloc[:1000]
            drugsens_test = drugsens_test.iloc[:100]
            batch_size = 16
        else:
            n_trials = 30
            max_tuning_epoch = 3
            max_epoch = 30
            folds=5
            batch_size = 64
        
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
            

        DatasetTest = DrugOmicsDataset(drugsens_test, omics_dataset, smiles_list, modelname, EVAL = True)
        testloader = get_dataloader(DatasetTest, modelname)
        
        if args['hyperpath'] is None:
            mainlogger.info('Start hyperparameters optimization')
            def candragat_tuning_simplified(trial):
                return candragat_tuning(trial, drugsens_tv, omics_dataset, smiles_list, modelname, status, batch_size, mainlogger, pbarlogger, args, max_tuning_epoch)
            study_name=f'{modelname}-{task}'
            run_hyper_study(study_func=candragat_tuning_simplified, n_trials=n_trials,hyperexpfilename=hyperexpdir+hyperexpname, study_name=study_name)
            pt_param = get_best_trial(hyperexpdir+hyperexpname)
            hypertune_stop_flag = True
            status.update({'hypertune_stop_flag':True})
            mainlogger.info('Hyperparameters optimization done')

        outtext_list.insert(0,exp_name)
        
        drop_rate = pt_param['drop_rate']
        lr = pt_param['lr']
        weight_decay = pt_param['weight_decay']
        omics_output_size = pt_param['omics_output_size']
        drug_output_size = pt_param['drug_output_size']
        report_metrics_name = [metric.name for metric in report_metrics]
        resultsdf = pd.DataFrame(columns=report_metrics_name,index=list(range(folds)))

        #############################
        
        
        for fold, (Trainset, Validset) in enumerate(df_kfold_split(drugsens_tv, n_splits=folds),start=0): 
            mainlogger.info(f'=============== Fold {fold+1}/{folds} ===============')

            seed = set_seed(100)
            mainlogger.info(f'-- TRAIN SET {fold+1} --')

            DatasetTrain = DrugOmicsDataset(Trainset, omics_dataset, smiles_list, modelname, EVAL = False)
            DatasetValid = DrugOmicsDataset(Validset, omics_dataset, smiles_list, modelname, EVAL = True)
            trainloader = get_dataloader(DatasetTrain, modelname, batch_size=batch_size)
            validloader = get_dataloader(DatasetValid, modelname)

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
                    input_size_list=omics_dataset.input_size,
                    args = args,
                )
                
            if optimizer is None:
                optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

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
                    for ii, Data in enumerate(pbar): 
                        if stop_flag:
                            break
                        
                        with torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ],
                                with_stack=True
                            ) as p:
                            [OmicsInput, DrugInput], Label = Data
                            DrugInput = DrugInputToDevice(DrugInput, modelname, CUDA)
                            OmicsInput = [tensor.to(CUDA) for tensor in OmicsInput]
                            # Label = Label.squeeze(-1).to(CUDA)    # [batch, task]
                            
                        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
                        output = model([OmicsInput, DrugInput])    # [batch, output_size]

                        loss = criterion(output.to(CPU), Label.to(CPU), requires_backward = True)
                        
                        cum_loss += loss.detach()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if (ii+1) % 20 == 0:
                            torch.cuda.empty_cache()
                        pbar.set_postfix({'loss':loss.item()}, refresh=False)

                time_end = datetime.now()
                duration_epoch = (time_end-time_start).total_seconds()
                trainmeanloss = (cum_loss/len(trainloader)).item()
                mainlogger.info(f"Epoch duration = {duration_epoch} seconds")
                mainlogger.info(f"Train loss on Epoch {num_epoch} = {trainmeanloss}")

                validresult = Validation(validloader, model, report_metrics, modelname, mainlogger, pbarlogger, CPU)[0]
                validmeanloss = validresult[mainmetric.name]

                mainlogger.info('===========================================================')
                stop_flag, _, _ = saver.SaveModel(model, optimizer, num_epoch, validresult, mainmetric)
                validloss.append(validmeanloss)
                trainloss.append(trainmeanloss)

            torch.cuda.empty_cache()

            bestmodel = saver.LoadModel()
            score, predIC50, labeledIC50 = Validation(testloader, bestmodel, report_metrics, modelname, mainlogger, pbarlogger, CPU)

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
            outtext_list.insert(1,note)
            outtext_list.insert(2,start_time_formatted)
            outtext_list.insert(3,end_time_formatted)
            outtext_list.insert(4,str(elapsed_time).split('.')[0])
            output_writer = csv.writer(outfile,delimiter = ',')
            output_writer.writerow(outtext_list)
    except:
        mainlogger.exception('Error occured', exc_info=True)
    mainlogger.info("Exiting at {}".format(arrow.now().format('DD/MM/YYYY HH:mm:ss')))
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()

