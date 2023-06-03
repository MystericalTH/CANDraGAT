from candragat import *

import argparse
import arrow
import torch
from torch import nn, optim
from datetime import datetime, date
import csv
import pandas as pd
import numpy as np
import json
import os
import parser

def main():

    start_time = arrow.now()
    start_time_formatted = start_time.format('DD/MM/YYYY HH:mm:ss')
    print('Start time:',start_time_formatted)

    set_base_seed(42)

    global ckpt_dir, max_epoch, max_tuning_epoch, device
    global use_mut, use_expr, use_meth, use_cnv, modelname, classify
    global status

    model_names = ["AGMI", "GraphDRP"]
    parser = argparse.ArgumentParser(description='ablation_analysis')
    parser.add_argument('--modelname', dest='modelname', action='store', default = "AttentiveFP",
                        choices=model_names, help="select model")
    parser.add_argument('-c','--classify', dest='classify', default=False,
                        action="store_true", help='flag this to perform classification task')
    # parser.add_argument('-rs', '--resume', dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument('--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    args = parser.parse_args()

    classify = args.classify
    modelname = args.modelname

    CPU = torch.device('cpu')
    CUDA = torch.device('cuda')
    

    # ------ Loading Dataset  ------

    # use_mut, use_expr, use_meth, use_cnv = args.use_mut, args.use_expr, args.use_meth, args.use_cnv
    dataset_dir = "./data/"
    # Mut_df = pd.read_csv(dataset_dir+'Mut_prep.csv', index_col=0, header=0) 
    # if not use_mut: 
    #     Mut_df = pd.DataFrame(0, index=np.arange(len(Mut_df)), columns=Mut_df.columns)
    # Expr_df = pd.read_csv(dataset_dir+'GeneExp_prep.csv', index_col=0, header=0) 
    # if not use_expr: 
    #     Expr_df = pd.DataFrame(0, index=np.arange(len(Expr_df)), columns=Expr_df.columns)
    # Meth_df = pd.read_csv(dataset_dir+'Meth_prep.csv', index_col=0, header=0) 
    # if not use_meth: 
    #     Meth_df = pd.DataFrame(0, index=np.arange(len(Meth_df)), columns=Meth_df.columns)
    # CNV_df = pd.read_csv(dataset_dir+'GeneCN_prep.csv', index_col=0, header=0) 
    # if not use_cnv: 
    #     CNV_df = pd.DataFrame(0, index=np.arange(len(CNV_df)), columns=CNV_df.columns)
    # print('Omics data loaded.')
    if classify:
         _drugsens_tv = pd.read_csv(dataset_dir + "DrugSens_train_CDG_binary.csv", index_col = 0, header = 0)
         _drugsens_test = pd.read_csv(dataset_dir + "DrugSens_test_CDG_binary.csv", index_col = 0, header = 0)
    else:
        _drugsens_tv = pd.read_csv(dataset_dir + "DrugSens_train_CDG.csv", index_col = 0, header = 0)
        _drugsens_test = pd.read_csv(dataset_dir + "DrugSens_test_CDG.csv", index_col = 0, header = 0)
    smiles_list, cl_list, drugsens = DrugsensTransform(pd.concat([_drugsens_tv, _drugsens_test], axis=0))
    drugsens_tv = drugsens.iloc[:len(_drugsens_tv)]
    drugsens_test = drugsens.iloc[len(_drugsens_tv):]
    # print(drugsens_tv.shape)


    outtext_list = [use_mut, use_expr, use_meth, use_cnv]
    
    assert len(drugsens_test) == len(_drugsens_test)
    
    # ------ Debug Mode ------

    if args.debug:
        print('-- DEBUG MODE --')
        n_trials = 1
        max_tuning_epoch = 1
        max_epoch = 3
        folds=2
        drugsens_tv = drugsens_tv.iloc[:100]
        drugsens_test = drugsens_test.iloc[:10]
        batch_size = 16
    else:
        n_trials = 30
        max_tuning_epoch = 3
        max_epoch = 30
        folds=5
        batch_size = 256
    
    # ----- Setup ------
    mainmetric, report_metrics, prediction_task = (BCE(),[BCE(),AUROC(),AUCPR()],'clas') if classify \
                                             else (MSE(),[MSE(),RMSE(),PCC(),R2(),SRCC()],'regr')
    criterion = mainmetric # NOTE must alert (changed from nn.MSELoss to mainmetric)
    print('Loading dataset...')

    print('-- TEST SET --')

    # ------- Storage Path -------
    today_date = date.today().strftime('%Y-%m-%d')

    DatasetTest = DrugOmicsDataset(drugsens_test, omics_dataset, smiles_list, modelname, EVAL = True)
    testloader = get_dataloader(DatasetTest, modelname)

    resultfolder = './results' if not args.debug else './test'
   

    RUN_DIR = f'{resultfolder}/{modelname}/{prediction_task}/'
    os.makedirs(RUN_DIR)
    # status = StatusReport(hyperexpname,hypertune_stop_flag)
    status.set_run_dir(RUN_DIR)
    print(f'Your run directory is "{RUN_DIR}"')

    
    outtext_list.insert(0,exp_name)

    report_metrics_name = [metric.name for metric in report_metrics]
    resultsdf = pd.DataFrame(columns=report_metrics_name,index=list(range(folds)))

    #############################
    
    for fold, (Trainset, Validset) in enumerate(df_kfold_split(drugsens_tv),start=0): 
        print(f'\n=============== Fold {fold+1}/{folds} ===============\n')

        seed = set_seed(100)
        print(f'-- TRAIN SET {fold+1} --')

        # DatasetTrain = DrugOmicsDataset(Trainset, omics_dataset, smiles_list, modelname, EVAL = False)
        # DatasetValid = DrugOmicsDataset(Validset, omics_dataset, smiles_list, modelname, EVAL = True)
        # print('DatasetTrain len =',len(DatasetTrain))
        # print('Trainset len =',len(Trainset))
        trainloader = get_dataloader(DatasetTrain, modelname, batch_size=batch_size)
        validloader = get_dataloader(DatasetValid, modelname)

        ckpt_dir = os.path.join(RUN_DIR, f'fold_{fold}-seed_{seed}/')
        saver = Saver(ckpt_dir, max_epoch)
        model = get_model(args)
        optimizer = optim.AdamW(model.parameters())

        torch.cuda.empty_cache()
        validloss, trainloss = train_and_evaluate(trainloader, validloader, mainmetric, report_metrics, criterion, fold, saver, model, optimizer)

        torch.cuda.empty_cache()

        bestmodel = saver.LoadModel()
        score, predIC50, labeledIC50 = Validation(testloader, bestmodel, report_metrics, modelname, CPU)

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
    print('\n===========================================================\n')
    print('Finish time:',end_time_formatted)
    elapsed_time = end_time - start_time

    print('Writing Output...')

    resultsdf.to_csv(f'{RUN_DIR}/ExperimentSummary.csv')

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

def train_and_evaluate(trainloader, validloader, mainmetric, report_metrics, criterion, fold, saver, model, optimizer):
    validloss = []
    trainloss = []
    printrate = 20

    optimizer.zero_grad()

    print("Start Training...")
    stop_flag = False

    for num_epoch in range(1,max_epoch+1):
        model.train().to(CUDA)
        time_start = datetime.now()
        cum_loss = 0.0
        printloss = 0.0
        print(f'Epoch:{num_epoch}/{max_epoch}')
        status.update({
                    # 'repeat':repeat,
                    'fold':fold,
                    'epoch':num_epoch
                })
        if stop_flag:
            break

        start_iter = datetime.now()
        for ii, Data in enumerate(trainloader): 
            if stop_flag:
                break

            [OmicsInput, DrugInput], Label = Data
            DrugInput = mapDrugDEVICE(DrugInput, modelname, CUDA)
            OmicsInput = [tensor.to(CUDA) for tensor in OmicsInput]
            Label = Label.squeeze(-1).to(CUDA)    # [batch, task]
            output = model([OmicsInput, DrugInput])    # [batch, output_size]

            loss = criterion(output, Label).float()
                
            cum_loss += loss.detach()
            printloss += loss.detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (ii+1) % 20 == 0:
                torch.cuda.empty_cache()

            if (ii+1) % printrate == 0:
                duration_iter = (datetime.now()-start_iter).total_seconds()
                print("Iteration {}/{}".format(ii+1, len(trainloader)))
                print("Duration = ", duration_iter,'seconds, loss = {:.4f}'.format((printloss/printrate)))
                printloss = 0.0
                start_iter = datetime.now()

        time_end = datetime.now()
        duration_epoch = (time_end-time_start).total_seconds()
            # scheduler.step()
        trainmeanloss = (cum_loss/len(trainloader)).item()
        print(f"Epoch duration = {duration_epoch} seconds")
        print(f"Train loss on Epoch {num_epoch} = {trainmeanloss}")

        validresult = Validation(validloader, model, report_metrics, modelname, CPU)[0]
        validmeanloss = validresult[mainmetric.name]

        print('===========================================================\n')
        stop_flag, _, _ = saver.SaveModel(model, optimizer, num_epoch, validresult, mainmetric)
        validloss.append(validmeanloss)
        trainloss.append(trainmeanloss)
    return validloss,trainloss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()

