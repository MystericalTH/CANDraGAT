from matplotlib.cbook import report_memory
from Model2 import *
import pandas as pd
import os
import sys
from pathlib import Path
from os.path import expanduser
from datetime import datetime, date
from Metrics import *
import torch
from torch import optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import csv
import argparse
import optuna
from sklearn.model_selection import train_test_split
import arrow


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

# ----- Hyperparameter ------

def get_best_trial(hyper_dir):
    with open(f'{hyper_dir}.json','r') as jsonfile:
        best_trial_param = json.load(jsonfile)
    return best_trial_param

def run_hyper_study(study_func, n_trials, hyperexpdir,study_name=None):
    study = optuna.create_study(direction="minimize", study_name=study_name, load_if_exists=True)
    study.optimize(study_func, n_trials=n_trials)
    # df = study.trials_dataframe()
    trial = study.best_trial
    best_trial_param = dict()
    for key, value in trial.params.items():
        best_trial_param[key]=value
    with open(f'{hyperexpdir}.json','w') as jsonfile:
        json.dump(best_trial_param,jsonfile,indent=4)
    return study

# def hyperparam_optim_optuna(trial,omicsdata,DatasetTrain,DatasetValid):
def hyperparam_optim_optuna(trial):

    criterion = nn.MSELoss()

    batchsize = 2**(trial.suggest_int('batchsize', 2,6))
    drop_rate = trial.suggest_float('drop_rate',0.1,0.9,step=0.1)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    WeightDecay = trial.suggest_uniform('WeightDecay',1e-7, 1e-5)
    omics_output_size = trial.suggest_int('omics_output_size',100,300,step=25)
    drug_output_size = trial.suggest_int('drug_output_size',100,300,step=25)
    
    if modelname == "AttentiveFP":
        trainloader = data.DataLoader(DatasetTrain, batch_size=batchsize, num_workers=2,
                                    drop_last=True, worker_init_fn=np.random.seed(), pin_memory=True)
        validloader = data.DataLoader(DatasetValid, batch_size=1, num_workers=0,
                                    drop_last=True, worker_init_fn=np.random.seed(), pin_memory=True)

    elif modelname in ("GAT", "GCN"):
        trainloader = PyGDataLoader(DatasetTrain, batch_size=batchsize, num_workers=2, drop_last=True,
                                    worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)
        validloader = PyGDataLoader(DatasetValid, batch_size=1, num_workers=0, drop_last=True,
                                    worker_init_fn=np.random.seed(), pin_memory=True, collate_fn=graph_collate_fn)

    model = MODPredIC50(
                OmicsData,  # = [Mut_df,GeneExp_df,Meth_df,GeneCN_df]
                FP_size=150,         # 'FP_size': 150
                DNN_layers=[256, 256],   # 'DNNLayers':[512] / [256,100]
                drug_output_size=drug_output_size,
                omics_output_size=omics_output_size,
                drop_rate=drop_rate,
                use_conv=False,
                use_mut=use_mut,
                use_gexp=use_gexp,
                use_meth=use_meth,
                use_genecn=use_genecn,
                modelname=modelname,
                classify=classify
    ).to(device)

    model.train()
    model.zero_grad()
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=WeightDecay)
    optimizer.zero_grad()
    
    for epoch in range(max_tuning_epochs):

        cum_loss = 0.0
        printloss = 0.0

        for Data in trainloader:
            
            loss = 0.0

            Input, Label = Data
            Label = Label.squeeze(-1).to(device)    # [batch, task]
            #Label = Label.t()            # [task, batch]

            output = model(Input)   # [batch, output_size]

            loss += criterion(output,Label).float()

            cum_loss += loss.detach()
            printloss += loss.detach()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # trainmseloss = (cum_loss/len(trainloader)).item()
        results = Validation(validloader, model, [MSE()])
        validmseloss = results['MSE']
        # result['RMSE_train'] = np.sqrt(trainmseloss)

        trial.report(validmseloss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return validmseloss

def main():

    dataset_dir = "/share/galaxy/nattawin/test/project/nci60/New_Cleaned_Dataset/"

    global ckpt_dir, max_epoch, max_tuning_epochs, device
    global DatasetTrain, DatasetValid, OmicsData
    global use_mut, use_gexp, use_meth, use_genecn, modelname, classify

    model_name = ["AttentiveFP", "GAT", "GCN","CANDraGAT"]
    parser = argparse.ArgumentParser(description='ablation_analysis')
    parser.add_argument('--modelname', dest='modelname', action='store',
                        choices=model_name, help="AttentiveFP or GAT or GCN")
    parser.add_argument('--no_mut', dest='no_mut', default=False,
                        action="store_true", help='use gene mutation or not')
    parser.add_argument('--no_gexp', dest='no_gexp', default=False,
                        action="store_true", help='use gene expression or not')
    parser.add_argument('--no_methy', dest='no_methy', default=False,
                        action="store_true", help='use methylation or not')
    parser.add_argument('--no_genecn', dest='no_genecn', default=False,
                        action="store_true", help='use copy number variation or not')
    parser.add_argument('-c','--classify', dest='classify', default=False,
                        action="store_true", help='use copy number variation or not')
    parser.add_argument('-rs', '--resume', dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')
    parser.add_argument('-db', '--debug', default=False, action="store_true", dest='debug', help='debug file/test run')
    parser.add_argument('-l', '--load_hyper', required=False,nargs=2, dest='hyperpath', help='load hyperparameter file, enter hyperparameter directory')

    args = parser.parse_args()

    classify = args.classify

    device = torch.device('cuda')

    # ------ Loading Dataset  ------

    Mut_df = pd.read_csv(dataset_dir+'Mut_prep.csv', index_col=0, header=0)
    GeneExp_df = pd.read_csv(dataset_dir+'GeneExp_prep.csv', index_col=0, header=0)
    Meth_df = pd.read_csv(dataset_dir+'Meth_prep.csv', index_col=0, header=0)
    GeneCN_df = pd.read_csv(dataset_dir+'GeneCN_prep.csv', index_col=0, header=0)
    print('Omics data loaded.')

    DrugOmics_df = pd.read_csv(dataset_dir+"DrugSens_train_CDG.csv",index_col = 0,header = 0)
    DrugOmics_df_test = pd.read_csv(dataset_dir+"DrugSens_test_CDG.csv",index_col = 0,header = 0)


    # ------ Debug Mode ------

    if args.debug:
        n_trials = 1
        max_tuning_epochs = 1
        num_test_round = 2
        max_epoch = 3
    else:
        n_trials = 250
        max_tuning_epochs = 500
        num_test_round = 10
        max_epoch = 500

    # ------ Ablation ------
    
    use_mut, use_gexp, use_meth, use_genecn = True, True, True, True
    feature_used = []
    outtext_list = [arrow.utcnow().to('local').format('YYYY-MM-DD_HH-mm-ss')]
    if args.no_mut:
        use_mut = False
        Mut_df= pd.DataFrame(0, index=np.arange(len(Mut_df)), columns=Mut_df.columns)
        outtext_list.append(0)
    else:
        feature_used.append('mut')
        outtext_list.append(1)

    if args.no_gexp:
        use_gexp = False
        GeneExp_df = pd.DataFrame(0, index=np.arange(len(GeneExp_df)), columns=GeneExp_df.columns)
        outtext_list.append(0)
    else:
        feature_used.append('geneexp')
        outtext_list.append(1)

    if args.no_methy:
        use_meth = False
        Meth_df = pd.DataFrame(0, index=np.arange(len(Meth_df)), columns=Meth_df.columns)
        outtext_list.append(0)
    else:
        feature_used.append('meth')
        outtext_list.append(1)

    if args.no_genecn:
        use_genecn = False
        GeneCN_df = pd.DataFrame(0, index=np.arange(len(GeneCN_df)), columns=GeneCN_df.columns)
        outtext_list.append(0)
    else:
        feature_used.append('genecn')
        outtext_list.append(1)

    OmicsData = [Mut_df, GeneExp_df, Meth_df, GeneCN_df]
    modelname = args.modelname

    # storagepath = f'CANDraGAT_June2022/{modelname}/{featurename}/'

    def random_seed():
        new_seed = random.randrange(2**32-1)
        random.seed(new_seed)
        torch.manual_seed(new_seed)
        np.random.seed(new_seed)
        return new_seed

    # ----- Setup ------
    report_metrics, prediction_task = ([BCE(),AUC(),AUC_PR()],'classification') if classify else ([MSE(),RMSE(),PCC(),R2(),SRCC()],'regression')
    criterion = nn.MSELoss()

    print('Loading dataset...')

    DrugOmicsDataset = DrugOmicsDatasetCreator().createDataset(DrugOmics_df) 
    Testset = DrugOmicsDatasetCreator().createDataset(DrugOmics_df_test) 
    Trainset, Validset = train_test_split(DrugOmicsDataset, test_size=0.1, random_state = 42)

    DatasetTrain = MolDatasetTrain(Trainset, omicsdata=OmicsData, modelname=modelname)
    DatasetValid = MolDatasetEval(Validset, omicsdata=OmicsData, modelname=modelname)
    DatasetTest = MolDatasetEval(Testset, omicsdata=OmicsData, modelname=modelname)
        
    # ------- Storage Path -------
    today_date = date.today().strftime('%Y-%m-%d')
    # target_dir = './MultiTask/'+target_path

    if args.hyperpath is None:

        num_hyperrun = 1
        hyperexpdir = f'CANDraGAT_June2022/results/{modelname}/hyperparameters/{today_date}_HyperRun{num_hyperrun}'
        while os.path.exists(hyperexpdir):
            num_hyperrun+=1
            hyperexpdir = f'CANDraGAT_June2022/results/{modelname}/hyperparameters/{today_date}_HyperRun{num_hyperrun}'
        os.makedirs(hyperexpdir)         

        hyperexpname = f'{today_date}_HyperRun{num_hyperrun}'
        run_dir = f'CANDraGAT_June2022/results/{modelname}/{prediction_task}/{hyperexpname}_TestRun1'
        outtext_list.insert(1,hyperexpname)
        os.makedirs(run_dir)

        print(f'Your run directory is "{run_dir}"')
        print('Start hyperparameters optimization')

        run_hyper_study(study_func=hyperparam_optim_optuna, n_trials=n_trials,hyperexpdir=hyperexpdir)
        best_trial_param = get_best_trial(hyperexpdir)
        exp_name=f'{hyperexpname}_TestRun{num_run}'

        print('Hyperparameters optimization done')

    else: 
        hyper_modelname,hyperexpname= args.hyperpath
        hyper_jsondir = f'CANDraGAT_June2022/results/{hyper_modelname}/hyperparameters/{hyperexpname}.json'
        best_trial_param = get_best_trial(hyper_jsondir)

        num_run = 1
        run_dir = f'CANDraGAT_June2022/results/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'
        while os.path.exists(run_dir):
            num_run+=1
            run_dir = f'CANDraGAT_June2022/results/{modelname}/{prediction_task}/{hyperexpname}_TestRun{num_run}'
        os.makedirs(run_dir)
        outtext_list.insert(1,hyperexpname)
        exp_name=f'{hyperexpname}_TestRun{num_run}'
        print(f'Your run directory is "{run_dir}"')

    outtext_list.insert(0,exp_name)

    batchsize = 2**best_trial_param['batchsize']
    drop_rate = best_trial_param['drop_rate']
    lr = best_trial_param['lr']
    weight_decay = best_trial_param['WeightDecay']
    omics_output_size = best_trial_param['omics_output_size']
    drug_output_size = best_trial_param['drug_output_size']

    resultsdf = pd.DataFrame(columns=[metric.name for metric in report_metrics],index=list(range(num_test_round)))

    #############################

    for round_ in range(num_test_round):
        print('\n=============== Round {}/5 ===============\n'.format(round_))

        seed = random_seed()



        if modelname == "AttentiveFP":
            trainloader = data.DataLoader(DatasetTrain, batch_size=batchsize, num_workers=2,
                                        drop_last=True, worker_init_fn=np.random.seed(seed), pin_memory=True)
            validloader = data.DataLoader(DatasetValid, batch_size=1, num_workers=0,
                                        drop_last=True, worker_init_fn=np.random.seed(seed), pin_memory=True)
            testloader = data.DataLoader(DatasetTest, batch_size=1, shuffle=False, num_workers=0,
                                        drop_last=True, worker_init_fn=np.random.seed(seed), pin_memory=True)
            max_epoch = 30
    
        elif modelname in ("GAT", "GCN"):
            trainloader = PyGDataLoader(DatasetTrain, batch_size=batchsize, num_workers=2, drop_last=True,
                                        worker_init_fn=np.random.seed(seed), pin_memory=True, collate_fn=graph_collate_fn)
            validloader = PyGDataLoader(DatasetValid, batch_size=1, num_workers=0, drop_last=True,
                                        worker_init_fn=np.random.seed(seed), pin_memory=True, collate_fn=graph_collate_fn)
            testloader = PyGDataLoader(DatasetTest, batch_size=1, shuffle=False, num_workers=0, drop_last=True,
                                    worker_init_fn=np.random.seed(seed), pin_memory=True, collate_fn=graph_collate_fn)
            max_epoch = 100

        ckpt_dir = os.path.join(run_dir, f'round{round_}_seed{seed}/Checkpoint/')

        # try:
        #     os.makedirs(ckpt_dir)
        # except:
        #     pass

        saver = Saver(ckpt_dir, max_epoch)

        model, optimizer, lastepoch = saver.LoadModel()

        # print(epoch)
        if model is None:
            model = MODPredIC50(
                OmicsData,  # = [Mut_df,GeneExp_df,Meth_df,GeneCN_df]
                FP_size=150,         # 'FP_size': 150
                DNN_layers=[256, 256],   # 'DNNLayers':[512] / [256,100]
                drug_output_size=drug_output_size,
                omics_output_size=omics_output_size,
                drop_rate=drop_rate,
                use_conv=False,
                use_mut=use_mut,
                use_gexp=use_gexp,
                use_meth=use_meth,
                use_genecn=use_genecn,
                modelname=modelname,
                classify=classify
            ).to(device)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
            scheduler = MultiStepLR(optimizer,[5,10],gamma=0.1)
        if lastepoch is None:
            lastepoch = 0
        else:
            if lastepoch == max_epoch:
                continue

        torch.cuda.empty_cache()
        validloss = []
        trainloss = []
        printrate = 20
        model.train()

        optimizer.zero_grad()
        model.zero_grad()

        print("Start Training...")
        stop_flag = False


        for epoch_ in range(1,max_epoch-lastepoch+1):

            time_start = datetime.now()
            num_epoch = epoch_+lastepoch
            cum_loss = 0.0
            printloss = 0.0
            if stop_flag:
                break

            print("Epoch: {}/{}".format(num_epoch, max_epoch))
            start_iter = datetime.now()
        
            for ii, Data in enumerate(trainloader):    

                loss = 0.0

                if stop_flag:
                    break

                Input, Label = Data
                Label = Label.squeeze(-1).to(device)    # [batch, task]
                output = model(Input)   # [batch, output_size]

                loss += criterion(output, Label)
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
                    print("Duration = ", duration_iter,
                        'seconds, loss = {:.4f}'.format((printloss/printrate)))
                    printloss = 0.0
                    start_iter = datetime.now()

            time_end = datetime.now()
            duration_epoch = (time_end-time_start).total_seconds()
            scheduler.step()
            trainmeanloss = (cum_loss/len(trainloader)).item()
            print("Epoch duration = ", duration_epoch, 'seconds')
            print("Train loss on Epoch {} = {} ".format(num_epoch, trainmeanloss))
            validresult = Validation(
                validloader, model, valid_metrics=report_metrics)
            validmeanloss = validresult['RMSE'] if model.regr else validresult['BCE']
            # result['RMSE_Train'] = np.sqrt(trainsumloss)
            print('===========================================================\n')
            stop_flag, best_ckpt, best_value = saver.SaveModel(
                model, optimizer, num_epoch, validresult)
            # print(stop_flag)
            validloss.append(validmeanloss)
            trainloss.append(trainmeanloss)
        
        bestmodel, *_ = saver.LoadModel()
        score, predIC50, labeledIC50 = Validation(testloader, bestmodel, valid_metrics=report_metrics, mode='GetAnswerLabel')
        torch.cuda.empty_cache()

        modelpath3 = run_dir + "/round{}/".format(round_)
        traintestloss = np.array([trainloss, validloss])
        np.savetxt(modelpath3+'traintestloss.csv', traintestloss, delimiter=',', fmt='%.5f')

        # create text file for test
        testpredlabel = np.array([predIC50, labeledIC50])
        np.savetxt(modelpath3+'test_predlabel.csv', testpredlabel, delimiter=',', fmt='%2.6f')

        loss_list = []
        for i in range(len(trainloss)):
            loss_list.append("Epoch " + str(i+1) + " train_loss " +
                            str(trainloss[i]) + " val_loss " + str(validloss[i]) + "\n")

        resultsdf.to_csv(f'{run_dir}/RunSummary.csv')

        for col in resultsdf.columns:
            mean, interval = compute_confidence_interval(resultsdf[col])
            outtext_list.extend((mean,interval))

        summaryfile = f'CANDraGAT_June2022/results/{modelname}/{prediction_task}/ResultSummarySheet.csv'
        if not os.path.exists(summaryfile):
            X = pd.DataFrame(columns=pd.MultiIndex.from_product((report_metrics,['mean','interval'])))
            X.index.name='experiment_time'
            X.insert(0,'mutation',np.nan)
            X.insert(1,'gene_expression',np.nan)
            X.insert(2,'methylation',np.nan)
            X.insert(3,'copy_number_variation', np.nan)
            X.insert(0,'hyperparameters_file',np.nan)
            X.to_csv(summaryfile)
        with open(summaryfile, 'a') as outfile:
            output_writer = csv.writer(outfile,delimiter = ',')
            output_writer.writerow(outtext_list)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()


