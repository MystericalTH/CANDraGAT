import torch
from candragat.utils import DrugInputToDevice
import tqdm
from candragat.utils import TqdmToLogger
from candragat.models import MultiOmicsMolNet
from torch.utils.data import DataLoader
import numpy as np

def Validation(validloader: DataLoader, model: MultiOmicsMolNet, metrics, modelname, mainlogger, pbarlogger, DEVICE):
    model.eval().to(DEVICE)
    mainlogger.info("Validating..")
    All_answer = []
    All_label = []
    validloader.dataset.precalculate_drug_tensor(model)
    for ii, Data in tqdm.tqdm(enumerate(validloader), total=len(validloader), file=TqdmToLogger(pbarlogger), mininterval=10, desc='Validation'):

        [ValidOmicsInput, ValidDrugInput], ValidLabel = Data
        ValidDrugInput = ValidDrugInput.to(DEVICE)
        ValidOmicsInput = [tensor.to(DEVICE) for tensor in ValidOmicsInput]
        ValidLabel = ValidLabel.squeeze(0).to(DEVICE)

        ValidOutput = model.forward_validation([ValidOmicsInput,ValidDrugInput])
        
        All_answer.append(ValidOutput)
        All_label.append(ValidLabel)

    scores = {}
    All_answer = torch.cat(All_answer, dim=0).squeeze()
    All_label = torch.cat(All_label, dim=0).squeeze()
    All_answer = All_answer.tolist()
    All_label = All_label.tolist()

    assert len(All_label) == len(All_answer)

    if len(metrics) != 1:
        for metric in metrics:
            result = metric(All_answer, All_label)
            scores.update({metric.name: result})
            mainlogger.info(f"{metric.name}: {result}")
    elif len(metrics) == 1:
        result = metrics(All_answer, All_label)
        scores.update({metrics.name: result})
        mainlogger.info(f"{metrics.name}: {result}")

    torch.cuda.empty_cache()
    model.train()
    validloader.dataset.clear_drug_tensor()

    return scores, All_answer, All_label