import torch
from candragat.utils import mapDrugDEVICE

def Validation(validloader, model, metrics, modelname, DEVICE):
    model.eval().to(DEVICE)
    print("Validating..")
    All_answer = []
    All_label = []
    for ii, Data in enumerate(validloader):

        [ValidOmicsInput, ValidDrugInput], ValidLabel = Data
        ValidDrugInput = mapDrugDEVICE(ValidDrugInput, modelname, DEVICE)
        ValidOmicsInput = [tensor.to(DEVICE) for tensor in ValidOmicsInput]

        if modelname == 'FragAttentiveFP':
            ValidOmicsInput = [x.squeeze(0) for x in ValidOmicsInput]
            ValidDrugInput = [x.squeeze(0) for x in ValidDrugInput]
        # ValidLabel = ValidLabel.squeeze(-1)
        ValidLabel = ValidLabel.squeeze(0).to(DEVICE)

        ValidOutput = model([ValidOmicsInput,ValidDrugInput])
        ValidOutputMean = ValidOutput.mean(dim=0, keepdims=True)  # [1, output_size]

        All_answer.append(ValidOutputMean.item())
        All_label.append(ValidLabel[0].item())

    scores = {}

    assert len(All_label) == len(All_answer)
    if len(metrics) != 1:
        for metric in metrics:
            result = metric(All_answer, All_label)
            scores.update({metric.name: result})
            print(metric.name, ': ', result)
    elif len(metrics) == 1:
        result = metrics(All_answer, All_label)
        scores.update({metrics.name: result})
        print(metrics.name, ': ', result)

    torch.cuda.empty_cache()
    model.train()

    return scores, All_answer, All_label