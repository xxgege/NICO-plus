from torch import optim
import torch_optimizer

def get_optim_and_scheduler(network, hparams):
    if hparams["optimizer"] == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    elif hparams["optimizer"] == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=hparams["lr"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"])
    elif hparams["optimizer"] == "AdamW":
        optimizer = optim.AdamW(network.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    elif hparams["optimizer"] == "Yogi":
        optimizer = torch_optimizer.Yogi(network.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    elif hparams["optimizer"] == "AdaBelief":
        optimizer = torch_optimizer.AdaBelief(network.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    elif hparams["optimizer"] == "Adahessian":
        optimizer = torch_optimizer.Adahessian(network.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])    
    else:
        raise NotImplementedError
    
    if hparams["scheduler"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(hparams["epochs"]*hparams["step_size_ratio"]), gamma=hparams["decay_ratio"])
    elif hparams["scheduler"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams["epochs"])
    elif hparams["scheduler"] == "None":
        scheduler = None
    else:
        raise NotImplementedError

    return optimizer, scheduler