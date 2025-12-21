from math import cos, pi
import torch
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
from torch.optim.lr_scheduler import StepLR
from .lr_scheduler import (
    WarmupMultiStepLR,
    LinearWarmupCosineAnnealingLR,
    WarmupExponentialLR,
)
from monai.losses import DiceCELoss, GeneralizedDiceFocalLoss
from .loss import FocalLoss, GlobalDiceLoss


def build_loss_fn(cfg):
    if cfg.SOLVER.LOSS_TYPE == "focal":
        loss_fn = FocalLoss(alpha=[1, 1], gamma=2)
    elif cfg.SOLVER.LOSS_TYPE == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda(
                cfg.rank
            )
        )
    elif cfg.SOLVER.LOSS_TYPE == "global_dice":
        loss_fn = GlobalDiceLoss()
    elif cfg.SOLVER.LOSS_TYPE == "dice":
        loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
    elif cfg.SOLVER.LOSS_TYPE == "dice_focal":
        loss_fn = GeneralizedDiceFocalLoss(
            include_background=False,
            to_onehot_y=True,
            lambda_gdl=0.5,
            lambda_focal=0.5,
            # lambda_focal=0.35,
        )
    else:
        raise ValueError(f"Unsupported loss: {cfg.SOLVER.LOSS_TYPE}")
    return loss_fn


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER_NAME == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")
    return optimizer


def build_lr_scheduler(cfg, len_data, optimizer, start_epoch=0, use_iteration=False):
    epochs = cfg.SOLVER.NUM_EPOCHS if not use_iteration else 1
    lr_mode = cfg.SOLVER.LR_TYPE
    lr_args = {
        "lr": cfg.SOLVER.BASE_LR,
        "momentum": cfg.SOLVER.MOMENTUM,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    lr_scheduler = LRScheduler(
        lr_mode, lr_args, len_data, optimizer, epochs, start_epoch
    )
    return lr_scheduler


class LRScheduler(object):
    def __init__(self, mode, lr_args, data_size, optimizer, num_epochs, start_epochs):
        super(LRScheduler, self).__init__()

        assert mode in ["multistep", "poly", "cosine"]
        self.mode = mode
        self.optimizer = optimizer
        self.data_size = data_size

        self.cur_iter = start_epochs * data_size
        self.max_iter = num_epochs * data_size

        # set learning rate
        self.base_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        self.cur_lr = [lr for lr in self.base_lr]

        # poly kwargs
        # TODO
        if mode == "poly":
            self.power = lr_args["power"] if lr_args.get("power", False) else 0.9
        if mode == "milestones":
            default_mist = list(range(0, num_epochs, num_epochs // 3))[1:]
            self.milestones = (
                lr_args["milestones"]
                if lr_args.get("milestones", False)
                else default_mist
            )
        if mode == "cosine":
            self.targetlr = lr_args["targetlr"]

    def step(self):
        self._step()
        self.update_lr()
        self.cur_iter += 1

    def _step(self):
        if self.mode == "step":
            epoch = self.cur_iter // self.data_size
            power = sum([1 for s in self.milestones if s <= epoch])
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * pow(0.1, power)
                self.cur_lr[i] = adj_lr
        elif self.mode == "poly":
            for i, lr in enumerate(self.base_lr):
                adj_lr = lr * (
                    (1 - float(self.cur_iter) / self.max_iter) ** (self.power)
                )
                self.cur_lr[i] = adj_lr
        elif self.mode == "cosine":
            for i, lr in enumerate(self.base_lr):
                adj_lr = (
                    self.targetlr
                    + (lr - self.targetlr)
                    * (1 + cos(pi * self.cur_iter / self.max_iter))
                    / 2
                )
                self.cur_lr[i] = adj_lr
        else:
            raise NotImplementedError

    def get_lr(self):
        return self.cur_lr

    def update_lr(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.cur_lr):
            param_group["lr"] = lr
