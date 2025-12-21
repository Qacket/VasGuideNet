import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import torch.nn as nn

from lib.utils.common_function import load_model
from lib.models.resnet import resnet50, resnet34

# from lib.models.swin_transformer import SwinTransformer
from lib.models.MISSFormer3D import MISSFormer3D_origin
from lib.models.unetr import SWINUNETR
from lib.models.UNet3D import UNet3D, ResidualUNet3D
from lib.models.TransSimeanNet import TransSimUNet, get_r50_b16_config
from monai.inferers import sliding_window_inference
from functools import partial


def build_val_model(model, cfg):
    return partial(
        sliding_window_inference,
        roi_size=cfg.MONAI.TRANSFORMS.ROI,
        sw_batch_size=cfg.DATA.VAL_BATCHSIZE,
        predictor=model,
        overlap=0.5,
    )


def build_backbone(cfg, training=True):
    cfg_model = cfg.MODEL
    if cfg_model.NAME == "resnet50":
        model = resnet50(pretrained=False)
    elif cfg_model.NAME == "resnet34":
        model = resnet34(pretrained=False)
    elif cfg_model.NAME == "UNet3D":
        model = UNet3D(
            in_channels=cfg.INPUT.NUM_CLASSES, out_channels=cfg.OUTPUT.NUM_CLASSES
        )
    elif cfg.MODEL.NAME == "ResidualUNet3D":
        model = ResidualUNet3D(
            in_channels=cfg.INPUT.NUM_CLASSES, out_channels=cfg.OUTPUT.NUM_CLASSES
        )
    # elif cfg_model.NAME == "missformer":
    #     model = MISSFormer(
    #         num_classes=cfg.OUTPUT.NUM_CLASSES,
    #         token_mlp_mode=cfg.MODEL.MISSFORMER.TOKEN_MLP_MODE,
    #     )
    elif cfg_model.NAME == "missformer3D":
        model = MISSFormer3D_origin(
            input_dims=cfg.INPUT.NUM_CLASSES,
            num_classes=cfg.OUTPUT.NUM_CLASSES,
            token_mlp_mode=cfg.MODEL.MISSFORMER.TOKEN_MLP_MODE,
            heads=cfg.MODEL.MISSFORMER.HEADS,
            reduction_ratios=cfg.MODEL.MISSFORMER.REDUCTION_RATIOS,
            bridge_reduction_ratios=cfg.MODEL.MISSFORMER.BRIDGE_REDUCTION_RATIOS,
            in_out_chan=cfg.MODEL.MISSFORMER.IN_OUT_CHAN,
            dims=cfg.MODEL.MISSFORMER.DIMS,
            layers=cfg.MODEL.MISSFORMER.LAYERS,
            d_base_feat_size=cfg.MODEL.MISSFORMER.D_BASE_FEAT_SIZE,
            d_base_depth_size=cfg.MODEL.MISSFORMER.D_BASE_DEPTH_SIZE,
            kernel_sizes=cfg.MODEL.MISSFORMER.KERNEL_SIZES,
            strides=cfg.MODEL.MISSFORMER.STRIDES,
            padding_sizes=cfg.MODEL.MISSFORMER.PADDING_SIZES,
        )
    elif cfg_model.NAME == "unetr":
        model = SWINUNETR(
            in_channels=cfg.INPUT.NUM_CLASSES,
            out_channels=cfg.OUTPUT.NUM_CLASSES,
            img_size=cfg.MONAI.TRANSFORMS.ROI,
            feature_size=cfg.MODEL.UNETR.FEATURE_SIZE,
            hidden_size=cfg.MODEL.UNETR.HIDDEN_SIZE,
            mlp_dim=cfg.MODEL.UNETR.MLP_DIM,
            num_heads=cfg.MODEL.UNETR.NUM_HEADS,
            pos_embed=cfg.MODEL.UNETR.POS_EMBED,
            norm_name=cfg.MODEL.UNETR.NORM_NAME,
            conv_block=cfg.MODEL.UNETR.CONV_BLOCK,
            res_block=cfg.MODEL.UNETR.RES_BLOCK,
            dropout_rate=cfg.MODEL.UNETR.DROPOUT_RATE,
        )
    elif cfg_model.NAME == "TransFusionNet":
        model = TransSimUNet(get_r50_b16_config())

    else:
        raise NotImplementedError

    # load pretrained model
    if len(cfg_model.PRETRAIN_MODEL_PATH) > 0 and training:
        load_model(model, cfg_model.PRETRAIN_MODEL_PATH, cfg, cfg.MODEL.DEVICE)

    if "resnet" in cfg.MODEL.NAME:
        model.fc = nn.Identity()

    return model
