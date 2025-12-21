# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
from einops import rearrange
from tensorboardX import SummaryWriter
from torch.amp import GradScaler, autocast
import nibabel as nib
from utils.lw_measure import cal_dsc
from utils.utils_new import (
    AverageMeter,
    distributed_all_gather,
    reduce_by_weight,
    resample_3d,
    resample_3d_order_0,
    resample_3d_torch,
    cal_dice,
    unSpatialPad_v2,
)

from utils.utils_new import info_if_main


def train_epoch(
        model,
        loader,
        optimizer,
        scaler,
        epoch,
        loss_func,
        args,
        loss_func2=None,
):
    """训练单个 epoch"""
    model.train()
    start_time = time.time()
    run_loss1 = AverageMeter()
    run_loss2 = AverageMeter()

    # # 启用异常检测
    # torch.autograd.set_detect_anomaly(True)

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        # data, target = batch_data[0], batch_data[1]
        # data = rearrange(data, "b n c w h d -> (b n) c w h d").contiguous().float()
        # target = rearrange(target, "b n c w h d -> (b n) c w h d").contiguous().long()

        data, target = data.cuda(args.rank), target.cuda(args.rank)

        if args.is_vessel:
            target[target > 0] = 1

        with autocast(device_type="cuda", enabled=args.amp):
            if args.model_name in [
                "VesselEnhancedNet",
                "HCFormer_2channels",
                "HCFormer_disEmbed",
                "HCFormer_tokenDecoder",
            ]:
                vessel_label = batch_data["label_vessel"].cuda(args.rank)
                logits = model(torch.cat([data, vessel_label], dim=1))
            else:
                logits = model(data)

            # **特殊情况: logits 被拆分**
            if logits.shape[1] != args.out_channels:
                features, logits = (
                    logits[:, : -args.out_channels],
                    logits[:, -args.out_channels:],
                )
            else:
                features = None

            # 计算 loss1
            loss1 = loss_func(logits, target)

            # **检查 loss1 是否需要梯度**
            if not loss1.requires_grad:
                raise RuntimeError(
                    "loss1 does not require gradients! Check computation graph."
                )

            loss = loss1
            # 计算 loss2（如果有）
            if loss_func2 is not None:
                if features is None:
                    raise RuntimeError("features is None, but loss_func2 requires it!")

                vessel_label = batch_data["label_vessel"].cuda(args.rank)
                loss2 = loss_func2(features, logits, target, vessel_label, epoch)

                # **检查 loss2 是否在计算图中**
                if not loss2.requires_grad:
                    raise RuntimeError(
                        "loss2 does not require gradients! Check computation graph."
                    )
                loss = loss1 + 0.1 * loss2
            else:
                loss2 = torch.tensor(0.0, device=loss1.device)

        # === AMP 与 backward ===
        if args.amp:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # # 进行梯度裁剪，限制最大梯度范数
            # scaler.unscale_(optimizer)  # 解缩放梯度
            # clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # **分布式训练: 同步 loss**
        if args.distributed:
            loss_list1 = distributed_all_gather(
                [loss1], out_numpy=True, is_valid=idx < loader.sampler.valid_length
            )
            run_loss1.update(
                np.mean(np.mean(np.stack(loss_list1, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size,
            )

            if loss_func2 is not None:
                loss_list2 = distributed_all_gather(
                    [loss2], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                run_loss2.update(
                    np.mean(np.mean(np.stack(loss_list2, axis=0), axis=0), axis=0),
                    n=args.batch_size * args.world_size,
                )
        else:
            run_loss1.update(loss1.item(), n=args.batch_size)
            if loss_func2 is not None:
                run_loss2.update(loss2.item(), n=args.batch_size)

        # # **打印日志**
        # learning_rate = optimizer.param_groups[0]["lr"]
        # info_if_main(
        #     "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
        #     "loss1: {:.4f}".format(run_loss1.avg),
        #     "loss2: {:.4f}".format(run_loss2.avg if loss_func2 is not None else 0),
        #     "lr:{:.6f}".format(learning_rate),
        #     "time {:.2f}s".format(time.time() - start_time),
        # )
        # start_time = time.time()

    # # **确保梯度清空**
    # for param in model.parameters():
    #     param.grad = None

    # return run_loss1.avg, 0
    return run_loss1.avg, run_loss2.avg


#     return run_loss1.avg, run_loss2.avg


def val_epoch_v2(model, loader, epoch, args, model_inferer=None):
    model.eval()
    data_num, dice_sum, acc_sum = 0, 0, 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            # print("val shape:", data.shape, target.shape, batch_data["label_vessel"].shape)
            # data, target = batch_data[0], batch_data[1]
            # data = data.contiguous().float()
            # target = target.contiguous().long()
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            if args.is_vessel:
                target[target > 0] = 1
            with autocast(enabled=args.amp, device_type="cuda"):
                assert model_inferer is not None, "model_inferer is None"
                if (
                        args.model_name == "VesselEnhancedNet"
                        or args.model_name == "HCFormer_2channels"
                        or args.model_name == "HCFormer_disEmbed"
                        or args.model_name == "HCFormer_tokenDecoder"
                ):
                    logits = model_inferer(
                        torch.cat(
                            [data, batch_data["label_vessel"].cuda(args.rank)],
                            dim=1,
                        )
                    )
                else:
                    logits = model_inferer(data)

            if logits.shape[1] != args.out_channels:
                features, logits = (
                    logits[:, : -args.out_channels],
                    logits[:, -args.out_channels:],
                )
            val_outputs = torch.argmax(logits, axis=1)[0]  # b, c, h, w, d -> h, w, d
            val_outputs = unSpatialPad_v2(val_outputs, data).astype(np.uint8)
            # val_outputs = val_outputs.detach().cpu().numpy().astype(np.uint8)
            val_labels = target.cpu().numpy()[0, 0, :, :, :].astype(np.uint8)
            target_shape = val_labels.shape
            val_outputs = resample_3d(val_outputs, target_shape)

            union_list = []
            intersect_list = []
            # dsc = cal_dice(val_outputs > 0, val_labels > 0)
            for j in range(1, args.out_channels):
                pre_binary = get_class(val_outputs, j)
                gt_binary = get_class(val_labels, j)
                intesect = np.count_nonzero(pre_binary * gt_binary)
                union = np.count_nonzero(pre_binary) + np.count_nonzero(gt_binary)
                intersect_list.append(intesect)
                union_list.append(union)
            dsc = safe_divide(2 * np.sum(intersect_list), np.sum(union_list))

            # dsc = cal_dice(val_outputs, val_labels)

            dice_sum += dsc
            # acc_sum += acc
            data_num += 1

            # if args.rank == 0:
            #     info_if_main(
            #         "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            #         "dice",
            #         dsc,
            #         "acc",
            #         acc,
            #         "time {:.2f}s".format(time.time() - start_time),
            #     )
            start_time = time.time()
        dist.barrier()
        avg_dice = reduce_by_weight(dice_sum / data_num, data_num)
        # avg_acc = reduce_by_weight(acc_sum / data_num, data_num)
    return avg_dice


def val_epoch_v3(model, loader, epoch, args, model_inferer=None):
    output_directory = os.path.join(args.logdir, args.output_name)
    os.makedirs(output_directory, exist_ok=True)
    model.eval()
    dice_sum, iou_sum, rvd_sum, data_num, dice_sum_old = (0, 0, 0, 0, 0)

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            # print("val shape:", data.shape, target.shape, batch_data["label_vessel"].shape)
            # data, target = batch_data[0], batch_data[1]
            # data = data.contiguous().float()
            # target = target.contiguous().long()
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            original_affine = batch_data["affine_matrix"][0].numpy()
            img_name = batch_data["name"][0]
            if args.is_vessel:
                target[target > 0] = 1
            with autocast(enabled=args.amp, device_type="cuda"):
                assert model_inferer is not None, "model_inferer is None"
                if (
                        args.model_name == "VesselEnhancedNet"
                        or args.model_name == "HCFormer_2channels"
                        or args.model_name == "HCFormer_disEmbed"
                        or args.model_name == "HCFormer_tokenDecoder"
                ):
                    logits = model_inferer(
                        torch.cat(
                            [data, batch_data["label_vessel"].cuda(args.rank)],
                            dim=1,
                        )
                    )
                else:
                    logits = model_inferer(data)

            if logits.shape[1] != args.out_channels:
                features, logits = (
                    logits[:, : -args.out_channels],
                    logits[:, -args.out_channels:],
                )
            # val_outputs = torch.argmax(logits, axis=1)[0]  # b, c, h, w, d -> h, w, d
            # val_outputs = unSpatialPad_v2(val_outputs, data).astype(np.uint8)
            # # val_outputs = val_outputs.detach().cpu().numpy().astype(np.uint8)
            # val_labels = target.cpu().numpy()[0, 0, :, :, :].astype(np.uint8)
            # target_shape = val_labels.shape
            # val_outputs = resample_3d(val_outputs, target_shape)

            # 优化 减小显存
            val_outputs = torch.argmax(logits, dim=1)[0].detach().cpu()
            del logits
            torch.cuda.empty_cache()
            val_outputs = unSpatialPad_v2(val_outputs, data)  # [H, W, D]
            val_labels = target[0, 0].detach().cpu()  # 💡避免 .numpy().astype() 占内存
            val_outputs = resample_3d_torch(val_outputs, val_labels.shape)
            val_outputs = val_outputs.byte().numpy()
            val_labels = val_labels.byte().numpy()

            intersect_list = []
            union1_list = []
            union2_list = []
            pre_area_list = []
            gt_area_list = []
            for j in range(1, args.num_class):
                pre_binary = get_class(val_outputs, j)
                gt_binary = get_class(val_labels, j)

                intesect = np.count_nonzero(pre_binary * gt_binary)
                union1 = np.count_nonzero(pre_binary) + np.count_nonzero(gt_binary)
                union2 = np.count_nonzero((pre_binary + gt_binary) > 0)
                pre_area_list.append(pre_binary.sum())
                gt_area_list.append(gt_binary.sum())
                intersect_list.append(intesect)
                union1_list.append(union1)
                union2_list.append(union2)

            dice_old = cal_dsc(val_outputs > 0, val_labels > 0)
            dice = safe_divide(2 * np.sum(intersect_list), np.sum(union1_list))
            iou = safe_divide(np.sum(intersect_list), np.sum(union2_list))
            rvd = safe_divide(
                abs(np.sum(pre_area_list) - np.sum(gt_area_list)),
                np.sum(gt_area_list),
            )

            dice_sum_old += dice_old
            dice_sum += dice
            iou_sum += iou
            rvd_sum += rvd
            data_num += 1
            # if args.rank == 0:
            #     info_if_main(f"Start validation: {idx}/{len(loader)}, Image: {img_name}")
            #     info_if_main(
            #         f"DSC: {dice:.4f}, IOU: {iou:.4f}, RVD: {rvd:.4f}, DSC_old: {dice_old:.4f}"
            #     )

            if args.save_output:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                    os.path.join(output_directory, img_name),
                )

            # === 显存清理 ===
            del data, target, val_outputs, val_labels
            torch.cuda.empty_cache()

        dist.barrier()
        avg_dice = reduce_by_weight(dice_sum / data_num, data_num)
        avg_iou = reduce_by_weight(iou_sum / data_num, data_num)
        avg_rvd = reduce_by_weight(rvd_sum / data_num, data_num)
        avg_dice_old = reduce_by_weight(dice_sum_old / data_num, data_num)
        info_if_main("Overall Mean Dice_old: {}".format(np.mean(avg_dice_old)))
        info_if_main("Overall Mean Dice: {}".format(np.mean(avg_dice)))
        info_if_main("Overall Mean iou: {}".format(np.mean(avg_iou)))
        info_if_main("Overall Mean rvd: {}".format(np.mean(avg_rvd)))
    return avg_dice, avg_iou, avg_rvd, avg_dice_old


def val_epoch_v4(model, loader, epoch, args, model_inferer=None):
    output_directory = os.path.join(args.logdir, args.output_name)
    os.makedirs(output_directory, exist_ok=True)
    model.eval()
    dice_sum, iou_sum, rvd_sum, data_num, dice_sum_old = (0, 0, 0, 0, 0)
    skipped = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data, target = data.cuda(args.rank), target.cuda(args.rank)
            original_affine = batch_data["affine_matrix"][0].numpy()
            img_name = batch_data["name"][0]

            if args.is_vessel:
                target[target > 0] = 1

            try:
                with autocast(enabled=args.amp, device_type="cuda"):
                    assert model_inferer is not None, "model_inferer is None"

                    if (
                            args.model_name == "VesselEnhancedNet"
                            or args.model_name == "HCFormer_2channels"
                            or args.model_name == "HCFormer_disEmbed"
                            or args.model_name == "HCFormer_tokenDecoder"
                    ):
                        logits = model_inferer(
                            torch.cat(
                                [data, batch_data["label_vessel"].cuda(args.rank)],
                                dim=1,
                            )
                        )
                    else:
                        logits = model_inferer(data)

                if logits.shape[1] != args.out_channels:
                    features, logits = (
                        logits[:, : -args.out_channels],
                        logits[:, -args.out_channels:],
                    )

                val_outputs = torch.argmax(logits, axis=1)[0]  # b, c, h, w, d -> h, w, d
                val_outputs = unSpatialPad_v2(val_outputs, data).astype(np.uint8)
                # val_outputs = val_outputs.detach().cpu().numpy().astype(np.uint8)
                val_labels = target.cpu().numpy()[0, 0, :, :, :].astype(np.uint8)
                target_shape = val_labels.shape
                val_outputs = resample_3d_order_0(val_outputs, target_shape)

                intersect_list = []
                union1_list = []
                union2_list = []
                pre_area_list = []
                gt_area_list = []
                for j in range(1, args.num_class):
                    pre_binary = get_class(val_outputs, j)
                    gt_binary = get_class(val_labels, j)

                    intesect = np.count_nonzero(pre_binary * gt_binary)
                    union1 = np.count_nonzero(pre_binary) + np.count_nonzero(gt_binary)
                    union2 = np.count_nonzero((pre_binary + gt_binary) > 0)
                    info_if_main(
                        f"[Debug Class {j}] pre_area={pre_binary.sum()}, gt_area={gt_binary.sum()}, intersect={intesect}, union1={union1}, union2={union2}")

                    pre_area_list.append(pre_binary.sum())
                    gt_area_list.append(gt_binary.sum())
                    intersect_list.append(intesect)
                    union1_list.append(union1)
                    union2_list.append(union2)

                dice_old = cal_dsc(val_outputs > 0, val_labels > 0)
                dice = safe_divide(2 * np.sum(intersect_list), np.sum(union1_list))
                iou = safe_divide(np.sum(intersect_list), np.sum(union2_list))
                rvd = safe_divide(
                    abs(np.sum(pre_area_list) - np.sum(gt_area_list)),
                    np.sum(gt_area_list),
                )
                info_if_main(f"dice_old: {dice_old}, dice: {dice}, iou: {iou}, rvd: {rvd}")
                dice_sum_old += dice_old
                dice_sum += dice
                iou_sum += iou
                rvd_sum += rvd
                data_num += 1

                if args.save_output:
                    nib.save(
                        nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                        os.path.join(output_directory, img_name),
                    )


            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    info_if_main(f"[⚠️ 跳过] 第 {idx} 个样本（{img_name}） OOM")
                    info_if_main(f"[DEBUG] shape={data.shape}, vessel={batch_data['label_vessel'].shape}")
                    info_if_main(
                        f"[CUDA] allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, "
                        f"reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB"
                    )
                    skipped += 1
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

                torch.cuda.empty_cache()

        dist.barrier()

        # ⚠️ 避免除 0 错误
        if data_num == 0:
            info_if_main("[⚠️ 警告] 所有样本都跳过了！")
            return 0, 0, 0, 0

        avg_dice = reduce_by_weight(dice_sum / data_num, data_num)
        avg_iou = reduce_by_weight(iou_sum / data_num, data_num)
        avg_rvd = reduce_by_weight(rvd_sum / data_num, data_num)
        avg_dice_old = reduce_by_weight(dice_sum_old / data_num, data_num)
        info_if_main(f"[验证完成] 跳过了 {skipped} 个样本，共处理 {data_num} 个样本")
        info_if_main("Overall Mean Dice_old: {}".format(np.mean(avg_dice_old)))
        info_if_main("Overall Mean Dice: {}".format(np.mean(avg_dice)))
        info_if_main("Overall Mean iou: {}".format(np.mean(avg_iou)))
        info_if_main("Overall Mean rvd: {}".format(np.mean(avg_rvd)))

        return avg_dice, avg_iou, avg_rvd, avg_dice_old


def safe_divide(a, b):
    EPS = 1e-10
    return (a + EPS) / (b + EPS)


def get_class(mask, class_id):
    if class_id == 0:
        return mask > 0
    else:
        return mask == class_id


def save_checkpoint(
        model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None
):
    state_dict = (
        model.state_dict() if not args.distributed else model.module.state_dict()
    )
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    info_if_main("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        args,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
        val_acc_max=0,
        scaler=None,
        loss_func2=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        info_if_main("Writing Tensorboard logs to ", args.logdir)
    scaler = scaler
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        # info_if_main(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss1, train_loss2 = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args,
            loss_func2=loss_func2,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        info_if_main(
            time.ctime(),
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss1),
            "contrast_loss: {:.4f}".format(
                train_loss2 if loss_func2 is not None else 0
            ),
            "lr:{:.6f}".format(learning_rate),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        if args.rank == 0 and writer is not None:
            if loss_func2 is not None:
                # writer.add_scalar("train_loss", train_loss1 + 0.1 * train_loss2, epoch)
                writer.add_scalar("train_loss", train_loss1, epoch)
                writer.add_scalar("contrast_loss", train_loss2, epoch)
            else:
                writer.add_scalar("train_loss", train_loss1, epoch)
            writer.add_scalar("lr", learning_rate, epoch)

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()

            avg_dice, avg_iou, avg_rvd, avg_dice_old = val_epoch_v4(model,
                                                                    val_loader,
                                                                    epoch=epoch,
                                                                    model_inferer=model_inferer,
                                                                    args=args,
                                                                    )

            avg_dice = np.mean(avg_dice)
            avg_iou = np.mean(avg_iou)
            avg_rvd = np.mean(avg_rvd)
            avg_dice_old = np.mean(avg_dice_old)

            if args.rank == 0:

                info_if_main(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "avg_dice",
                    avg_dice,
                    "avg_iou",
                    avg_iou,
                    "avg_rvd",
                    avg_rvd,
                    "avg_dice_old",
                    avg_dice_old,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    # writer.add_scalar("new_score", val_avg_acc, epoch)
                    # writer.add_scalar("val_acc", val_avg_dice, epoch)
                    writer.add_scalar("avg_dice", avg_dice, epoch)
                    writer.add_scalar("avg_iou", avg_iou, epoch)
                    writer.add_scalar("avg_rvd", avg_rvd, epoch)
                    writer.add_scalar("avg_dice_old", avg_dice_old, epoch)

                if avg_dice > val_acc_max:
                    info_if_main(
                        "new best ({:.6f} --> {:.6f}). ".format(
                            val_acc_max, avg_dice
                        )
                    )
                    val_acc_max = avg_dice
                    if (
                            args.rank == 0
                            and args.logdir is not None
                            and args.save_checkpoint
                    ):
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max)
                        if loss_func2 is not None:
                            loss_func2.save(os.path.join(args.logdir, args.memobank_path))
                # if (epoch + 1) % args.save_intervals == 0:
                #     save_checkpoint(
                #         model,
                #         epoch,
                #         args,
                #         best_acc=val_acc_max,
                #         filename=f"model_{epoch}.pt",
                #     )
                #     if loss_func2 is not None:
                #         loss_func2.save(os.path.join(args.logdir, args.memobank_path))

            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(
                    model,
                    epoch,
                    args,
                    best_acc=val_acc_max,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    filename="model_final.pt",
                )

        if scheduler is not None:
            scheduler.step()

    info_if_main("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
