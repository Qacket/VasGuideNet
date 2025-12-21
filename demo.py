import argparse
import os
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)
from monai.data import MetaTensor
from monai.utils import set_determinism

# 你项目里的模型
from lib.models.VesselEnhancedNet import VesselEnhancedNet


def smart_load_state_dict(ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # 兼容 DDP / 非 DDP：自动去掉或补齐 module.
    # 规则：如果当前 sd 大部分 key 以 module. 开头，就先去掉；
    # 否则保持原样。
    keys = list(sd.keys())
    if len(keys) == 0:
        raise ValueError("Empty state_dict in checkpoint.")

    module_ratio = sum(k.startswith("module.") for k in keys) / len(keys)
    if module_ratio > 0.5:
        new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        return new_sd
    return sd


@torch.no_grad()
def infer_one_volume(
    model,
    image_tensor: torch.Tensor,  # (1, 1, H, W, D) or (1, 1, D, H, W) depending on your transforms
    roi_size=(96, 96, 32),
    sw_batch_size=4,
    overlap=0.7,
):
    # 输出 logits: (B, C, ...)
    logits = sliding_window_inference(
        inputs=image_tensor,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )
    # softmax -> argmax 得到 label
    prob = torch.softmax(logits, dim=1)
    pred = torch.argmax(prob, dim=1, keepdim=True)  # (B, 1, ...)
    return pred, prob


def save_nifti_like(meta, pred_tensor: torch.Tensor, out_path: str):
    """
    pred_tensor: (1, 1, D, H, W) or (1, 1, H, W, D) 取决于前面 transforms
    这里直接用 MONAI MetaTensor 的元信息写回：最省事的方式是用 monai.data.MetaTensor + nibabel
    但为了脚本更“少依赖”，我们用 monai 的 SaveImage 也可以。
    """
    from monai.transforms import SaveImaged

    # 构造一个 dict，复用原图的 meta
    data = {
        "pred": MetaTensor(pred_tensor[0], meta=meta)  # 去掉 batch 维
    }
    saver = SaveImaged(
        keys="pred",
        output_dir=os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
        output_postfix="",
        output_ext=os.path.splitext(out_path)[1] if out_path.endswith(".nii") else ".nii.gz",
        separate_folder=False,
        resample=False,  # 不要再 resample，保持与当前空间一致
        print_log=False,
    )
    saver(data)

    # SaveImaged 会按 output_postfix/output_ext 生成文件名；我们强制 rename 成用户给的 out_path
    # 它默认输出名是 pred{postfix}{ext}，在 output_dir 下
    gen_name = os.path.join(
        os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
        "pred" + (".nii.gz" if out_path.endswith(".nii.gz") else ".nii"),
    )
    if os.path.abspath(gen_name) != os.path.abspath(out_path):
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(gen_name, out_path)


def export_quick_png(pred: torch.Tensor, img: torch.Tensor, out_dir: str, num_slices: int = 12):
    """
    快速导出一些切片 png（不追求完美可视化，只求别人一眼看到“有输出”）。
    pred/img: (1, 1, D, H, W) or (1, 1, H, W, D)
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # 尝试判断维度布局：常见是 (B,1,D,H,W)
    x = img[0, 0].detach().cpu().float().numpy()
    y = pred[0, 0].detach().cpu().numpy().astype(np.int32)

    # 取最后一维当作 slice 轴（保守做法：哪个维度最大就当 slice 轴）
    axis = int(np.argmax(x.shape))
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)

    z = x.shape[-1]
    idxs = np.linspace(0, z - 1, num_slices).round().astype(int)

    for i, k in enumerate(idxs):
        fig = plt.figure()
        plt.imshow(x[..., k], cmap="gray")
        # 简单叠加：mask >0 的地方加一层
        m = y[..., k]
        plt.imshow((m > 0).astype(np.float32), alpha=0.25)
        plt.title(f"slice={k}")
        plt.axis("off")
        fig.savefig(os.path.join(out_dir, f"slice_{i:02d}_{k:04d}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def build_preprocess(args):
    """
    这里用 MONAI 的通用预处理：
    - Load NIfTI
    - 保证通道在前
    - 统一方向（RAS）
    - 统一 spacing（与你训练参数一致可改）
    - 强度归一化（ScaleIntensityRanged，用你训练时的 a_min/a_max/b_min/b_max）
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear",),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


def main():
    parser = argparse.ArgumentParser("VesselEnhancedNet demo inference")
    parser.add_argument("--ckpt", required=True, type=str, help="path to trained checkpoint")
    parser.add_argument("--image", required=True, type=str, help="input image path (.nii/.nii.gz recommended)")
    parser.add_argument("--out", required=True, type=str, help="output seg path (.nii or .nii.gz)")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=9, type=int)

    # 和你训练脚本一致的关键参数（可直接沿用默认）
    parser.add_argument("--a_min", default=-10.0, type=float)
    parser.add_argument("--a_max", default=225.0, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)
    parser.add_argument("--space_x", default=1.5, type=float)
    parser.add_argument("--space_y", default=1.5, type=float)
    parser.add_argument("--space_z", default=2.0, type=float)

    # sliding window
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=32, type=int)
    parser.add_argument("--sw_batch_size", default=4, type=int)
    parser.add_argument("--overlap", default=0.7, type=float)

    # 可视化
    parser.add_argument("--save_png", action="store_true", help="export quick png slices")
    parser.add_argument("--png_dir", default="demo_png", type=str)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    set_determinism(seed=args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) build model
    model = VesselEnhancedNet(
        in_channels=args.in_channels,
        num_classes=args.out_channels,
    ).to(device)
    model.eval()

    # 2) load weights
    sd = smart_load_state_dict(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  missing keys (first 20):", missing[:20])
    if len(unexpected) > 0:
        print("  unexpected keys (first 20):", unexpected[:20])

    # 3) preprocess
    preprocess = build_preprocess(args)
    batch = preprocess({"image": args.image})
    img = batch["image"]  # MetaTensor: (1, H, W, D) or (1, D, H, W) after EnsureChannelFirstd
    meta = img.meta if hasattr(img, "meta") else {}

    # add batch dim -> (1, 1, ...)
    img = img.unsqueeze(0).to(device)

    # 4) inference
    pred, prob = infer_one_volume(
        model,
        img,
        roi_size=(args.roi_x, args.roi_y, args.roi_z),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )

    # 5) save output (nifti)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    save_nifti_like(meta, pred.cpu(), args.out)
    print(f"[ok] saved seg -> {args.out}")

    # 6) quick png
    if args.save_png:
        export_quick_png(pred.cpu(), img.cpu(), args.png_dir)
        print(f"[ok] saved png slices -> {args.png_dir}")


if __name__ == "__main__":
    main()
