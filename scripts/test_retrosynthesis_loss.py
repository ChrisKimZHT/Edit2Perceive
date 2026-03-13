import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils.cycle_loss import get_cycle_consistency_retrosynthesis_loss


def pil_to_tensor_bchw(image: Image.Image, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    ten = ten * 2.0 - 1.0
    return ten


def tensor_chw_to_np01(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    return x


def make_toy_pair(size: int = 512) -> tuple[Image.Image, Image.Image]:
    gt = Image.new("RGB", (size, size), (255, 255, 255))
    pred = Image.new("RGB", (size, size), (255, 255, 255))
    d_gt = ImageDraw.Draw(gt)
    d_pred = ImageDraw.Draw(pred)

    # GT: simple ring-like structure and bonds
    d_gt.ellipse((120, 120, 260, 260), outline=(0, 0, 0), width=5)
    d_gt.line((260, 190, 380, 190), fill=(0, 0, 0), width=5)
    d_gt.line((380, 190, 450, 130), fill=(0, 0, 0), width=5)
    d_gt.line((380, 190, 450, 250), fill=(0, 0, 0), width=5)

    # Pred: with shifts and blur-like thicker strokes (simulated errors)
    d_pred.ellipse((125, 125, 270, 270), outline=(0, 0, 0), width=7)
    d_pred.line((270, 198, 385, 198), fill=(0, 0, 0), width=7)
    d_pred.line((385, 198, 450, 150), fill=(0, 0, 0), width=7)

    return pred, gt


def compute_debug_maps(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6):
    # The implementation mirrors get_cycle_consistency_retrosynthesis_loss.
    pred01 = (pred.clamp(-1, 1) + 1.0) / 2.0
    gt01 = (gt.clamp(-1, 1) + 1.0) / 2.0

    gt_gray = 0.299 * gt01[:, 0:1] + 0.587 * gt01[:, 1:2] + 0.114 * gt01[:, 2:3]
    pred_gray = 0.299 * pred01[:, 0:1] + 0.587 * pred01[:, 1:2] + 0.114 * pred01[:, 2:3]
    fg_soft = (1.0 - gt_gray).clamp(0.0, 1.0)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[1] != 1:
            mask = mask[:, :1]
        fg_soft = fg_soft * mask.to(device=pred.device, dtype=pred.dtype)

    diff = pred01 - gt01
    charbonnier = torch.sqrt(diff * diff + eps * eps)
    fg_weight = 1.0 + 4.0 * fg_soft
    recon_map = charbonnier.mean(dim=1, keepdim=True) * fg_weight
    recon_loss = recon_map.mean()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
    gt_gx = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_gy = F.conv2d(gt_gray, sobel_y, padding=1)

    edge_weight = 1.0 + 6.0 * fg_soft
    edge_map = (torch.abs(pred_gx - gt_gx) + torch.abs(pred_gy - gt_gy)) * edge_weight
    edge_loss = edge_map.mean()

    total_map = recon_map + 0.5 * edge_map
    total_loss = recon_loss + 0.5 * edge_loss

    return {
        "pred01": pred01,
        "gt01": gt01,
        "gt_gray": gt_gray,
        "pred_gray": pred_gray,
        "fg_soft": fg_soft,
        "fg_weight": fg_weight,
        "recon_map": recon_map,
        "edge_map": edge_map,
        "total_map": total_map,
        "recon_loss": recon_loss,
        "edge_loss": edge_loss,
        "total_loss": total_loss,
    }


def save_visualization(debug, out_file: str):
    pred_img = debug["pred01"][0].detach().cpu().permute(1, 2, 0).numpy()
    gt_img = debug["gt01"][0].detach().cpu().permute(1, 2, 0).numpy()

    fg_soft = debug["fg_soft"][0, 0].detach().cpu().numpy()
    fg_weight = debug["fg_weight"][0, 0].detach().cpu().numpy()
    recon_map = debug["recon_map"][0, 0].detach().cpu().numpy()
    edge_map = debug["edge_map"][0, 0].detach().cpu().numpy()
    total_map = debug["total_map"][0, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title("GT image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pred_img)
    axes[0, 1].set_title("Pred image")
    axes[0, 1].axis("off")

    abs_diff = np.abs(pred_img - gt_img).mean(axis=2)
    im = axes[0, 2].imshow(abs_diff, cmap="magma")
    axes[0, 2].set_title("Abs diff (RGB mean)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[0, 3].imshow(fg_soft, cmap="viridis")
    axes[0, 3].set_title("Foreground soft prior")
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(fg_weight, cmap="viridis")
    axes[1, 0].set_title("Foreground weight")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(recon_map, cmap="magma")
    axes[1, 1].set_title("Reconstruction map")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(edge_map, cmap="magma")
    axes[1, 2].set_title("Edge map")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im = axes[1, 3].imshow(total_map, cmap="magma")
    axes[1, 3].set_title("Total weighted map")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test and visualize retrosynthesis extra loss.")
    parser.add_argument("--pred", type=str, default=None, help="Path to predicted image.")
    parser.add_argument("--gt", type=str, default=None, help="Path to GT image.")
    parser.add_argument("--mask", type=str, default=None, help="Optional mask image path. Non-zero area is valid.")
    parser.add_argument("--size", type=int, default=768, help="Resize side length.")
    parser.add_argument("--out_dir", type=str, default="./samples/loss_debug", help="Output directory.")
    parser.add_argument("--toy", action="store_true", help="Use synthetic toy pair instead of file inputs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.toy:
        pred_pil, gt_pil = make_toy_pair(size=args.size)
    else:
        if args.pred is None or args.gt is None:
            raise ValueError("Please provide --pred and --gt, or use --toy.")
        pred_pil = Image.open(args.pred).convert("RGB")
        gt_pil = Image.open(args.gt).convert("RGB")

    pred = pil_to_tensor_bchw(pred_pil, size=args.size)
    gt = pil_to_tensor_bchw(gt_pil, size=args.size)

    mask = None
    if args.mask is not None:
        m = Image.open(args.mask).convert("L")
        if args.size is not None:
            m = m.resize((args.size, args.size), Image.Resampling.NEAREST)
        m_arr = (np.asarray(m, dtype=np.float32) > 0).astype(np.float32)
        mask = torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        loss_val = get_cycle_consistency_retrosynthesis_loss(pred, gt, mask=mask)
        debug = compute_debug_maps(pred, gt, mask=mask)

    out_png = os.path.join(args.out_dir, "retrosynthesis_loss_visualization.png")
    save_visualization(debug, out_png)

    out_txt = os.path.join(args.out_dir, "retrosynthesis_loss_values.txt")
    with open(out_txt, "w") as f:
        f.write(f"total_loss_from_function: {float(loss_val):.8f}\n")
        f.write(f"recon_loss: {float(debug['recon_loss']):.8f}\n")
        f.write(f"edge_loss: {float(debug['edge_loss']):.8f}\n")
        f.write(f"total_loss_recomputed: {float(debug['total_loss']):.8f}\n")

    print("Saved visualization:", out_png)
    print("Saved scalar breakdown:", out_txt)
    print("total_loss_from_function:", float(loss_val))


if __name__ == "__main__":
    main()
