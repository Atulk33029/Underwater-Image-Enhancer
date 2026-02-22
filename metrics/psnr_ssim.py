import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------- PSNR ---------------- #
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ---------------- SSIM ---------------- #
def compute_ssim(img1, img2):
    return ssim(
        img1,
        img2,
        channel_axis=-1,   # NEW API
        data_range=255,
        win_size=7         # SAFE default
    )



# ---------------- Evaluation ---------------- #
def evaluate_psnr_ssim(gt_dir, enhanced_dir):
    psnr_scores = []
    ssim_scores = []

    files = os.listdir(gt_dir)

    for file in files:
        gt_path = os.path.join(gt_dir, file)
        enh_path = os.path.join(enhanced_dir, file)

        if not os.path.exists(enh_path):
            continue

        gt = cv2.imread(gt_path)
        enh = cv2.imread(enh_path)

        gt = cv2.resize(gt, (256, 256))
        enh = cv2.resize(enh, (256, 256))

        psnr_scores.append(compute_psnr(gt, enh))
        ssim_scores.append(compute_ssim(gt, enh))

    print("📊 PSNR & SSIM Results")
    print(f"Average PSNR : {np.mean(psnr_scores):.2f}")
    print(f"Average SSIM : {np.mean(ssim_scores):.4f}")


if __name__ == "__main__":
    evaluate_psnr_ssim(
        gt_dir="../data/test/target",
        enhanced_dir="../results/images"
    )
