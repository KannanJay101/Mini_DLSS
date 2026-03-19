import cv2
import numpy as np


def compute_psnr(predicted, ground_truth):
    """
    Peak Signal-to-Noise Ratio.

    Measures raw pixel-level accuracy.
    Higher = better. Typical range: 25-45 dB.

    Formula:
        MSE  = mean( (predicted - ground_truth)^2 )
        PSNR = 10 * log10( MAX_PIXEL^2 / MSE )

    Args:
        predicted:    numpy array (H, W, 3), dtype uint8
        ground_truth: numpy array (H, W, 3), dtype uint8

    Returns:
        psnr: float (in dB)
    """
    assert predicted.shape == ground_truth.shape, \
        f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}"

    pred = predicted.astype(np.float64)
    gt = ground_truth.astype(np.float64)

    mse = np.mean((pred - gt) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)

    return psnr


def compute_ssim(predicted, ground_truth):
    """
    Structural Similarity Index.

    Measures perceived visual quality (structure, contrast, luminance).
    Range: 0 to 1. Higher = better. >0.95 is very good.

    Args:
        predicted:    numpy array (H, W, 3), dtype uint8
        ground_truth: numpy array (H, W, 3), dtype uint8

    Returns:
        ssim: float (0 to 1)
    """
    assert predicted.shape == ground_truth.shape, \
        f"Shape mismatch: {predicted.shape} vs {ground_truth.shape}"

    pred_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel_size = 11
    mu_pred = cv2.GaussianBlur(pred_gray, (kernel_size, kernel_size), 1.5)
    mu_gt = cv2.GaussianBlur(gt_gray, (kernel_size, kernel_size), 1.5)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = cv2.GaussianBlur(pred_gray ** 2, (kernel_size, kernel_size), 1.5) - mu_pred_sq
    sigma_gt_sq = cv2.GaussianBlur(gt_gray ** 2, (kernel_size, kernel_size), 1.5) - mu_gt_sq
    sigma_pred_gt = cv2.GaussianBlur(pred_gray * gt_gray, (kernel_size, kernel_size), 1.5) - mu_pred_gt

    numerator = (2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)
    denominator = (mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2)

    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


def evaluate_triplet(predicted, ground_truth_path):
    """
    Run full evaluation on a single predicted frame.
    """
    gt = cv2.imread(ground_truth_path)
    assert gt is not None, f"Failed to load {ground_truth_path}"

    psnr = compute_psnr(predicted, gt)
    ssim = compute_ssim(predicted, gt)

    return {"psnr": psnr, "ssim": ssim}


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.baseline import naive_blend
    from src.interpolate import interpolate_with_occlusion

    gt_path = "data/frames/frame_00001.png"

    naive = naive_blend("data/frames/frame_00000.png",
                        "data/frames/frame_00002.png")
    naive_scores = evaluate_triplet(naive, gt_path)

    flow_result = interpolate_with_occlusion(
        "data/frames/frame_00000.png",
        "data/frames/frame_00002.png"
    )
    flow_scores = evaluate_triplet(flow_result, gt_path)

    print("=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    print(f"Naive Blend:        PSNR={naive_scores['psnr']:.2f} dB  "
          f"SSIM={naive_scores['ssim']:.4f}")
    print(f"Flow Interpolation: PSNR={flow_scores['psnr']:.2f} dB  "
          f"SSIM={flow_scores['ssim']:.4f}")
    print(f"Improvement:        PSNR=+{flow_scores['psnr']-naive_scores['psnr']:.2f} dB  "
          f"SSIM=+{flow_scores['ssim']-naive_scores['ssim']:.4f}")
