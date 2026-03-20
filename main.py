import os
import json
import cv2

from src.extract import extract_frames
from src.triplets import create_triplets
from src.baseline import naive_blend
from src.flow import compute_optical_flow, visualize_flow
from src.warp import interpolate_with_flow
from src.interpolate import interpolate_with_occlusion
from src.evaluate import evaluate_triplet


def run_pipeline(video_path, num_triplets=10):
    """
    Run the complete mini-DLSS pipeline.

    Args:
        video_path:   path to input gameplay video
        num_triplets: how many triplets to evaluate
    """
    print("=" * 60)
    print("  MINI-DLSS FRAME INTERPOLATION PIPELINE")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[Step 1] Extracting frames...")
    frame_count = extract_frames(video_path, "data/frames/")

    if frame_count == 0:
        print("ERROR: No frames extracted. Check your video path.")
        return

    # Step 2: Create triplets
    print("\n[Step 2] Creating frame triplets...")
    triplets = create_triplets("data/frames/", "data/triplets/index.json")

    if not triplets:
        print("ERROR: No triplets created. Need at least 3 frames.")
        return

    # Step 3-7: Process each triplet
    os.makedirs("outputs/results", exist_ok=True)

    naive_psnr_total = 0
    naive_ssim_total = 0
    flow_psnr_total = 0
    flow_ssim_total = 0

    n = min(num_triplets, len(triplets))

    for i in range(n):
        t = triplets[i]
        print(f"\n--- Triplet {i+1}/{n} ---")

        # Step 3: Naive baseline
        naive = naive_blend(t["frame_a"], t["frame_c"])
        naive_scores = evaluate_triplet(naive, t["frame_b"])

        # Step 4-6: Flow-based interpolation with occlusion handling
        flow_result = interpolate_with_occlusion(t["frame_a"], t["frame_c"])
        flow_scores = evaluate_triplet(flow_result, t["frame_b"])

        # Accumulate scores
        naive_psnr_total += naive_scores["psnr"]
        naive_ssim_total += naive_scores["ssim"]
        flow_psnr_total += flow_scores["psnr"]
        flow_ssim_total += flow_scores["ssim"]

        print(f"  Naive: PSNR={naive_scores['psnr']:.2f} dB  SSIM={naive_scores['ssim']:.4f}")
        print(f"  Flow:  PSNR={flow_scores['psnr']:.2f} dB  SSIM={flow_scores['ssim']:.4f}")

        # Save sample outputs (first 3 triplets)
        if i < 3:
            cv2.imwrite(f"outputs/results/triplet_{i}_naive.png", naive)
            cv2.imwrite(f"outputs/results/triplet_{i}_flow.png", flow_result)
            gt = cv2.imread(t["frame_b"])
            cv2.imwrite(f"outputs/results/triplet_{i}_gt.png", gt)

            # Save flow visualization for first triplet
            if i == 0:
                from src.flow import compute_optical_flow
                flow = compute_optical_flow(t["frame_a"], t["frame_c"])
                visualize_flow(flow, "outputs/results/flow_vis.png")

    # Final Report
    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS (averaged over {n} triplets)")
    print("=" * 60)
    print(f"  Naive Blend:        PSNR={naive_psnr_total/n:.2f} dB  "
          f"SSIM={naive_ssim_total/n:.4f}")
    print(f"  Flow Interpolation: PSNR={flow_psnr_total/n:.2f} dB  "
          f"SSIM={flow_ssim_total/n:.4f}")
    print(f"  Improvement:        PSNR=+{(flow_psnr_total-naive_psnr_total)/n:.2f} dB  "
          f"SSIM=+{(flow_ssim_total-naive_ssim_total)/n:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline("data/videos/0320.mp4", num_triplets=10)
