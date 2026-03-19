import cv2
import numpy as np


def compute_optical_flow(frame_a_path, frame_c_path):
    """
    Compute dense optical flow from Frame A to Frame C
    using Farneback's algorithm.

    Args:
        frame_a_path: path to first frame
        frame_c_path: path to third frame

    Returns:
        flow: numpy array, shape (H, W, 2), dtype float32
              flow[y, x, 0] = horizontal displacement (dx)
              flow[y, x, 1] = vertical displacement (dy)
    """
    frame_a = cv2.imread(frame_a_path)
    frame_c = cv2.imread(frame_c_path)

    assert frame_a is not None, f"Failed to load {frame_a_path}"
    assert frame_c is not None, f"Failed to load {frame_c_path}"

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev=gray_a,
        next=gray_c,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    print(f"Flow shape: {flow.shape}")
    print(f"Max motion: dx={flow[:,:,0].max():.1f}, dy={flow[:,:,1].max():.1f}")

    return flow


def visualize_flow(flow, output_path):
    """
    Convert optical flow to a color image for visualization.
    Hue = direction of motion, Value = magnitude of motion.
    """
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:,:,0] = ang * 180 / np.pi / 2
    hsv[:,:,1] = 255
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, vis)
    print(f"Saved flow visualization to {output_path}")
    return vis


if __name__ == "__main__":
    flow = compute_optical_flow(
        "data/frames/frame_00000.png",
        "data/frames/frame_00002.png"
    )
    visualize_flow(flow, "outputs/results/flow_vis.png")
