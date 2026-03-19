import cv2
import numpy as np


def compute_occlusion_mask(flow_forward, flow_backward, threshold=1.0):
    """
    Detect occluded regions by checking flow consistency.

    Idea: If a pixel's forward flow and backward flow don't agree,
    that pixel is likely occluded (hidden in one frame).

    Args:
        flow_forward:  flow from A -> C, shape (H, W, 2)
        flow_backward: flow from C -> A, shape (H, W, 2)
        threshold:     error tolerance in pixels

    Returns:
        mask: numpy array (H, W), float32, range [0, 1]
              1.0 = visible in both frames (trust both warps)
              0.0 = occluded (trust only one warp)
    """
    h, w = flow_forward.shape[:2]

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + flow_forward[:,:,0]).astype(np.float32)
    map_y = (y_coords + flow_forward[:,:,1]).astype(np.float32)

    warped_back_flow = cv2.remap(flow_backward, map_x, map_y,
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    consistency = flow_forward + warped_back_flow
    error = np.sqrt(consistency[:,:,0]**2 + consistency[:,:,1]**2)

    mask = np.exp(-error / threshold)

    return mask.astype(np.float32)


def interpolate_with_occlusion(frame_a_path, frame_c_path):
    """
    Full interpolation pipeline with occlusion-aware blending.
    """
    from src.flow import compute_optical_flow
    from src.warp import warp_frame

    frame_a = cv2.imread(frame_a_path)
    frame_c = cv2.imread(frame_c_path)

    assert frame_a is not None, f"Failed to load {frame_a_path}"
    assert frame_c is not None, f"Failed to load {frame_c_path}"

    flow_ac = compute_optical_flow(frame_a_path, frame_c_path)  # A -> C
    flow_ca = compute_optical_flow(frame_c_path, frame_a_path)  # C -> A

    mask = compute_occlusion_mask(flow_ac, flow_ca)

    warped_a = warp_frame(frame_a, flow_ac, t=0.5)
    warped_c = warp_frame(frame_c, flow_ca, t=0.5)

    mask_3ch = np.stack([mask]*3, axis=-1)

    predicted = (mask_3ch * warped_a.astype(np.float32) +
                 (1 - mask_3ch) * warped_c.astype(np.float32))
    predicted = np.clip(predicted, 0, 255).astype(np.uint8)

    return predicted


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result = interpolate_with_occlusion(
        "data/frames/frame_00000.png",
        "data/frames/frame_00002.png"
    )
    cv2.imwrite("outputs/results/occlusion_interpolated.png", result)
    print(f"Saved occlusion-aware interpolation, shape: {result.shape}")
