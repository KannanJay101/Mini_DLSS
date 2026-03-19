import cv2
import numpy as np


def warp_frame(frame, flow, t=0.5):
    """
    Warp a frame using optical flow, moving pixels by t * flow.

    For interpolation at the midpoint:
      - Warp frame_a FORWARD with t = +0.5 (push pixels halfway)
      - Warp frame_c BACKWARD with t = -0.5 (pull pixels back halfway)

    Args:
        frame: numpy array (H, W, 3), the image to warp
        flow:  numpy array (H, W, 2), optical flow field
        t:     float, interpolation factor
               +0.5 = warp forward to midpoint
               -0.5 = warp backward to midpoint

    Returns:
        warped: numpy array (H, W, 3), the warped image
    """
    h, w = frame.shape[:2]

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (x_coords - t * flow[:,:,0]).astype(np.float32)
    map_y = (y_coords - t * flow[:,:,1]).astype(np.float32)

    warped = cv2.remap(
        src=frame,
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


def interpolate_with_flow(frame_a_path, frame_c_path, flow):
    """
    Generate a predicted middle frame using optical flow warping.

    Strategy:
        1. Warp Frame A forward to midpoint (using flow * 0.5)
        2. Warp Frame C backward to midpoint (using flow * -0.5)
        3. Blend both warped frames

    Args:
        frame_a_path: path to first frame
        frame_c_path: path to third frame
        flow: optical flow from A to C, shape (H, W, 2)

    Returns:
        predicted: blended result, numpy array (H, W, 3)
    """
    frame_a = cv2.imread(frame_a_path)
    frame_c = cv2.imread(frame_c_path)

    assert frame_a is not None, f"Failed to load {frame_a_path}"
    assert frame_c is not None, f"Failed to load {frame_c_path}"

    warped_a = warp_frame(frame_a, flow, t=0.5)
    warped_c = warp_frame(frame_c, flow, t=-0.5)

    predicted = cv2.addWeighted(warped_a, 0.5, warped_c, 0.5, 0)

    return predicted


if __name__ == "__main__":
    from flow import compute_optical_flow

    flow = compute_optical_flow(
        "data/frames/frame_00000.png",
        "data/frames/frame_00002.png"
    )

    result = interpolate_with_flow(
        "data/frames/frame_00000.png",
        "data/frames/frame_00002.png",
        flow
    )

    cv2.imwrite("outputs/results/flow_interpolated.png", result)
    print(f"Saved flow-based interpolation, shape: {result.shape}")
