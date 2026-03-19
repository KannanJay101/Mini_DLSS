import cv2
import numpy as np


def naive_blend(frame_a_path, frame_c_path):
    """
    Create a predicted middle frame by averaging Frame A and Frame C.

    This is the simplest baseline — no motion estimation.
    Moving objects will appear as ghostly double images.

    Args:
        frame_a_path: path to first frame
        frame_c_path: path to third frame

    Returns:
        predicted: numpy array (H, W, 3), dtype uint8
    """
    frame_a = cv2.imread(frame_a_path)
    frame_c = cv2.imread(frame_c_path)

    assert frame_a is not None, f"Failed to load {frame_a_path}"
    assert frame_c is not None, f"Failed to load {frame_c_path}"

    a_float = frame_a.astype(np.float32)
    c_float = frame_c.astype(np.float32)

    predicted = 0.5 * a_float + 0.5 * c_float
    predicted = np.clip(predicted, 0, 255).astype(np.uint8)

    return predicted


def weighted_blend(frame_a_path, frame_c_path, alpha=0.5):
    """
    Blend with adjustable weight.
    alpha=0.5 -> equal blend (same as naive)
    alpha=0.3 -> closer to frame A
    alpha=0.7 -> closer to frame C
    """
    frame_a = cv2.imread(frame_a_path)
    frame_c = cv2.imread(frame_c_path)

    assert frame_a is not None, f"Failed to load {frame_a_path}"
    assert frame_c is not None, f"Failed to load {frame_c_path}"

    frame_a = frame_a.astype(np.float32)
    frame_c = frame_c.astype(np.float32)

    predicted = (1.0 - alpha) * frame_a + alpha * frame_c
    predicted = np.clip(predicted, 0, 255).astype(np.uint8)

    return predicted


if __name__ == "__main__":
    result = naive_blend("data/frames/frame_00000.png",
                         "data/frames/frame_00002.png")
    cv2.imwrite("outputs/results/naive_blend.png", result)
    print(f"Saved naive blend, shape: {result.shape}")
