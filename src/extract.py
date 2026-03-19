import cv2
import os


def extract_frames(video_path, output_dir):
    """
    Extract all frames from a video file and save as PNGs.

    Args:
        video_path: path to .mp4 file (e.g. "data/videos/gameplay.mp4")
        output_dir: folder to save frames (e.g. "data/frames/")
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total} frames")

    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        filename = os.path.join(output_dir, f"frame_{frame_number:05d}.png")
        cv2.imwrite(filename, frame)
        frame_number += 1

    cap.release()
    print(f"Extracted {frame_number} frames to {output_dir}")
    return frame_number


if __name__ == "__main__":
    extract_frames("data/videos/gameplay.mp4", "data/frames/")
