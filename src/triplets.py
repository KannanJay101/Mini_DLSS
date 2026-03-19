import os
import json


def create_triplets(frames_dir, output_file, step=1):
    """
    Create frame triplets from sequential frames.

    Args:
        frames_dir: folder containing frame_00000.png, frame_00001.png, ...
        output_file: path to save triplet index (JSON)
        step: gap between frames (1 = consecutive, 2 = skip one, etc.)
              larger step = more motion between frames = harder task

    Returns:
        list of triplets: [{"frame_a": ..., "frame_b": ..., "frame_c": ...}, ...]
    """
    all_frames = sorted([
        f for f in os.listdir(frames_dir) if f.endswith('.png')
    ])

    print(f"Found {len(all_frames)} frames")

    triplets = []

    for i in range(0, len(all_frames) - 2 * step):
        frame_a = os.path.join(frames_dir, all_frames[i])
        frame_b = os.path.join(frames_dir, all_frames[i + step])       # middle (ground truth)
        frame_c = os.path.join(frames_dir, all_frames[i + 2 * step])   # end

        triplets.append({
            "frame_a": frame_a,
            "frame_b": frame_b,
            "frame_c": frame_c,
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(triplets, f, indent=2)

    print(f"Created {len(triplets)} triplets (step={step})")
    if triplets:
        print(f"Example: {triplets[0]}")

    return triplets


if __name__ == "__main__":
    triplets = create_triplets("data/frames/", "data/triplets/index.json", step=1)
