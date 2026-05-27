import math
from typing import List, Tuple, Dict

def generate_label_json(points: List[Tuple[int, int]], labels: List[str], offset_px: int = 10) -> List[Dict]:
    """
    Create label JSON data with leader line anchors and draggable positions.
    Each label will be offset slightly from its origin point.
    """
    label_data = []
    used_positions = set()

    for (x, y), text in zip(points, labels):
        # Try vertical offset until no overlap
        attempts = 0
        max_attempts = 10
        offset_y = -offset_px
        offset_x = 0
        while attempts < max_attempts:
            key = (x + offset_x, y + offset_y)
            if key not in used_positions:
                used_positions.add(key)
                break
            offset_y -= offset_px
            attempts += 1

        label_data.append({
            "text": text,
            "anchor": {"x": x, "y": y},
            "position": {"x": x + offset_x, "y": y + offset_y},
            "movable": True
        })

    return label_data

# EXAMPLE USAGE
if __name__ == "__main__":
    keypoints = [(100, 200), (120, 210), (140, 230)]
    texts = ["45.0 cm", "52.2 cm", "90.0 deg"]
    labels = generate_label_json(keypoints, texts)
    import json
    print(json.dumps(labels, indent=2))
