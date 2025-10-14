import cv2
import numpy as np
import os
from pathlib import Path
def _is_background(image_bytes, entropy_thresh=5.0, edge_thresh=0.01, var_thresh=800):
    """Detect if image is a simple background"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Calculate entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # 2. Calculate edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size

        # 3. Calculate color variance
        variance = np.var(img)

        # Check if it's background
        is_bg = (entropy < entropy_thresh) and (edge_ratio < edge_thresh) and (variance < var_thresh)

        metrics = {
            'entropy': round(entropy, 2),
            'edge_ratio': round(edge_ratio, 4),
            'variance': round(variance, 2)
        }

        return is_bg
    except Exception as e:
        print(f"Error processing image: {e}")
        return False, None

MIN_IMAGE_SIZE = 1000
def _is_valid_size(image_bytes):
    return len(image_bytes) > MIN_IMAGE_SIZE

def is_valid_image(image_bytes):
    return _is_valid_size(image_bytes) and not _is_background(image_bytes)