import cv2
import numpy as np
import os

def load_cap(source=None, path=None):
    """
    Load a video capture object from a camera index or an image-sequence pattern.
    
    :param source: Camera index (e.g., 0) 
    :param path: or an OpenCV image-sequence pattern (e.g. ``in%06d.jpg``).
    :return: cv2.VideoCapture object
    """
    if source is not None and isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {source}.")
        return cap
    elif path is not None and isinstance(path, str):
        directory = os.path.dirname(path) or os.curdir
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open image sequence: {path}")
        return cap
    else:
        raise ValueError("Either source (camera index) or path (folder) must be provided.")
