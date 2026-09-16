from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def preprocess_image(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a BGR frame, convert to RGB, normalize to [0, 1], and add batch axis."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr must contain image data")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    image = cv2.resize(frame_bgr, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def classify_output(
    prediction: np.ndarray, labels: Sequence[str]
) -> tuple[str, float, int]:
    """Return `(label, confidence, class_id)` from one classifier output tensor."""
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1:
        raise ValueError("prediction must be shape (classes,) or (1, classes)")
    if len(labels) != values.shape[0]:
        raise ValueError("label count must match model output classes")

    class_id = int(np.argmax(values))
    return labels[class_id], float(values[class_id]), class_id


def read_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError("label file is empty")
    return labels
