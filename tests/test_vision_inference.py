import numpy as np

from src.vision_inference import classify_output, preprocess_image


def test_preprocess_resizes_normalizes_and_adds_batch_axis() -> None:
    frame = np.full((10, 20, 3), 255, dtype=np.uint8)
    output = preprocess_image(frame, width=224, height=224)

    assert output.shape == (1, 224, 224, 3)
    assert output.dtype == np.float32
    assert float(output.min()) == 1.0
    assert float(output.max()) == 1.0


def test_preprocess_converts_bgr_to_rgb() -> None:
    blue_bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
    output = preprocess_image(blue_bgr, width=1, height=1)
    assert output[0, 0, 0].tolist() == [0.0, 0.0, 1.0]


def test_classify_output_returns_top_class() -> None:
    label, confidence, class_id = classify_output(
        np.array([[0.1, 0.7, 0.2]], dtype=np.float32),
        ["airplane", "automobile", "bird"],
    )
    assert label == "automobile"
    assert class_id == 1
    assert abs(confidence - 0.7) < 1e-6
