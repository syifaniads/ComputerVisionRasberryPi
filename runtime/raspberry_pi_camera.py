from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.vision_inference import classify_output, preprocess_image, read_labels


def load_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError as exc:
        raise RuntimeError(
            "tflite_runtime is required on the Raspberry Pi runtime environment"
        ) from exc

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def open_camera():
    try:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
        camera.start()
        return "picamera2", camera
    except Exception:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("unable to open PiCamera2 or OpenCV camera 0")
        return "opencv", capture


def get_frame(kind: str, camera) -> np.ndarray:
    if kind == "picamera2":
        frame = camera.capture_array()
        # PiCamera2 arrays may be RGB/RGBA depending on configuration.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("camera opened but frame capture failed")
    return frame


def close_camera(kind: str, camera) -> None:
    if kind == "picamera2":
        camera.stop()
    else:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow Lite camera classifier")
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    labels = read_labels(args.labels)
    interpreter = load_interpreter(args.model)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    _, height, width, _ = input_details["shape"]

    kind, camera = open_camera()
    try:
        while True:
            frame = get_frame(kind, camera)
            tensor = preprocess_image(frame, int(width), int(height))

            started = time.perf_counter()
            interpreter.set_tensor(input_details["index"], tensor)
            interpreter.invoke()
            elapsed_ms = (time.perf_counter() - started) * 1000

            prediction = interpreter.get_tensor(output_details["index"])
            label, confidence, _ = classify_output(prediction, labels)
            display_label = label if confidence >= args.threshold else "uncertain"

            cv2.putText(
                frame,
                f"{display_label}: {confidence:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"inference: {elapsed_ms:.1f} ms",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Raspberry Pi Edge Vision", frame)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        close_camera(kind, camera)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
