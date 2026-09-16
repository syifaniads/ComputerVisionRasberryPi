# Model card — retained Raspberry Pi experiment

## Intended experiment

The retained notebook evaluates a lightweight transfer-learning path for edge classification on Raspberry Pi-class hardware.

## Dataset

Primary retained dataset: CIFAR-10, with ten classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`). The notebook sets `SAMPLE_SIZE = 10000` for the training experiment and resizes images to 224×224 for MobileNetV2.

## Architecture

- MobileNetV2 pretrained on ImageNet, `include_top=False`
- base layers frozen
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dense(10, softmax)

## Training evidence

The retained output shows validation accuracy values from 0.7390 at epoch 1 to 0.7855 at epoch 7. These are notebook training/validation logs, not Raspberry Pi inference accuracy measurements.

## Conversion

The notebook converts the Keras model to TensorFlow Lite and separately requests `tf.lite.Optimize.DEFAULT`. No representative-dataset calibration code is retained, so the portfolio does not describe the optimized artifact as calibrated full-int8 quantization.

## Limitations

CIFAR-10 classification is not representative of aerial robotics perception. No retained device benchmark, power measurement, thermal profile, confusion matrix, or field-captured evaluation dataset is presented as completed evidence.
