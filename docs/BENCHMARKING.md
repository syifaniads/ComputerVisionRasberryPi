# Raspberry Pi benchmarking plan

A credible edge deployment should measure the device, not infer performance from Colab.

For each model artifact, record:

- Raspberry Pi model and RAM;
- OS release, Python version, CPU architecture;
- TensorFlow Lite runtime version;
- model file size and input tensor shape/dtype;
- warm-up iterations;
- at least 100 timed inference iterations;
- p50 / p95 / p99 inference latency;
- end-to-end frame rate including capture and preprocessing;
- CPU utilization, memory, temperature, and throttling state;
- accuracy on a held-out task-appropriate dataset.

Use `time.perf_counter()` around `interpreter.invoke()` for model-only latency and a separate timer around capture → preprocessing → inference → overlay for end-to-end latency.

The current repository intentionally leaves device benchmark values blank because no retained reproducible Raspberry Pi measurement artifact supports them.
