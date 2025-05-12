# Step by step Raspberry Comvis

#!/bin/bash
# setup_vision_raspi.sh
# Skrip untuk menyiapkan Computer Vision di Raspberry Pi

echo "========================================================"
echo "Persiapan Computer Vision di Raspberry Pi"
echo "========================================================"

# 1. Update sistem
echo "[STEP 1] Update sistem..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Instal dependensi
echo "[STEP 2] Instal dependensi..."
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libcblas-dev
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libjasper-dev || echo "Paket libjasper-dev mungkin tidak tersedia, melanjutkan..."
sudo apt-get install -y libqt4-test || echo "Paket libqt4-test mungkin tidak tersedia, melanjutkan..."
sudo apt-get install -y libqtgui4 || echo "Paket libqtgui4 mungkin tidak tersedia, melanjutkan..."
sudo apt-get install -y libtiff5-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y python3-picamera2 || echo "Menginstal libcamera-dev sebagai alternatif..."
[ $? -ne 0 ] && sudo apt-get install -y libcamera-dev

# 3. Instal paket Python
echo "[STEP 3] Instal package Python..."
python3 -m pip install --upgrade pip
pip3 install opencv-python-headless || pip3 install opencv-python
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl || pip3 install tflite-runtime
pip3 install numpy
pip3 install pillow
pip3 install matplotlib

# 4. Buat direktori project
echo "[STEP 4] Membuat direktori project..."
mkdir -p ~/raspi_vision_project
cd ~/raspi_vision_project

# 5. Download sampel program
echo "[STEP 5] Download kode program..."
cat > ~/raspi_vision_project/vision_detect.py << 'EOL'
#!/usr/bin/env python3
# vision_detect.py - Program deteksi objek pada Raspberry Pi

import os
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2  # Untuk Raspberry Pi OS Bullseye dan yg lebih baru

# Fungsi untuk memproses gambar input
def preprocess_image(image, input_size):
    # Ubah ukuran gambar
    image = cv2.resize(image, input_size)
    # Konversi BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalisasi
    image = image.astype(np.float32) / 255.0
    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)
    return image

# Fungsi untuk menjalankan inferensi
def run_inference(interpreter, image):
    # Dapatkan detail tentang input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Dapatkan input shape
    _, height, width, _ = input_details[0]['shape']
    input_size = (width, height)
    
    # Preprocess gambar sesuai ukuran input model
    processed_image = preprocess_image(image, input_size)
    
    # Set tensor input
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Jalankan inferensi
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    
    # Dapatkan hasil
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data, inference_time, input_size

# Load label kelas
def load_labels(label_path):
    if not os.path.exists(label_path):
        print(f"File label tidak ditemukan: {label_path}")
        # Default CIFAR-10 class names jika tidak ada file
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    # Cek apakah model ada
    model_path = 'vision_model_quantized.tflite'
    label_path = 'class_names.txt'
    
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan: {model_path}")
        print("Silakan transfer model ke Raspberry Pi terlebih dahulu.")
        print("Contoh: scp vision_model_quantized.tflite pi@<raspberry-ip>:~/raspi_vision_project/")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    # Mendapatkan ukuran input model
    input_details = interpreter.get_input_details()
    _, input_height, input_width, _ = input_details[0]['shape']
    print(f"Model input size: {input_width}x{input_height}")
    
    # Load label
    labels = load_labels(label_path)
    print(f"Classes: {labels}")
    
    # Siapkan kamera
    print("Initializing camera...")
    try:
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(preview_config)
        picam2.start()
        use_picamera = True
    except Exception as e:
        print(f"Error initializing PiCamera2: {e}")
        print("Trying to use regular webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        use_picamera = False
    
    # Loop utama deteksi
    try:
        print("Starting detection. Press 'q' to quit.")
        
        while True:
            # Capture frame
            if use_picamera:
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break
            
            # Jalankan inferensi
            prediction, inference_time, _ = run_inference(interpreter, frame)
            
            # Dapatkan kelas dengan probabilitas tertinggi
            class_id = np.argmax(prediction[0])
            confidence = prediction[0][class_id]
            
            # Tampilkan hasil pada frame
            label_text = f"{labels[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tampilkan info kinerja
            fps_text = f"Inference: {inference_time*1000:.1f}ms"
            cv2.putText(frame, fps_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Computer Vision - Raspberry Pi', frame)
            
            # Keluar dengan menekan 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program dihentikan oleh pengguna")
        
    finally:
        cv2.destroyAllWindows()
        if use_picamera:
            picam2.stop()
        else:
            cap.release()
        print("Program selesai")

if __name__ == "__main__":
    main()
EOL

# 6. Buat file untuk mengunduh model dari URL (jika ada)
cat > ~/raspi_vision_project/download_model.py << 'EOL'
#!/usr/bin/env python3
# download_model.py - Script untuk mengunduh model dari URL

import os
import sys
import urllib.request
import argparse

def download_file(url, destination):
    """
    Download file dari URL ke destination
    """
    print(f"Downloading {url} ke {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Download berhasil: {destination}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download model Computer Vision')
    parser.add_argument('--model_url', type=str, help='URL untuk model TFLite')
    parser.add_argument('--labels_url', type=str, help='URL untuk file label')
    
    args = parser.parse_args()
    
    # Cek apakah URL disediakan
    if not args.model_url and not args.labels_url:
        print("Masukkan URL untuk mengunduh model atau file label.")
        print("Contoh: python download_model.py --model_url https://contoh.com/model.tflite --labels_url https://contoh.com/labels.txt")
        return
    
    # Download model jika URL disediakan
    if args.model_url:
        model_path = os.path.join(os.getcwd(), "vision_model_quantized.tflite")
        download_file(args.model_url, model_path)
    
    # Download label jika URL disediakan
    if args.labels_url:
        labels_path = os.path.join(os.getcwd(), "class_names.txt")
        download_file(args.labels_url, labels_path)

if __name__ == "__main__":
    main()
EOL

# 7. Buat file README
cat > ~/raspi_vision_project/README.md << 'EOL'
# Computer Vision pada Raspberry Pi

## Persiapan
1. Pastikan Raspberry Pi Anda sudah terinstal dengan Raspberry Pi OS terbaru
2. Jalankan skrip setup: `bash setup_vision_raspi.sh`

## Transfer Model
Sebelum menjalankan deteksi, Anda perlu mentransfer model yang telah dilatih ke Raspberry Pi:

### Opsi 1: Download dari URL
Jika model Anda tersedia secara online:
```
python download_model.py --model_url https://your-url/vision_model_quantized.tflite --labels_url https://your-url/class_names.txt
```

### Opsi 2: Transfer dari komputer lokal
Dari komputer lokal Anda, gunakan perintah scp:
```
scp vision_model_quantized.tflite pi@<raspberry-pi-ip>:~/raspi_vision_project/
scp class_names.txt pi@<raspberry-pi-ip>:~/raspi_vision_project/
```

## Menjalankan Deteksi
```
cd ~/raspi_vision_project
python vision_detect.py
```

## Keluar dari program
Tekan 'q' saat jendela preview aktif untuk keluar dari program deteksi.
EOL

# 8. Buat skrip executable
chmod +x ~/raspi_vision_project/vision_detect.py
chmod +x ~/raspi_vision_project/download_model.py

echo "========================================================"
echo "Setup selesai! File ditempatkan di ~/raspi_vision_project/"
echo "Untuk menjalankan deteksi, transfer file model Anda terlebih dahulu,"
echo "kemudian jalankan: python ~/raspi_vision_project/vision_detect.py"
echo "========================================================"
