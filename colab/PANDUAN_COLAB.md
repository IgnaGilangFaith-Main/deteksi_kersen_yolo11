# 🍒 Panduan Training YOLOv11 di Google Colab

## 📋 Langkah-Langkah

### 1️⃣ Persiapan Dataset

**Jalankan dulu di lokal:**

```bash
python scripts/1_organize_and_auto_label.py
python scripts/2_split_dataset.py
```

Pastikan folder `train_split/` sudah ada dengan struktur:

```
train_split/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

---

### 2️⃣ Upload Dataset ke Google Drive

1. Buka [Google Drive](https://drive.google.com)
2. Buat folder: `deteksi_kersen`
3. Upload folder `train_split/` ke dalam folder tersebut

**Struktur di Google Drive:**

```
MyDrive/
└── deteksi_kersen/
    └── train_split/
        ├── images/
        ├── labels/
        └── data.yaml
```

> 💡 **Tips:** Compress folder `train_split` menjadi ZIP, upload, lalu extract di Colab untuk upload lebih cepat.

---

### 3️⃣ Buka Notebook di Google Colab

**Opsi A: Upload langsung**

1. Buka [Google Colab](https://colab.research.google.com)
2. File → Upload notebook
3. Upload file `Train_YOLOv11_Kersen.ipynb`

**Opsi B: Dari Google Drive**

1. Upload notebook ke Google Drive
2. Klik kanan → Open with → Google Colaboratory

---

### 4️⃣ Aktifkan GPU

**PENTING! Lakukan sebelum menjalankan notebook:**

1. Klik menu **Runtime** → **Change runtime type**
2. Pilih **GPU** (T4 atau yang tersedia)
3. Klik **Save**

---

### 5️⃣ Jalankan Notebook

Jalankan cell satu per satu dari atas ke bawah:

| Cell | Fungsi                         |
| ---- | ------------------------------ |
| 1    | Install packages (ultralytics) |
| 2    | Cek GPU                        |
| 3    | Mount Google Drive             |
| 4    | Setup path                     |
| 5    | Copy dataset ke local          |
| 6    | Update data.yaml               |
| 7    | Verifikasi data                |
| 8    | Konfigurasi training           |
| 9    | Load model                     |
| 10   | **TRAINING** (30-60 menit)     |
| 11   | Evaluasi model                 |
| 12   | Tampilkan grafik               |
| 13   | Simpan ke Drive                |
| 14   | Download model                 |
| 15   | Test inference (opsional)      |

---

### 6️⃣ Download Model

Setelah training selesai:

1. Model akan otomatis di-download sebagai `yolo11s_kersen_best.pt`
2. Atau ambil dari Google Drive: `MyDrive/deteksi_kersen/results_colab/kersen_yolo11/weights/best.pt`

---

### 7️⃣ Gunakan Model di Lokal

1. Copy file `yolo11s_kersen_best.pt` ke folder `models/`
2. Jalankan aplikasi:
   ```bash
   python app.py
   ```
3. Buka browser: http://localhost:5000

---

## ⚙️ Konfigurasi yang Bisa Diubah

Di cell **Konfigurasi Training**, ubah sesuai kebutuhan:

```python
MODEL_NAME = "yolo11s.pt"   # Model: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
EPOCHS = 150                # Jumlah epoch (lebih banyak = lebih lama tapi akurat)
BATCH_SIZE = 16             # Batch size (kurangi jika error memory)
IMG_SIZE = 640              # Ukuran gambar (640 atau 416)
PATIENCE = 30               # Early stopping
LEARNING_RATE = 0.001       # Learning rate
```

### Rekomendasi Model:

| Model       | Ukuran | Kecepatan | Akurasi    | Rekomendasi     |
| ----------- | ------ | --------- | ---------- | --------------- |
| yolo11n     | 6 MB   | ⚡⚡⚡    | ⭐⭐       | Mobile/Edge     |
| **yolo11s** | 20 MB  | ⚡⚡      | ⭐⭐⭐     | **Balanced** ✅ |
| yolo11m     | 40 MB  | ⚡        | ⭐⭐⭐⭐   | Desktop         |
| yolo11l     | 80 MB  | 🐢        | ⭐⭐⭐⭐⭐ | Akurasi tinggi  |

---

## 🔧 Troubleshooting

### ❌ Error: GPU not available

- Pastikan sudah pilih GPU di Runtime → Change runtime type
- Coba restart runtime

### ❌ Error: CUDA out of memory

- Kurangi `BATCH_SIZE` menjadi 8 atau 4
- Kurangi `IMG_SIZE` menjadi 416

### ❌ Error: Dataset not found

- Pastikan path di Google Drive sudah benar
- Cek struktur folder sesuai panduan

### ❌ Training terlalu lama

- Kurangi `EPOCHS` menjadi 50-100
- Gunakan model lebih kecil: `yolo11n.pt`
- Kurangi `IMG_SIZE` menjadi 416

---

## 📊 Estimasi Waktu Training

| GPU | Model   | Epochs | Estimasi Waktu |
| --- | ------- | ------ | -------------- |
| T4  | yolo11s | 100    | ~30-45 menit   |
| T4  | yolo11s | 150    | ~45-60 menit   |
| T4  | yolo11m | 150    | ~90 menit      |
| CPU | yolo11s | 100    | ~4-6 jam       |

---

## ✅ Hasil yang Diharapkan

Setelah training, kamu akan mendapatkan:

1. **Model terbaik**: `best.pt` (~20 MB untuk yolo11s)
2. **Grafik training**: `results.png`
3. **Confusion Matrix**: `confusion_matrix.png`
4. **Metrik evaluasi**: mAP, Precision, Recall

**Target metrik yang bagus:**

- mAP50: > 0.85 (85%)
- Precision: > 0.80 (80%)
- Recall: > 0.80 (80%)

---

Selamat training! 🚀
