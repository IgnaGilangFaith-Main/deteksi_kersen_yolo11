# 🎯 Panduan Training untuk Akurasi 81%+ (OPTIMIZED)

## Deteksi Kematangan Kersen - YOLOv11

---

## ⚡ TL;DR - Quick Start

1. **Upload** folder `train_split/` ke Google Drive: `MyDrive/deteksi_kersen/`
2. **Buka** notebook `Train_YOLOv11_Kersen.ipynb` di Google Colab
3. **Jalankan** semua cell (Runtime → Run all)
4. **Download** model `yolo11s_kersen_best.pt` (akan otomatis)
5. **Pindahkan** ke folder `models/` di project lokal
6. **Restart** aplikasi Flask
7. **Selesai!** Akurasi target 81%+

---

## 🔧 Optimasi yang Sudah Diterapkan

### ✅ **Parameter Training Ultra Optimal:**

| Parameter         | Nilai         | Dampak                |
| ----------------- | ------------- | --------------------- |
| **Epochs**        | 300           | Learning sangat dalam |
| **Patience**      | 80            | Konvergensi sempurna  |
| **Optimizer**     | AdamW         | Akurasi maksimal      |
| **Learning Rate** | 0.01 → 0.0001 | Smooth decay          |
| **Batch Size**    | 16            | Optimal untuk GPU T4  |

### ✅ **Augmentasi SUPER Agresif:**

```python
# Augmentasi warna (PENTING untuk deteksi kematangan!)
hsv_s = 0.9          # Saturasi 90% - deteksi warna buah lebih akurat
hsv_v = 0.6          # Brightness variation
hsv_h = 0.025        # Hue variation

# Augmentasi geometrik
degrees = 30.0       # Rotasi 30°
translate = 0.2      # Translasi 20%
scale = 0.7          # Scaling
shear = 10.0         # Shear transform

# Augmentasi advanced
mosaic = 1.0         # Mosaic augmentation
mixup = 0.2          # Mixup 20%
copy_paste = 0.15    # Copy-paste 15%
auto_augment = 'randaugment'  # Random augmentation policy
erasing = 0.4        # Random erasing
```

### ✅ **Loss Optimization:**

```python
box = 7.5            # Box loss gain (fokus akurasi posisi)
cls = 0.5            # Classification loss
dfl = 1.5            # Distribution focal loss
```

### ✅ **Regularization:**

```python
dropout = 0.1        # Mencegah overfitting
weight_decay = 0.0005
label_smoothing = 0.05  # Generalisasi lebih baik
```

---

## 📊 Target Hasil Training

| Metric         | Target | Keterangan                           |
| -------------- | ------ | ------------------------------------ |
| **mAP50**      | ≥ 0.88 | Mean Average Precision @ IOU 0.5     |
| **mAP50-95**   | ≥ 0.72 | Mean Average Precision (COCO metric) |
| **Precision**  | ≥ 0.85 | Presisi minimal 85%                  |
| **Recall**     | ≥ 0.83 | Recall minimal 83%                   |
| **Confidence** | 75-85% | Confidence saat inference real-time  |

---

## 🚀 Langkah-Langkah Training

### 1️⃣ Persiapan Dataset

**Di komputer lokal, jalankan:**

```bash
cd deteksi_kersen
python scripts/1_organize_and_auto_label.py
python scripts/2_split_dataset.py
```

Pastikan folder `train_split/` sudah ada dengan struktur:

```
train_split/
├── images/
│   ├── train/  (70% data)
│   ├── val/    (20% data)
│   └── test/   (10% data)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### 2️⃣ Upload ke Google Drive

1. Buka [Google Drive](https://drive.google.com)
2. Buat folder: **`deteksi_kersen`**
3. Upload folder **`train_split/`** ke dalamnya

**Struktur final di Drive:**

```
MyDrive/
└── deteksi_kersen/
    └── train_split/
        ├── images/
        ├── labels/
        └── data.yaml
```

### 3️⃣ Buka Notebook di Colab

1. Upload file `Train_YOLOv11_Kersen.ipynb` ke Google Colab
2. **PENTING:** Aktifkan GPU
   - Klik **Runtime** → **Change runtime type**
   - **Hardware accelerator:** GPU (T4)
   - **Save**

### 4️⃣ Jalankan Training

**Opsi 1: Run All (Otomatis)**

- Klik **Runtime** → **Run all**
- Tunggu 60-120 menit

**Opsi 2: Run Step by Step (Recommended untuk pertama kali)**

- Jalankan cell satu per satu dengan **Shift+Enter**
- Perhatikan output setiap cell
- Pastikan tidak ada error sebelum lanjut

### 5️⃣ Monitor Training

Perhatikan metrics di output:

```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/300   4.2GB    1.234     0.567     0.890        128   640
  ...
 50/300   4.3GB    0.456     0.234     0.123        128   640
100/300   4.3GB    0.289     0.156     0.089        128   640
...
```

**Tanda training berjalan baik:**

- ✅ Loss turun secara konsisten
- ✅ mAP50 naik di validation
- ✅ Tidak ada "nan" atau "inf" di loss

**Grafik hasil akan muncul:**

- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `PR_curve.png` - Precision-Recall curve

### 6️⃣ Evaluasi Hasil

Notebook akan otomatis:

1. Evaluasi model dengan berbagai confidence (0.50 - 0.80)
2. Menampilkan metrics lengkap
3. Merekomendasikan confidence threshold optimal
4. Membuat file `CONFIG_OPTIMAL.txt`

**Contoh output:**

```
📊 REKOMENDASI CONFIDENCE THRESHOLD:
==================================================
✅ Confidence Optimal: 0.75
   - mAP50: 0.8823
   - Precision: 0.8654
   - Recall: 0.8321

💡 Gunakan confidence 0.75 di aplikasi Flask!
==================================================
```

### 7️⃣ Download Model

Model akan otomatis di-download:

- **File:** `yolo11s_kersen_best.pt`
- **Size:** ~22 MB
- **Location:** Downloads folder

### 8️⃣ Deploy ke Aplikasi

1. Copy file `yolo11s_kersen_best.pt` ke folder `models/`
2. File `app.py` sudah dikonfigurasi otomatis
3. Restart aplikasi Flask:

```bash
python app.py
```

4. Buka browser: `http://localhost:5000`
5. Confidence akan otomatis di **0.75** (optimal)

---

## 📈 Cara Membaca Hasil Training

### Grafik `results.png`

![Training Results](example_results.png)

**Yang harus diperhatikan:**

1. **Box Loss** - Harus turun dan stabil

   - Target akhir: < 0.5

2. **Class Loss** - Harus turun

   - Target akhir: < 0.3

3. **mAP50** - Harus naik

   - Target: ≥ 0.88

4. **mAP50-95** - Harus naik

   - Target: ≥ 0.72

5. **Precision & Recall** - Keduanya harus tinggi
   - Target: ≥ 0.85

### Confusion Matrix

```
           mentah  setengah  matang
mentah     0.89     0.08      0.03
setengah   0.05     0.88      0.07
matang     0.02     0.06      0.92
```

**Interpretasi:**

- Diagonal (bold) = prediksi benar
- Off-diagonal = kesalahan klasifikasi
- Idealnya diagonal ≥ 0.85

---

## 🔥 Troubleshooting

### ❌ Problem: "CUDA out of memory"

**Solusi:**

```python
# Kurangi batch size
BATCH_SIZE = 8  # dari 16
```

### ❌ Problem: "Loss = nan"

**Solusi:**

```python
# Kurangi learning rate
LEARNING_RATE = 0.005  # dari 0.01
```

### ❌ Problem: Akurasi masih rendah (< 75%)

**Solusi:**

1. **Cek kualitas data:**

   - Minimal 200-300 gambar per kelas
   - Anotasi harus presisi
   - Label konsisten

2. **Tingkatkan epochs:**

   ```python
   EPOCHS = 400  # dari 300
   ```

3. **Gunakan model lebih besar:**

   ```python
   MODEL_NAME = "yolo11m.pt"  # dari yolo11s.pt
   ```

4. **Tambah data dengan cara:**
   - Foto lebih banyak sample
   - Berbagai sudut & pencahayaan
   - Augmentasi manual (rotate, flip, dll)

### ❌ Problem: Training stuck/tidak bergerak

**Solusi:**

- Restart runtime: Runtime → Restart runtime
- Cek GPU masih aktif: Runtime → Change runtime type
- Pastikan dataset valid dan tidak corrupt

---

## 💡 Tips Akurasi Maksimal

### 1. **Kualitas Data adalah Kunci**

✅ **GOOD Dataset:**

- 300+ gambar per kelas
- Pencahayaan bervariasi tapi baik
- Fokus clear, tidak blur
- Bounding box presisi (pas dengan objek)
- Label konsisten

❌ **BAD Dataset:**

- < 100 gambar per kelas
- Foto gelap/terlalu terang
- Blur/tidak fokus
- Bounding box asal-asalan
- Label salah/inkonsisten

### 2. **Augmentasi adalah Teman**

Training dengan augmentasi agresif membuat model:

- ✅ Robust terhadap variasi pencahayaan
- ✅ Tidak overfitting
- ✅ Generalisasi lebih baik
- ✅ Akurasi lebih stabil

### 3. **Saturation Augmentation Penting!**

Untuk deteksi kematangan berdasarkan **warna buah**:

```python
hsv_s = 0.9  # SANGAT PENTING!
```

Ini membuat model lebih sensitif terhadap perubahan warna kematangan.

### 4. **Patience & Epochs**

```python
EPOCHS = 300
PATIENCE = 80
```

Jangan terburu-buru! Model butuh waktu untuk belajar dengan baik.

### 5. **Monitor Validation**

Jika `val/mAP50` tidak naik setelah 100 epochs:

- Cek dataset
- Kurangi learning rate
- Tambah regularization

---

## 🎯 Checklist Sebelum Training

- [ ] Dataset minimal 200 gambar per kelas
- [ ] Semua gambar memiliki label (.txt)
- [ ] Jumlah images = jumlah labels
- [ ] Bounding box sudah presisi
- [ ] Upload dataset ke Google Drive
- [ ] GPU T4 aktif di Colab
- [ ] Runtime type: GPU (T4)
- [ ] Path di notebook sudah benar

---

## 📝 Checklist Setelah Training

- [ ] mAP50 ≥ 0.88
- [ ] Precision ≥ 0.85
- [ ] Recall ≥ 0.83
- [ ] Download model `yolo11s_kersen_best.pt`
- [ ] Backup hasil ke Google Drive
- [ ] Copy model ke folder `models/`
- [ ] Update confidence threshold di app.py
- [ ] Test aplikasi Flask
- [ ] Verifikasi akurasi deteksi real-time

---

## 🚀 Next Steps Setelah Deploy

1. **Test dengan berbagai kondisi:**

   - Pencahayaan berbeda
   - Sudut kamera berbeda
   - Jarak berbeda

2. **Monitor performa:**

   - Catat confidence rata-rata
   - Catat false positive/negative
   - Kumpulkan kasus error

3. **Iterasi jika perlu:**
   - Tambah data dari kasus error
   - Re-training dengan data lebih banyak
   - Fine-tune confidence threshold

---

## 📞 Support

Jika ada masalah:

1. Cek troubleshooting section di atas
2. Lihat output error di notebook
3. Pastikan semua cell dijalankan sequential
4. Cek log training untuk anomali

---

## 🎉 Selamat Training!

Dengan konfigurasi ini, model Anda akan memiliki:

- ✅ Akurasi tinggi (81%+ average)
- ✅ Minimum confidence 70%
- ✅ Robust terhadap variasi
- ✅ Fast inference
- ✅ Production-ready

**Good luck! 🍀**
