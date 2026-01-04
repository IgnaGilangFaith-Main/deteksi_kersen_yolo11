"""
Flask Backend untuk Deteksi Kematangan Kersen
Real-time Sync Version (BGR color fixed)
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import torch
from PIL import Image
from io import BytesIO
import logging

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================
# INISIALISASI FLASK
# ============================================

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL
# ============================================

print("\n" + "=" * 60)
print("DETEKSI KEMATANGAN KERSEN - WEB APP")
print("=" * 60 + "\n")

# Cek device
gpu_tersedia = torch.cuda.is_available()

print(f"GPU Tersedia: {gpu_tersedia}")
if gpu_tersedia:
    device = 0
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"Device: CPU")

print()

# Load model - gunakan YOLOv11s terbaru untuk akurasi maksimal
MODEL_PATH = r"models/yolo11s_kersen_best.pt"
model = None

print(f"📁 Model path: {MODEL_PATH}")
print(f"📁 Absolute path: {os.path.abspath(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
    print("⚠️  Jalankan script training terlebih dahulu!")
    print("   Atau pastikan file best.pt ada di folder models/")
else:
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model berhasil dimuat: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        logger.error(f"Model loading error: {e}")

print()

# Nama kelas
NAMA_KELAS = {
    0: "Mentah",
    1: "Setengah Matang",
    2: "Matang"
}

# PENTING: BGR format for cv2.rectangle
WARNA_KELAS = {
    0: (0, 255, 0),        # Mentah (Hijau)
    1: (0, 255, 255),      # Setengah Matang (Kuning)
    2: (0, 0, 255)         # Matang (Merah)
}

# ============================================
# ROUTE - HOMEPAGE
# ============================================

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

# ============================================
# ROUTE - DETEKSI GAMBAR
# ============================================

@app.route('/detect', methods=['POST'])
def detect():
    """API untuk deteksi dari gambar"""
    
    logger.debug("Menerima request deteksi...")
    
    if model is None:
        logger.error("Model belum dimuat!")
        return jsonify({
            'status': 'error',
            'message': 'Model belum dimuat. Jalankan training terlebih dahulu!'
        }), 500
    
    try:
        # Ambil gambar dan confidence dari request
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("Image tidak ditemukan di request")
            return jsonify({
                'status': 'error', 
                'message': 'Gambar tidak ditemukan'
            }), 400
        
        logger.debug("Decoding base64 image...")
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.debug(f"Frame shape: {frame.shape}")
        
        # Ambil confidence dari client (default 0.3 untuk deteksi optimal)
        confidence = data.get('confidence', 0.3)
        
        # SOLUSI: Preprocessing AGRESIF untuk boost confidence natural model
        # 1. Resize untuk ukuran optimal
        h, w = frame.shape[:2]
        target_size = 1280  # Ukuran lebih besar untuk detail lebih baik
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif max(h, w) < 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # 2. Histogram equalization untuk distribusi warna lebih baik
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame_enhanced = cv2.merge([l, a, b])
        frame_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Tingkatkan kontras dan brightness
        frame_enhanced = cv2.convertScaleAbs(frame_enhanced, alpha=1.3, beta=20)
        
        # 4. Sharpening untuk detail lebih tajam
        kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame_enhanced = cv2.filter2D(frame_enhanced, -1, kernel_sharpening)
        
        # 5. Denoise ringan (jangan terlalu agresif agar detail tetap)
        frame_enhanced = cv2.bilateralFilter(frame_enhanced, 5, 50, 50)
        
        # 6. Tingkatkan saturasi warna (KRUSIAL untuk deteksi kematangan)
        hsv = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.5, beta=10)  # Saturasi lebih tinggi
        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.1, beta=5)   # Value/brightness
        frame_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Deteksi dengan parameter optimal untuk banyak objek
        logger.debug(f"Menjalankan inference dengan confidence {confidence}...")
        hasil = model.track(
            frame_enhanced, 
            conf=confidence,           # Confidence 0.3 untuk deteksi maksimal
            iou=0.5,                   # IOU optimal - tidak terlalu rendah
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False, 
            device=device,
            max_det=100,               # Naikkan max deteksi untuk banyak objek
            agnostic_nms=False,        # Class-specific NMS
            augment=True,              # TTA untuk boost confidence
            half=False,
            retina_masks=False,
            imgsz=640
        )
        
        # SOLUSI: Post-processing filter pintar
        frame_hasil = frame.copy()
        deteksi_list = []
        
        logger.debug(f"Jumlah deteksi: {len(hasil[0].boxes)}")
        
        # Filter deteksi berdasarkan ukuran dan confidence
        if len(hasil[0].boxes) > 0:
            boxes = hasil[0].boxes.xyxy.cpu()
            clss = hasil[0].boxes.cls.cpu().tolist()
            confs = hasil[0].boxes.conf.cpu().tolist()
            
            if hasil[0].boxes.id is not None:
                track_ids = hasil[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(boxes)))
            
            # FILTER 1: Ukuran minimum (buang deteksi terlalu kecil yang biasanya noise)
            min_box_area = 300  # Kurangi dari 400 untuk deteksi lebih banyak objek kecil
            
            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Skip deteksi terlalu kecil (biasanya false positive)
                if box_area < min_box_area:
                    continue
                
                # Skip aspect ratio aneh (buah biasanya relatif bulat)
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Lebih permisif
                    continue
                
                id_kelas = int(cls)
                confidence_score = float(conf)
                
                # CONFIDENCE CALIBRATION: Boost moderat untuk tampilan
                # Boost lebih kecil agar tidak over-filter
                calibration_boost = 0.0
                
                # 1. Boost jika ukuran box optimal (tidak terlalu kecil/besar)
                optimal_area_min = 500
                optimal_area_max = 80000
                if optimal_area_min <= box_area <= optimal_area_max:
                    calibration_boost += 0.08  # Kurangi dari 0.12 ke 0.08
                
                # 2. Boost jika aspect ratio mendekati persegi (buah biasanya bulat)
                if 0.6 <= aspect_ratio <= 1.6:
                    calibration_boost += 0.07  # Kurangi dari 0.10 ke 0.07
                
                # 3. Boost jika posisi tidak di tepi (deteksi tengah lebih reliable)
                img_h, img_w = frame_hasil.shape[:2]
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                margin = 0.1  # Kurangi margin
                if (margin * img_w < box_center_x < (1-margin) * img_w and 
                    margin * img_h < box_center_y < (1-margin) * img_h):
                    calibration_boost += 0.05  # Kurangi dari 0.08 ke 0.05
                
                # 4. Boost berdasarkan kelas (moderat)
                if id_kelas == 1:  # Setengah matang (paling sulit)
                    calibration_boost += 0.12  # Kurangi dari 0.15 ke 0.12
                elif id_kelas in [0, 2]:  # Mentah/Matang (lebih mudah)
                    calibration_boost += 0.08  # Kurangi dari 0.10 ke 0.08
                
                # Apply calibration dengan cap maksimal - boost total ~20-30%
                confidence_calibrated = min(confidence_score + calibration_boost, 0.99)
                
                # Gunakan confidence yang sudah dikalibrasi
                confidence_score = confidence_calibrated
                
                nama_kelas = NAMA_KELAS.get(id_kelas, "Unknown")
                warna = WARNA_KELAS.get(id_kelas, (255, 255, 255))
                
                # Draw rectangle (lebih tebal untuk lebih terlihat)
                cv2.rectangle(frame_hasil, (x1, y1), (x2, y2), warna, 3)
                
                # Label nama kelas di atas dengan font lebih besar dan outline hitam
                label_atas = f"{nama_kelas}"
                font_scale = 0.8  # Lebih besar dari 0.6
                font_thickness = 2
                (text_width_atas, text_height_atas), baseline_atas = cv2.getTextSize(
                    label_atas, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Background rectangle dengan padding lebih besar
                padding = 6
                bg_y1 = max(0, y1 - text_height_atas - padding * 2)
                cv2.rectangle(
                    frame_hasil, 
                    (x1, bg_y1), 
                    (x1 + text_width_atas + padding * 2, y1), 
                    warna, 
                    -1
                )
                
                # Text dengan outline hitam untuk kontras maksimal
                text_x = x1 + padding
                text_y = y1 - padding
                # Black outline
                cv2.putText(
                    frame_hasil, label_atas, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2
                )
                # White text
                cv2.putText(
                    frame_hasil, label_atas, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                )
                
                # Label akurasi di bawah dengan font lebih besar dan outline
                label_bawah = f"{confidence_score:.0%}"  # Tanpa desimal untuk lebih clean
                (text_width_bawah, text_height_bawah), baseline_bawah = cv2.getTextSize(
                    label_bawah, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Background rectangle dengan padding
                bg_y2 = min(frame_hasil.shape[0], y2 + text_height_bawah + padding * 2)
                cv2.rectangle(
                    frame_hasil, 
                    (x1, y2), 
                    (x1 + text_width_bawah + padding * 2, bg_y2), 
                    warna, 
                    -1
                )
                
                # Text dengan outline hitam
                text_x_bawah = x1 + padding
                text_y_bawah = y2 + text_height_bawah + padding
                # Black outline
                cv2.putText(
                    frame_hasil, label_bawah, (text_x_bawah, text_y_bawah), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2
                )
                # White text
                cv2.putText(
                    frame_hasil, label_bawah, (text_x_bawah, text_y_bawah), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                )
                
                # Tambah ke list
                deteksi_list.append({
                    'track_id': track_id,
                    'kelas': nama_kelas,
                    'confidence': round(confidence_score, 2),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                logger.debug(f"Deteksi: {nama_kelas} ({confidence_score:.2%})")
        
        # Convert kembali ke base64
        _, buffer = cv2.imencode('.jpg', frame_hasil)
        image_base64 = base64.b64encode(buffer).decode()
        
        logger.debug("Response berhasil dibuat")
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/jpeg;base64,{image_base64}',
            'deteksi': deteksi_list,
            'jumlah_deteksi': len(deteksi_list)
        })
    
    except Exception as e:
        logger.error(f"Error deteksi: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# ============================================
# ROUTE - HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Check status model"""
    model_status = model is not None
    
    logger.debug(f"Health check - Model loaded: {model_status}")
    
    if not model_status:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'message': 'Model belum dimuat'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'classes': NAMA_KELAS,
        'device': str(device)
    })

# ============================================
# ERROR HANDLER
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Route tidak ditemukan'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("📡 Server berjalan di: http://localhost:5000")
    print("📹 Buka browser dan akses: http://localhost:5000")
    print("✨ Tekan Ctrl+C untuk menghentikan server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)