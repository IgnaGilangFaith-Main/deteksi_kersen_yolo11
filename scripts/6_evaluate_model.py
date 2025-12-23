"""
Script untuk Mengevaluasi Model YOLO yang Sudah Ditraining
==========================================================
Script ini melakukan evaluasi lengkap pada model YOLO (.pt file),
menampilkan:
1. Confusion Matrix
2. F1-Score per kelas
3. Precision & Recall per kelas
4. mAP (mean Average Precision)
5. Classification Report lengkap

Author: Computer Vision - UMNU
Date: 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics tidak terinstall. Jalankan: pip install ultralytics")
    sys.exit(1)

# Konfigurasi path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_YAML = BASE_DIR / "train_split" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "results" / "model_evaluation"

# Buat folder output jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nama kelas
CLASS_NAMES = ['mentah', 'setengah_matang', 'matang']


def list_available_models():
    """
    List semua model yang tersedia di folder models
    """
    print("\n📂 Model yang tersedia:")
    models = list(MODELS_DIR.glob("*.pt"))
    
    if not models:
        print("   ❌ Tidak ada model ditemukan di folder models/")
        return []
    
    for i, model in enumerate(models, 1):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"   {i}. {model.name} ({size_mb:.2f} MB)")
    
    return models


def select_model():
    """
    Memilih model untuk dievaluasi
    """
    models = list_available_models()
    
    if not models:
        return None
    
    # Default ke yolo11s_kersen_best.pt jika ada
    default_model = MODELS_DIR / "yolo11s_kersen_best.pt"
    if default_model.exists():
        print(f"\n✅ Menggunakan model default: {default_model.name}")
        return default_model
    
    # Jika tidak ada default, gunakan model pertama
    print(f"\n✅ Menggunakan model: {models[0].name}")
    return models[0]


def evaluate_model(model_path, data_yaml=None):
    """
    Menjalankan evaluasi model YOLO
    """
    print("\n" + "="*70)
    print(f"🔬 MENGEVALUASI MODEL: {model_path.name}")
    print("="*70)
    
    # Load model
    print("\n📂 Memuat model...")
    model = YOLO(str(model_path))
    print(f"✅ Model berhasil dimuat")
    
    # Info model
    print(f"\n📊 Informasi Model:")
    print(f"   - Task: {model.task}")
    print(f"   - Nama: {model_path.name}")
    
    # Cek data.yaml
    if data_yaml is None or not Path(data_yaml).exists():
        data_yaml = DATA_YAML
    
    if not Path(data_yaml).exists():
        print(f"\n⚠️ File data.yaml tidak ditemukan: {data_yaml}")
        print("   Mencoba path alternatif...")
        alt_paths = [
            BASE_DIR / "dataset_organized" / "data.yaml",
            BASE_DIR / "data.yaml",
        ]
        for alt in alt_paths:
            if alt.exists():
                data_yaml = alt
                break
    
    print(f"\n📂 Menggunakan data config: {data_yaml}")
    
    # Jalankan validasi
    print("\n🔄 Menjalankan validasi pada test set...")
    results = model.val(
        data=str(data_yaml),
        split='test',  # Gunakan test set
        verbose=False,
        plots=True,    # Generate plots
        save_json=True,
        project=str(OUTPUT_DIR),
        name='evaluation',
        exist_ok=True
    )
    
    return model, results


def analyze_results(results, output_dir):
    """
    Menganalisis hasil evaluasi
    """
    print("\n" + "="*70)
    print("📊 HASIL EVALUASI MODEL")
    print("="*70)
    
    # Metrics utama
    print("\n🔹 METRICS UTAMA:")
    
    # Box metrics
    box_map = results.box.map      # mAP50-95
    box_map50 = results.box.map50  # mAP50
    box_map75 = results.box.map75  # mAP75
    
    print(f"\n   📈 mAP (mean Average Precision):")
    print(f"      - mAP@50:    {box_map50:.4f} ({box_map50*100:.2f}%)")
    print(f"      - mAP@75:    {box_map75:.4f} ({box_map75*100:.2f}%)")
    print(f"      - mAP@50-95: {box_map:.4f} ({box_map*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n   📊 Metrics Per Kelas:")
    
    # Get per-class data from results
    class_names = results.names
    num_classes = len(CLASS_NAMES)
    
    # Access per-class metrics - handle different formats
    # Try to get per-class data from results.box.class_result or similar
    try:
        # Get class-wise results from the results object
        p_per_class = np.array(results.box.p)  # Precision per class
        r_per_class = np.array(results.box.r)  # Recall per class
        ap50_per_class = np.array(results.box.ap50)  # AP50 per class
        ap_per_class = np.array(results.box.ap)  # AP50-95 per class
        
        # Ensure they're arrays
        if p_per_class.ndim == 0:
            p_per_class = np.array([p_per_class])
        if r_per_class.ndim == 0:
            r_per_class = np.array([r_per_class])
        if ap50_per_class.ndim == 0:
            ap50_per_class = np.array([ap50_per_class])
        if ap_per_class.ndim == 0:
            ap_per_class = np.array([ap_per_class])
            
    except Exception as e:
        print(f"   ⚠️ Tidak dapat mengakses metrics per kelas: {e}")
        # Fallback: gunakan mean values untuk semua kelas
        mp = float(results.box.mp) if hasattr(results.box, 'mp') else box_map50
        mr = float(results.box.mr) if hasattr(results.box, 'mr') else box_map50
        p_per_class = np.array([mp] * num_classes)
        r_per_class = np.array([mr] * num_classes)
        ap50_per_class = np.array([box_map50] * num_classes)
        ap_per_class = np.array([box_map] * num_classes)
    
    # Calculate F1 scores
    f1_per_class = 2 * (p_per_class * r_per_class) / (p_per_class + r_per_class + 1e-7)
    
    print("\n   " + "-"*65)
    print(f"   {'Kelas':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'mAP50':<12}")
    print("   " + "-"*65)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(p_per_class):
            print(f"   {class_name:<20} {p_per_class[i]:.4f}       {r_per_class[i]:.4f}       {f1_per_class[i]:.4f}       {ap50_per_class[i]:.4f}")
    
    print("   " + "-"*65)
    print(f"   {'RATA-RATA':<20} {p_per_class.mean():.4f}       {r_per_class.mean():.4f}       {f1_per_class.mean():.4f}       {ap50_per_class.mean():.4f}")
    print("   " + "-"*65)
    
    return {
        'mAP50': box_map50,
        'mAP75': box_map75,
        'mAP50-95': box_map,
        'precision_per_class': p_per_class,
        'recall_per_class': r_per_class,
        'f1_per_class': f1_per_class,
        'ap50_per_class': ap50_per_class,
        'mean_precision': float(p_per_class.mean()),
        'mean_recall': float(r_per_class.mean()),
        'mean_f1': float(f1_per_class.mean())
    }


def plot_metrics_bar(metrics, save_path):
    """
    Plot bar chart untuk metrics per kelas
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation Metrics Per Class', fontsize=16, fontweight='bold')
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.6
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Precision
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, metrics['precision_per_class'], width, color=colors)
    ax1.axhline(y=metrics['mean_precision'], color='red', linestyle='--', 
                label=f'Mean: {metrics["mean_precision"]:.4f}')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision per Kelas')
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for bar, val in zip(bars1, metrics['precision_per_class']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Recall
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, metrics['recall_per_class'], width, color=colors)
    ax2.axhline(y=metrics['mean_recall'], color='red', linestyle='--', 
                label=f'Mean: {metrics["mean_recall"]:.4f}')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall per Kelas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars2, metrics['recall_per_class']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1-Score
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, metrics['f1_per_class'], width, color=colors)
    ax3.axhline(y=metrics['mean_f1'], color='red', linestyle='--', 
                label=f'Mean: {metrics["mean_f1"]:.4f}')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score per Kelas')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASS_NAMES)
    ax3.set_ylim([0, 1.1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars3, metrics['f1_per_class']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # mAP50 per class
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, metrics['ap50_per_class'], width, color=colors)
    ax4.axhline(y=metrics['mAP50'], color='red', linestyle='--', 
                label=f'Mean: {metrics["mAP50"]:.4f}')
    ax4.set_ylabel('mAP@50')
    ax4.set_title('mAP@50 per Kelas')
    ax4.set_xticks(x)
    ax4.set_xticklabels(CLASS_NAMES)
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    for bar, val in zip(bars4, metrics['ap50_per_class']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Grafik metrics per kelas disimpan: {save_path}")


def plot_overall_metrics(metrics, save_path):
    """
    Plot overview metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overall_metrics = {
        'mAP@50': metrics['mAP50'],
        'mAP@75': metrics['mAP75'],
        'mAP@50-95': metrics['mAP50-95'],
        'Mean Precision': metrics['mean_precision'],
        'Mean Recall': metrics['mean_recall'],
        'Mean F1-Score': metrics['mean_f1']
    }
    
    x = np.arange(len(overall_metrics))
    values = list(overall_metrics.values())
    labels = list(overall_metrics.keys())
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
    bars = ax.barh(x, values, color=colors)
    
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlim([0, 1.1])
    ax.set_xlabel('Score')
    ax.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik overall metrics disimpan: {save_path}")


def generate_evaluation_report(model_name, metrics, output_path):
    """
    Generate laporan evaluasi lengkap
    """
    print("\n" + "="*70)
    print("📋 MEMBUAT LAPORAN EVALUASI")
    print("="*70)
    
    import pandas as pd
    
    report = []
    report.append("=" * 70)
    report.append("LAPORAN EVALUASI MODEL YOLO - DETEKSI BUAH KERSEN")
    report.append("=" * 70)
    report.append(f"\nTanggal Evaluasi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: {model_name}")
    report.append(f"Kelas: {', '.join(CLASS_NAMES)}")
    
    # Overall Metrics
    report.append("\n" + "-" * 70)
    report.append("1. METRICS KESELURUHAN")
    report.append("-" * 70)
    report.append(f"   mAP@50:       {metrics['mAP50']:.5f} ({metrics['mAP50']*100:.2f}%)")
    report.append(f"   mAP@75:       {metrics['mAP75']:.5f} ({metrics['mAP75']*100:.2f}%)")
    report.append(f"   mAP@50-95:    {metrics['mAP50-95']:.5f} ({metrics['mAP50-95']*100:.2f}%)")
    report.append(f"   Mean Precision: {metrics['mean_precision']:.5f}")
    report.append(f"   Mean Recall:    {metrics['mean_recall']:.5f}")
    report.append(f"   Mean F1-Score:  {metrics['mean_f1']:.5f}")
    
    # Per-class Metrics
    report.append("\n" + "-" * 70)
    report.append("2. METRICS PER KELAS")
    report.append("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(metrics['precision_per_class']):
            report.append(f"\n   📌 {class_name.upper()}:")
            report.append(f"      - Precision: {metrics['precision_per_class'][i]:.5f}")
            report.append(f"      - Recall:    {metrics['recall_per_class'][i]:.5f}")
            report.append(f"      - F1-Score:  {metrics['f1_per_class'][i]:.5f}")
            report.append(f"      - AP@50:     {metrics['ap50_per_class'][i]:.5f}")
    
    # F1-Score Summary
    report.append("\n" + "-" * 70)
    report.append("3. RINGKASAN F1-SCORE")
    report.append("-" * 70)
    
    best_class_idx = np.argmax(metrics['f1_per_class'])
    worst_class_idx = np.argmin(metrics['f1_per_class'])
    
    report.append(f"   F1-Score Rata-rata: {metrics['mean_f1']:.5f}")
    report.append(f"   Kelas Terbaik:     {CLASS_NAMES[best_class_idx]} (F1: {metrics['f1_per_class'][best_class_idx]:.5f})")
    report.append(f"   Kelas Terburuk:    {CLASS_NAMES[worst_class_idx]} (F1: {metrics['f1_per_class'][worst_class_idx]:.5f})")
    
    # Performance Assessment
    report.append("\n" + "-" * 70)
    report.append("4. PENILAIAN PERFORMA MODEL")
    report.append("-" * 70)
    
    mean_f1 = metrics['mean_f1']
    if mean_f1 >= 0.9:
        grade = "SANGAT BAIK (A)"
        status = "✅"
    elif mean_f1 >= 0.8:
        grade = "BAIK (B)"
        status = "✅"
    elif mean_f1 >= 0.7:
        grade = "CUKUP BAIK (C)"
        status = "⚠️"
    elif mean_f1 >= 0.6:
        grade = "KURANG (D)"
        status = "⚠️"
    else:
        grade = "PERLU PERBAIKAN (E)"
        status = "❌"
    
    report.append(f"   {status} Grade: {grade}")
    report.append(f"   📊 Mean F1-Score: {mean_f1:.5f}")
    
    # Recommendations
    report.append("\n" + "-" * 70)
    report.append("5. REKOMENDASI")
    report.append("-" * 70)
    
    if mean_f1 >= 0.85:
        report.append("   ✅ Model sudah sangat baik untuk deployment")
        report.append("   💡 Pertimbangkan untuk melakukan pruning/quantization untuk optimasi")
    elif mean_f1 >= 0.7:
        report.append("   ⚠️ Model cukup baik, namun masih bisa ditingkatkan")
        report.append("   💡 Saran: Tambah data training atau gunakan augmentation lebih banyak")
    else:
        report.append("   ❌ Model perlu perbaikan signifikan")
        report.append("   💡 Saran:")
        report.append("      - Periksa kualitas anotasi dataset")
        report.append("      - Tambah lebih banyak data training")
        report.append("      - Coba hyperparameter tuning")
        report.append("      - Gunakan pretrained model yang lebih besar")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Laporan evaluasi disimpan: {output_path}")
    
    return report_text


def copy_yolo_plots(eval_dir):
    """
    Copy plot yang dihasilkan YOLO ke output folder
    """
    from shutil import copy2
    
    plots_to_copy = [
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'F1_curve.png',
        'P_curve.png',
        'R_curve.png',
        'PR_curve.png',
    ]
    
    eval_path = OUTPUT_DIR / "evaluation"
    
    print("\n📂 Menyalin plot dari YOLO...")
    for plot in plots_to_copy:
        src = eval_path / plot
        if src.exists():
            dst = OUTPUT_DIR / plot
            copy2(src, dst)
            print(f"   ✅ {plot}")


def main():
    """
    Main function untuk menjalankan evaluasi
    """
    print("\n" + "="*70)
    print("🔬 EVALUASI MODEL YOLO - DETEKSI BUAH KERSEN")
    print("="*70)
    print("Script ini akan mengevaluasi model YOLO pada test dataset")
    print("="*70)
    
    # 1. Pilih model
    model_path = select_model()
    if model_path is None:
        print("\n❌ Tidak ada model yang tersedia untuk dievaluasi")
        return
    
    # 2. Jalankan evaluasi
    model, results = evaluate_model(model_path)
    
    # 3. Analisis hasil
    metrics = analyze_results(results, OUTPUT_DIR)
    
    # 4. Generate visualisasi
    print("\n" + "="*60)
    print("📊 MEMBUAT VISUALISASI")
    print("="*60)
    
    plot_metrics_bar(metrics, OUTPUT_DIR / "metrics_per_class.png")
    plot_overall_metrics(metrics, OUTPUT_DIR / "overall_metrics.png")
    
    # 5. Copy YOLO plots
    copy_yolo_plots(OUTPUT_DIR)
    
    # 6. Generate report
    generate_evaluation_report(model_path.name, metrics, OUTPUT_DIR / "laporan_evaluasi_model.txt")
    
    # 7. Final Summary
    print("\n" + "="*70)
    print("✅ EVALUASI SELESAI!")
    print("="*70)
    print(f"\n📂 Semua hasil evaluasi disimpan di: {OUTPUT_DIR}")
    print("\nFile yang dihasilkan:")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            print(f"   - {f.name}")
    
    print("\n💡 Tips:")
    print("   - Buka confusion_matrix.png untuk melihat kesalahan klasifikasi")
    print("   - Buka metrics_per_class.png untuk perbandingan performa antar kelas")
    print("   - Baca laporan_evaluasi_model.txt untuk ringkasan lengkap")


if __name__ == "__main__":
    main()
