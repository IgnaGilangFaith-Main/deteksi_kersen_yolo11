"""
Script untuk Menganalisis Hasil Training YOLO
==============================================
Script ini menganalisis hasil training model YOLO yang ditraining di Colab,
menampilkan:
1. Training Loss (box_loss, cls_loss, dfl_loss)
2. Validation Loss (box_loss, cls_loss, dfl_loss)
3. Training vs Validation Accuracy (mAP, Precision, Recall)
4. Confusion Matrix
5. F1-Score

Author: Computer Vision - UMNU
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi path
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "results" / "analysis_output"

# Buat folder output jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nama kelas
CLASS_NAMES = ['mentah', 'setengah_matang', 'matang']


def find_results_csv():
    """
    Mencari file results.csv dari hasil training
    Bisa dari folder results lokal atau upload dari Colab
    """
    possible_paths = [
        RESULTS_DIR / "kersen_v2" / "results.csv",
        RESULTS_DIR / "results.csv",
        BASE_DIR / "runs" / "detect" / "train" / "results.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Jika tidak ditemukan, minta user untuk input path
    print("\n" + "="*60)
    print("FILE results.csv TIDAK DITEMUKAN!")
    print("="*60)
    print("\nSilakan copy file results.csv dari hasil training Colab ke salah satu lokasi:")
    for p in possible_paths:
        print(f"  - {p}")
    
    custom_path = input("\nAtau masukkan path lengkap file results.csv: ").strip()
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    
    return None


def load_training_data(csv_path):
    """
    Load data training dari file CSV
    """
    print(f"\n📂 Memuat data dari: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Berhasil memuat {len(df)} epochs data")
    return df


def analyze_loss(df):
    """
    Analisis Training Loss dan Validation Loss
    """
    print("\n" + "="*60)
    print("📊 ANALISIS LOSS")
    print("="*60)
    
    # Training Loss
    print("\n🔹 Training Loss:")
    train_losses = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']
    for loss in train_losses:
        if loss in df.columns:
            initial = df[loss].iloc[0]
            final = df[loss].iloc[-1]
            min_val = df[loss].min()
            min_epoch = df[loss].idxmin() + 1
            print(f"   {loss}:")
            print(f"      - Initial: {initial:.5f}")
            print(f"      - Final: {final:.5f}")
            print(f"      - Minimum: {min_val:.5f} (Epoch {min_epoch})")
            print(f"      - Penurunan: {((initial - final) / initial * 100):.2f}%")
    
    # Total Training Loss
    if all(l in df.columns for l in train_losses):
        df['train_total_loss'] = df[train_losses].sum(axis=1)
        initial_total = df['train_total_loss'].iloc[0]
        final_total = df['train_total_loss'].iloc[-1]
        print(f"\n   📈 Total Training Loss:")
        print(f"      - Initial: {initial_total:.5f}")
        print(f"      - Final: {final_total:.5f}")
        print(f"      - Penurunan: {((initial_total - final_total) / initial_total * 100):.2f}%")
    
    # Validation Loss
    print("\n🔹 Validation Loss:")
    val_losses = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']
    for loss in val_losses:
        if loss in df.columns:
            initial = df[loss].iloc[0]
            final = df[loss].iloc[-1]
            min_val = df[loss].min()
            min_epoch = df[loss].idxmin() + 1
            print(f"   {loss}:")
            print(f"      - Initial: {initial:.5f}")
            print(f"      - Final: {final:.5f}")
            print(f"      - Minimum: {min_val:.5f} (Epoch {min_epoch})")
            print(f"      - Penurunan: {((initial - final) / initial * 100):.2f}%")
    
    # Total Validation Loss
    if all(l in df.columns for l in val_losses):
        df['val_total_loss'] = df[val_losses].sum(axis=1)
        initial_total = df['val_total_loss'].iloc[0]
        final_total = df['val_total_loss'].iloc[-1]
        print(f"\n   📈 Total Validation Loss:")
        print(f"      - Initial: {initial_total:.5f}")
        print(f"      - Final: {final_total:.5f}")
        print(f"      - Penurunan: {((initial_total - final_total) / initial_total * 100):.2f}%")
    
    return df


def analyze_metrics(df):
    """
    Analisis Metrics: Precision, Recall, mAP
    """
    print("\n" + "="*60)
    print("📊 ANALISIS METRICS (AKURASI)")
    print("="*60)
    
    metrics = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP@50',
        'metrics/mAP50-95(B)': 'mAP@50-95'
    }
    
    results = {}
    
    for col, name in metrics.items():
        if col in df.columns:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            max_val = df[col].max()
            max_epoch = df[col].idxmax() + 1
            
            results[name] = {
                'initial': initial,
                'final': final,
                'max': max_val,
                'max_epoch': max_epoch
            }
            
            print(f"\n🔹 {name}:")
            print(f"   - Initial: {initial:.5f}")
            print(f"   - Final: {final:.5f}")
            print(f"   - Maximum: {max_val:.5f} (Epoch {max_epoch})")
            
            if initial > 0:
                improvement = ((final - initial) / initial * 100)
                print(f"   - Peningkatan: {improvement:+.2f}%")
    
    # Hitung F1-Score dari Precision dan Recall
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        df['f1_score'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / \
                         (df['metrics/precision(B)'] + df['metrics/recall(B)'] + 1e-7)
        
        final_f1 = df['f1_score'].iloc[-1]
        max_f1 = df['f1_score'].max()
        max_f1_epoch = df['f1_score'].idxmax() + 1
        
        results['F1-Score'] = {
            'final': final_f1,
            'max': max_f1,
            'max_epoch': max_f1_epoch
        }
        
        print(f"\n🔹 F1-Score (Calculated):")
        print(f"   - Final: {final_f1:.5f}")
        print(f"   - Maximum: {max_f1:.5f} (Epoch {max_f1_epoch})")
    
    return df, results


def plot_training_loss(df, save_path):
    """
    Plot grafik Training Loss
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # Box Loss
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Train Box Loss', linewidth=2)
    if 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/box_loss'], 'r--', label='Val Box Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Box Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Classification Loss
    ax2 = axes[0, 1]
    if 'train/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train Cls Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val Cls Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Classification Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # DFL Loss
    ax3 = axes[1, 0]
    if 'train/dfl_loss' in df.columns:
        ax3.plot(df['epoch'], df['train/dfl_loss'], 'b-', label='Train DFL Loss', linewidth=2)
    if 'val/dfl_loss' in df.columns:
        ax3.plot(df['epoch'], df['val/dfl_loss'], 'r--', label='Val DFL Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('DFL Loss (Distribution Focal Loss)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Total Loss
    ax4 = axes[1, 1]
    if 'train_total_loss' in df.columns:
        ax4.plot(df['epoch'], df['train_total_loss'], 'b-', label='Train Total Loss', linewidth=2)
    if 'val_total_loss' in df.columns:
        ax4.plot(df['epoch'], df['val_total_loss'], 'r--', label='Val Total Loss', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Total Loss (Box + Cls + DFL)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Grafik Training Loss disimpan: {save_path}")


def plot_metrics(df, save_path):
    """
    Plot grafik Metrics (Accuracy)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Analysis', fontsize=16, fontweight='bold')
    
    # Precision
    ax1 = axes[0, 0]
    if 'metrics/precision(B)' in df.columns:
        ax1.plot(df['epoch'], df['metrics/precision(B)'], 'g-', linewidth=2)
        ax1.axhline(y=df['metrics/precision(B)'].max(), color='r', linestyle='--', alpha=0.5, label=f'Max: {df["metrics/precision(B)"].max():.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Recall
    ax2 = axes[0, 1]
    if 'metrics/recall(B)' in df.columns:
        ax2.plot(df['epoch'], df['metrics/recall(B)'], 'orange', linewidth=2)
        ax2.axhline(y=df['metrics/recall(B)'].max(), color='r', linestyle='--', alpha=0.5, label=f'Max: {df["metrics/recall(B)"].max():.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # mAP50
    ax3 = axes[1, 0]
    if 'metrics/mAP50(B)' in df.columns:
        ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'purple', linewidth=2)
        ax3.axhline(y=df['metrics/mAP50(B)'].max(), color='r', linestyle='--', alpha=0.5, label=f'Max: {df["metrics/mAP50(B)"].max():.4f}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP@50')
    ax3.set_title('mAP@50')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # mAP50-95
    ax4 = axes[1, 1]
    if 'metrics/mAP50-95(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'brown', linewidth=2)
        ax4.axhline(y=df['metrics/mAP50-95(B)'].max(), color='r', linestyle='--', alpha=0.5, label=f'Max: {df["metrics/mAP50-95(B)"].max():.4f}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP@50-95')
    ax4.set_title('mAP@50-95')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik Metrics disimpan: {save_path}")


def plot_f1_score(df, save_path):
    """
    Plot grafik F1-Score
    """
    if 'f1_score' not in df.columns:
        print("⚠️ F1-Score tidak tersedia dalam data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['epoch'], df['f1_score'], 'b-', linewidth=2, label='F1-Score')
    ax.fill_between(df['epoch'], df['f1_score'], alpha=0.3)
    
    max_f1 = df['f1_score'].max()
    max_epoch = df['f1_score'].idxmax() + 1
    ax.axhline(y=max_f1, color='r', linestyle='--', alpha=0.7, label=f'Max: {max_f1:.4f} (Epoch {max_epoch})')
    ax.scatter([max_epoch], [max_f1], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Progress During Training', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik F1-Score disimpan: {save_path}")


def plot_train_vs_val_comparison(df, save_path):
    """
    Plot perbandingan Training vs Validation
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Training vs Validation Loss Comparison', fontsize=16, fontweight='bold')
    
    comparisons = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
        ('train/dfl_loss', 'val/dfl_loss', 'DFL Loss')
    ]
    
    for idx, (train_col, val_col, title) in enumerate(comparisons):
        ax = axes[idx]
        if train_col in df.columns and val_col in df.columns:
            ax.plot(df['epoch'], df[train_col], 'b-', label='Training', linewidth=2)
            ax.plot(df['epoch'], df[val_col], 'r--', label='Validation', linewidth=2)
            
            # Highlight gap (overfitting indicator)
            ax.fill_between(df['epoch'], df[train_col], df[val_col], 
                          where=(df[val_col] > df[train_col]), 
                          alpha=0.3, color='red', label='Overfitting Gap')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Grafik Train vs Val Comparison disimpan: {save_path}")


def display_confusion_matrix(results_folder):
    """
    Menampilkan dan menganalisis Confusion Matrix dari hasil training
    """
    print("\n" + "="*60)
    print("📊 CONFUSION MATRIX")
    print("="*60)
    
    # Cari file confusion matrix
    cm_files = [
        results_folder / "confusion_matrix_normalized.png",
        results_folder / "confusion_matrix.png",
    ]
    
    found_cm = None
    for cm_file in cm_files:
        if cm_file.exists():
            found_cm = cm_file
            break
    
    if found_cm:
        print(f"\n✅ Confusion Matrix ditemukan: {found_cm}")
        print(f"   Silakan buka file untuk melihat detail confusion matrix.")
        
        # Copy confusion matrix ke output folder
        from shutil import copy2
        output_cm = OUTPUT_DIR / found_cm.name
        copy2(found_cm, output_cm)
        print(f"   📂 Disalin ke: {output_cm}")
        
        return found_cm
    else:
        print("\n⚠️ File confusion matrix tidak ditemukan.")
        print("   File ini biasanya dihasilkan oleh YOLO saat training selesai.")
        print("   Pastikan untuk mengcopy file confusion_matrix.png dari hasil training Colab.")
        return None


def display_f1_curve(results_folder):
    """
    Menampilkan F1 Curve dari hasil training
    """
    print("\n" + "="*60)
    print("📊 F1 CURVE")
    print("="*60)
    
    f1_files = [
        results_folder / "BoxF1_curve.png",
        results_folder / "F1_curve.png",
    ]
    
    found_f1 = None
    for f1_file in f1_files:
        if f1_file.exists():
            found_f1 = f1_file
            break
    
    if found_f1:
        print(f"\n✅ F1 Curve ditemukan: {found_f1}")
        
        # Copy ke output folder
        from shutil import copy2
        output_f1 = OUTPUT_DIR / found_f1.name
        copy2(found_f1, output_f1)
        print(f"   📂 Disalin ke: {output_f1}")
        
        return found_f1
    else:
        print("\n⚠️ File F1 curve tidak ditemukan.")
        return None


def generate_summary_report(df, metrics_results, output_path):
    """
    Generate laporan ringkasan hasil training
    """
    print("\n" + "="*60)
    print("📋 MEMBUAT LAPORAN RINGKASAN")
    print("="*60)
    
    report = []
    report.append("=" * 70)
    report.append("LAPORAN ANALISIS HASIL TRAINING YOLO - DETEKSI KERSEN")
    report.append("=" * 70)
    report.append(f"\nTanggal Analisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Epochs: {len(df)}")
    report.append(f"Kelas: {', '.join(CLASS_NAMES)}")
    
    # Training Loss Summary
    report.append("\n" + "-" * 70)
    report.append("1. RINGKASAN TRAINING LOSS")
    report.append("-" * 70)
    
    if 'train_total_loss' in df.columns:
        report.append(f"   Total Training Loss Awal: {df['train_total_loss'].iloc[0]:.5f}")
        report.append(f"   Total Training Loss Akhir: {df['train_total_loss'].iloc[-1]:.5f}")
        report.append(f"   Penurunan: {((df['train_total_loss'].iloc[0] - df['train_total_loss'].iloc[-1]) / df['train_total_loss'].iloc[0] * 100):.2f}%")
    
    # Validation Loss Summary
    report.append("\n" + "-" * 70)
    report.append("2. RINGKASAN VALIDATION LOSS")
    report.append("-" * 70)
    
    if 'val_total_loss' in df.columns:
        report.append(f"   Total Validation Loss Awal: {df['val_total_loss'].iloc[0]:.5f}")
        report.append(f"   Total Validation Loss Akhir: {df['val_total_loss'].iloc[-1]:.5f}")
        report.append(f"   Penurunan: {((df['val_total_loss'].iloc[0] - df['val_total_loss'].iloc[-1]) / df['val_total_loss'].iloc[0] * 100):.2f}%")
    
    # Metrics Summary
    report.append("\n" + "-" * 70)
    report.append("3. RINGKASAN METRICS (PERFORMA MODEL)")
    report.append("-" * 70)
    
    for metric_name, values in metrics_results.items():
        report.append(f"\n   {metric_name}:")
        if 'initial' in values:
            report.append(f"      - Nilai Awal: {values['initial']:.5f}")
        report.append(f"      - Nilai Akhir: {values['final']:.5f}")
        report.append(f"      - Nilai Maksimum: {values['max']:.5f} (Epoch {values['max_epoch']})")
    
    # F1-Score Summary
    report.append("\n" + "-" * 70)
    report.append("4. F1-SCORE")
    report.append("-" * 70)
    
    if 'F1-Score' in metrics_results:
        f1_data = metrics_results['F1-Score']
        report.append(f"   F1-Score Akhir: {f1_data['final']:.5f}")
        report.append(f"   F1-Score Maksimum: {f1_data['max']:.5f} (Epoch {f1_data['max_epoch']})")
    
    # Overfitting Analysis
    report.append("\n" + "-" * 70)
    report.append("5. ANALISIS OVERFITTING")
    report.append("-" * 70)
    
    if 'train_total_loss' in df.columns and 'val_total_loss' in df.columns:
        final_train = df['train_total_loss'].iloc[-1]
        final_val = df['val_total_loss'].iloc[-1]
        gap = final_val - final_train
        
        report.append(f"   Training Loss (Final): {final_train:.5f}")
        report.append(f"   Validation Loss (Final): {final_val:.5f}")
        report.append(f"   Gap (Val - Train): {gap:.5f}")
        
        if gap > 0.5:
            report.append("   ⚠️ Status: KEMUNGKINAN OVERFITTING (gap > 0.5)")
            report.append("   💡 Saran: Coba kurangi epochs atau tambah data augmentation")
        elif gap > 0.2:
            report.append("   ⚠️ Status: SEDIKIT OVERFITTING (gap 0.2 - 0.5)")
            report.append("   💡 Saran: Model cukup baik, pertimbangkan early stopping")
        else:
            report.append("   ✅ Status: MODEL BAIK (gap < 0.2)")
            report.append("   💡 Saran: Training berjalan dengan baik")
    
    # Best Model Epoch
    report.append("\n" + "-" * 70)
    report.append("6. REKOMENDASI")
    report.append("-" * 70)
    
    if 'metrics/mAP50-95(B)' in df.columns:
        best_epoch = df['metrics/mAP50-95(B)'].idxmax() + 1
        best_map = df['metrics/mAP50-95(B)'].max()
        report.append(f"   Best Model berdasarkan mAP50-95: Epoch {best_epoch} (mAP: {best_map:.5f})")
    
    if 'f1_score' in df.columns:
        best_f1_epoch = df['f1_score'].idxmax() + 1
        best_f1 = df['f1_score'].max()
        report.append(f"   Best Model berdasarkan F1-Score: Epoch {best_f1_epoch} (F1: {best_f1:.5f})")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✅ Laporan disimpan: {output_path}")
    
    return report_text


def main():
    """
    Main function untuk menjalankan analisis
    """
    print("\n" + "="*70)
    print("🔬 ANALISIS HASIL TRAINING YOLO - DETEKSI BUAH KERSEN")
    print("="*70)
    print("Script ini akan menganalisis hasil training dari Colab")
    print("="*70)
    
    # 1. Cari dan load data
    csv_path = find_results_csv()
    if csv_path is None:
        print("\n❌ Tidak dapat melanjutkan tanpa file results.csv")
        return
    
    df = load_training_data(csv_path)
    results_folder = csv_path.parent
    
    # 2. Analisis Loss
    df = analyze_loss(df)
    
    # 3. Analisis Metrics dan F1-Score
    df, metrics_results = analyze_metrics(df)
    
    # 4. Generate visualisasi
    print("\n" + "="*60)
    print("📊 MEMBUAT VISUALISASI")
    print("="*60)
    
    plot_training_loss(df, OUTPUT_DIR / "training_loss.png")
    plot_metrics(df, OUTPUT_DIR / "training_metrics.png")
    plot_f1_score(df, OUTPUT_DIR / "f1_score.png")
    plot_train_vs_val_comparison(df, OUTPUT_DIR / "train_vs_val_comparison.png")
    
    # 5. Display Confusion Matrix
    display_confusion_matrix(results_folder)
    
    # 6. Display F1 Curve
    display_f1_curve(results_folder)
    
    # 7. Generate Summary Report
    generate_summary_report(df, metrics_results, OUTPUT_DIR / "laporan_analisis_training.txt")
    
    # 8. Final Summary
    print("\n" + "="*70)
    print("✅ ANALISIS SELESAI!")
    print("="*70)
    print(f"\n📂 Semua hasil analisis disimpan di: {OUTPUT_DIR}")
    print("\nFile yang dihasilkan:")
    for f in OUTPUT_DIR.iterdir():
        print(f"   - {f.name}")
    
    print("\n💡 Tips:")
    print("   - Buka file .png untuk melihat grafik visualisasi")
    print("   - Baca file laporan_analisis_training.txt untuk ringkasan lengkap")
    print("   - Perhatikan gap antara training dan validation loss untuk deteksi overfitting")


if __name__ == "__main__":
    main()
