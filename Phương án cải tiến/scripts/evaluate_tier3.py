import os
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from src.config import RESULTS_DIR, MODEL_PATH, CLASS_MAPPING_PATH, NUM_CLASSES, USE_AMP
from src.dataset import get_data_loaders
from src.model import build_resnet50

def plot_training_curves(history_file, save_path):
    """Vẽ biểu đồ Loss và Accuracy qua các Epoch."""
    if not history_file.exists():
        return
        
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    plt.title("Đường cong Mất mát (CrossEntropy Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
    plt.plot(epochs, history["val_acc"], "g--^", label="Val Acc", linewidth=2)
    plt.title("Đường cong Độ chính xác (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ huấn luyện: {save_path}")

def plot_confusion_matrix(cm, class_names, save_path):
    """Vẽ ma trận nhầm lẫn 38 lớp đẹp mắt."""
    plt.figure(figsize=(20, 18))
    # Rút gọn tên lớp cho dễ nhìn trên trục tọa độ
    short_names = [c.replace("___", "\n").replace("_", " ") for c in class_names]
    
    sns.heatmap(
        cm,
        annot=False,
        cmap="YlGnBu",
        xticklabels=short_names,
        yticklabels=short_names,
        cbar_kws={"label": "Số lượng mẫu"}
    )
    plt.title("Ma trận Nhầm lẫn (Confusion Matrix) - Tầng 3: Fine-tuned ResNet50 (38 Lớp)", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Nhãn Dự đoán (Predicted Label)", fontsize=13, fontweight="bold")
    plt.ylabel("Nhãn Thực tế (Ground Truth)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu Ma trận nhầm lẫn: {save_path}")

def main():
    print("=" * 65)
    print("📊 ĐÁNH GIÁ TẦNG 3 (RESNET50 FINE-TUNED) TRÊN TẬP TEST ĐỘC LẬP")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_PATH
    mapping_path = CLASS_MAPPING_PATH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}. Hãy chạy scripts/train_tier3.py trước.")
        
    # Load class mapping
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Load model
    print("-> Đang nạp mô hình...")
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Load test set
    print("-> Đang nạp tập Test...")
    _, _, test_loader = get_data_loaders(batch_size=32)
    print(f"   * Tổng số mẫu Test: {len(test_loader.dataset):,} ảnh")
    
    all_preds = []
    all_labels = []
    all_probs = []

    start_time = time.time()

    print("-> Đang chạy suy luận trên tập Test...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference Test"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=USE_AMP and device.type == "cuda"):
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
            
    total_infer_time = time.time() - start_time
    avg_latency_ms = (total_infer_time / len(all_labels)) * 1000
    
    # Tính toán các chỉ số
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted") * 100
    
    print("\n" + "-" * 65)
    print(f"🎯 KẾT QUẢ TRÊN TẬP TEST ĐỘC LẬP:")
    print(f"   * Accuracy:              {acc:.2f}%")
    print(f"   * Macro F1-Score:        {macro_f1:.2f}%")
    print(f"   * Weighted F1-Score:     {weighted_f1:.2f}%")
    print(f"   * Tốc độ suy luận (GPU): {avg_latency_ms:.2f} ms/ảnh ({len(all_labels)/total_infer_time:.1f} ảnh/giây)")
    print("-" * 65)
    
    # Xuất Classification Report chi tiết
    report_str = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_path = RESULTS_DIR / "tier3_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== BÁO CÁO PHÂN LOẠI TẦNG 3: RESNET50 FINE-TUNING ===\n")
        f.write(f"Accuracy: {acc:.2f}%\nMacro F1: {macro_f1:.2f}%\nWeighted F1: {weighted_f1:.2f}%\n")
        f.write(f"Độ trễ trung bình: {avg_latency_ms:.2f} ms/ảnh\n\n")
        f.write(report_str)
    print(f"-> Đã lưu Báo cáo phân loại: {report_path}")
    
    # Lưu metrics JSON
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_latency_ms": avg_latency_ms,
        "total_test_samples": len(all_labels)
    }
    with open(RESULTS_DIR / "tier3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Lưu dự đoán từng ảnh + ma trận xác suất để phân tích lỗi (scripts/analyze_errors_tier3.py)
    # test_loader không shuffle nên thứ tự trùng với test_loader.dataset.df
    all_probs = np.concatenate(all_probs, axis=0)
    np.save(RESULTS_DIR / "tier3_test_probs.npy", all_probs)
    pred_df = test_loader.dataset.df[["rel_path", "class_name", "label_idx"]].copy()
    pred_df["pred_idx"] = all_preds
    pred_df["pred_class"] = [class_names[i] for i in all_preds]
    pred_df["confidence"] = all_probs.max(axis=1)
    pred_df["correct"] = pred_df["pred_idx"] == pred_df["label_idx"]
    pred_df.to_csv(RESULTS_DIR / "tier3_test_predictions.csv", index=False, encoding="utf-8")
    print(f"-> Đã lưu dự đoán từng ảnh: {RESULTS_DIR / 'tier3_test_predictions.csv'}")

    # Vẽ Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / "tier3_confusion_matrix.png")
    
    # Vẽ Training Curves
    plot_training_curves(RESULTS_DIR / "tier3_training_history.json", RESULTS_DIR / "tier3_training_curves.png")
    
    print("\n" + "=" * 65)
    print("✅ ĐÁNH GIÁ VÀ XUẤT BÁO CÁO HOÀN TẤT!")
    print("=" * 65)

if __name__ == "__main__":
    main()
