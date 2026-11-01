import os
import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import SPLITS_CSV, RESULTS_DIR, CLASS_INFO
from src.dataset import load_split_df

def generate_eda_reports():
    print("=" * 65)
    print("📈 TẠO BÁO CÁO KHÁM PHÁ DỮ LIỆU (EDA) CHO BÁO CÁO THẠC SĨ")
    print("=" * 65)
    
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Chưa tìm thấy {SPLITS_CSV}. Hãy chạy src/split_data.py trước.")

    df = load_split_df()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Bảng thống kê tóm tắt theo lớp
    summary = []
    classes = sorted(df["class_name"].unique())
    
    for cls in classes:
        cls_df = df[df["class_name"] == cls]
        train_count = len(cls_df[cls_df["split"] == "train"])
        val_count = len(cls_df[cls_df["split"] == "val"])
        test_count = len(cls_df[cls_df["split"] == "test"])
        total = len(cls_df)
        
        info = CLASS_INFO.get(cls, {"vi": cls, "status": "unknown"})
        
        summary.append({
            "Tên tiếng Anh (Kaggle)": cls,
            "Tên tiếng Việt": info["vi"],
            "Trạng thái": "Khỏe mạnh" if info["status"] == "healthy" else "Bị bệnh",
            "Tổng số ảnh": total,
            "Tập Train (80%)": train_count,
            "Tập Val (10%)": val_count,
            "Tập Test (10%)": test_count,
            "Tỷ lệ %": round((total / len(df)) * 100, 2)
        })
        
    sum_df = pd.DataFrame(summary)
    csv_out = RESULTS_DIR / "eda_summary_table.csv"
    sum_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu bảng thống kê phân bố: {csv_out}")
    
    # 2. Vẽ biểu đồ phân bố 38 lớp
    plt.figure(figsize=(18, 10))
    short_names = [c.replace("___", "\n").replace("_", " ") for c in classes]
    
    counts_train = [len(df[(df["class_name"] == c) & (df["split"] == "train")]) for c in classes]
    counts_val = [len(df[(df["class_name"] == c) & (df["split"] == "val")]) for c in classes]
    counts_test = [len(df[(df["class_name"] == c) & (df["split"] == "test")]) for c in classes]
    
    x = range(len(classes))
    plt.bar(x, counts_train, label="Train (80%)", color="#10b981", alpha=0.9)
    plt.bar(x, counts_val, bottom=counts_train, label="Val (10%)", color="#f59e0b", alpha=0.9)
    plt.bar(x, counts_test, bottom=[i+j for i,j in zip(counts_train, counts_val)], label="Test (10%)", color="#3b82f6", alpha=0.9)
    
    plt.title("Phân Bố Số Lượng Mẫu Trong 38 Lớp (PlantVillage Dataset) Theo Tập Dữ Liệu", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Lớp Cây Trồng & Bệnh", fontsize=12, fontweight="bold")
    plt.ylabel("Số lượng ảnh", fontsize=12, fontweight="bold")
    plt.xticks(x, short_names, rotation=90, fontsize=8)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    chart_out = RESULTS_DIR / "eda_distribution.png"
    plt.savefig(chart_out, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố: {chart_out}")
    
    # 3. Tạo Grid ảnh mẫu các bệnh tiêu biểu
    print("-> Đang trích xuất ảnh mẫu để tạo grid trực quan...")
    sample_classes = [
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy",
        "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Pepper,_bell___Bacterial_spot"
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, cls in enumerate(sample_classes):
        cls_df = df[df["class_name"] == cls]
        if not cls_df.empty:
            img_path = cls_df.iloc[0]["filepath"]
            img = Image.open(img_path)
            axes[idx].imshow(img)
            info = CLASS_INFO.get(cls, {"vi": cls})
            axes[idx].set_title(info["vi"], fontsize=10, fontweight="bold", pad=6)
        axes[idx].axis("off")
        
    plt.suptitle("Một Số Mẫu Ảnh Đại Diện Cho Các Bệnh Điển Hình (PlantVillage)", fontsize=16, fontweight="bold", y=0.95)
    plt.tight_layout()
    
    grid_out = RESULTS_DIR / "eda_sample_grid.png"
    plt.savefig(grid_out, dpi=300)
    plt.close()
    print(f"-> Đã lưu grid ảnh mẫu: {grid_out}")
    print("=" * 65)
    print("✅ EDA HOÀN TẤT!")

if __name__ == "__main__":
    generate_eda_reports()
