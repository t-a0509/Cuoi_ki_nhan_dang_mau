import os
import sys
import json
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, DATA_DIR, SPLITS_CSV, MODELS_DIR

def create_dataset_splits(val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Quét toàn bộ ảnh từ RAW_DATA_DIR và phân chia thành train (80%), val (10%), test (10%)
    theo phương pháp phân tầng (Stratified Split) bảo toàn tỉ lệ giữa các lớp.
    Lưu kết quả ra file data/dataset_splits.csv mà không cần copy nhân đôi file ảnh.
    """
    print(f"[1/3] Đang quét dữ liệu từ: {RAW_DATA_DIR}")
    
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {RAW_DATA_DIR}")
    
    class_names = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    print(f"-> Tìm thấy {len(class_names)} lớp cây trồng.")
    
    # Tạo mapping class_name <-> index
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}
    
    # Lưu class mapping vào thư mục models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = MODELS_DIR / "tier3_class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu bảng ánh xạ lớp vào: {mapping_path}")
    
    # Thu thập tất cả đường dẫn ảnh
    records = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for cls in class_names:
        cls_dir = RAW_DATA_DIR / cls
        # sorted() để thứ tự file giống nhau trên mọi hệ điều hành -> cách chia tái lập được
        for file in sorted(cls_dir.iterdir()):
            if file.suffix.lower() in valid_exts:
                records.append({
                    # Đường dẫn tương đối so với thư mục data/ để chạy được trên máy khác
                    "rel_path": file.relative_to(DATA_DIR).as_posix(),
                    "class_name": cls,
                    "label_idx": class_to_idx[cls]
                })
                
    df = pd.DataFrame(records)
    print(f"[2/3] Tổng số ảnh hợp lệ: {len(df):,} ảnh")
    
    # Stratified split: Train (80%) vs (Val + Test = 20%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label_idx"],
        random_state=random_seed
    )
    
    # Tỉ lệ chia đều temp_df (20%) thành val (10%) và test (10%)
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label_idx"],
        random_state=random_seed
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df.to_csv(SPLITS_CSV, index=False, encoding="utf-8")
    
    print(f"[3/3] Phân chia hoàn tất và lưu vào: {SPLITS_CSV}")
    print(f"  * Train set: {len(train_df):,} ảnh ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  * Val set:   {len(val_df):,} ảnh ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  * Test set:  {len(test_df):,} ảnh ({len(test_df)/len(df)*100:.1f}%)")
    
    return final_df

if __name__ == "__main__":
    create_dataset_splits()
