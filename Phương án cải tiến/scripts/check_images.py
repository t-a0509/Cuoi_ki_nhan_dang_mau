"""
Quét toàn bộ ảnh trong data/dataset_splits.csv, liệt kê ảnh không đọc được vào data/corrupt_images.csv.
src/dataset.py tự loại các ảnh trong danh sách này khi nạp dữ liệu.

Ghi chú: trên máy nhóm có 352 file (Apple Black rot, Apple Cedar rust, Blueberry healthy) toàn byte 0
do giải nén lỗi. Nếu tải/giải nén lại PlantVillage, hãy chạy lại script này để cập nhật danh sách.
"""
import sys
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
from PIL import Image
from tqdm import tqdm

from src.config import SPLITS_CSV, CORRUPT_CSV, DATA_DIR


def main():
    df = pd.read_csv(SPLITS_CSV)
    bad = []
    for r in tqdm(df.itertuples(), total=len(df), desc="Kiểm tra ảnh"):
        path = DATA_DIR / r.rel_path
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as e:
            bad.append({"rel_path": r.rel_path, "class_name": r.class_name, "split": r.split,
                        "error": type(e).__name__})
    out = pd.DataFrame(bad, columns=["rel_path", "class_name", "split", "error"])
    out.to_csv(CORRUPT_CSV, index=False, encoding="utf-8")
    print(f"-> {len(out)} / {len(df):,} ảnh không đọc được. Đã lưu: {CORRUPT_CSV}")
    if len(out):
        print(out.groupby(["class_name", "split"]).size().to_string())


if __name__ == "__main__":
    main()
