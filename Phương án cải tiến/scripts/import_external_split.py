"""
Nhập một cách chia train/val/test từ bên ngoài (vd. cách chia theo nhóm ảnh gần trùng của phương án Random Forest)
thành data/dataset_splits_<tên>.csv, cùng định dạng với data/dataset_splits.csv.

File đầu vào (CSV) cần 2 cột, tên cột được tự nhận diện hoặc chỉ định bằng tham số:
  - đường dẫn ảnh: đường dẫn đầy đủ, tương đối, hoặc chỉ tên file (tên file PlantVillage là duy nhất)
  - tập: train / val (validation, valid) / test
Cột nhóm (group) nếu có sẽ được giữ lại.

Chạy:
  python scripts/import_external_split.py --input <file.csv> --name group
Sau đó huấn luyện và đánh giá trên cách chia này (Windows PowerShell: $env:PV_SPLIT="group"):
  PV_SPLIT=group python scripts/train_tier3.py
  PV_SPLIT=group python scripts/evaluate_tier3.py
"""
import sys
import json
import argparse
from pathlib import Path, PureWindowsPath

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from src.config import DATA_DIR, CLASS_MAPPING_PATH

PATH_COLS = ["rel_path", "filepath", "file_path", "path", "image_path", "img_path", "image", "file", "filename", "file_name"]
SPLIT_COLS = ["split", "set", "subset", "fold", "partition"]
GROUP_COLS = ["group", "group_id", "cluster", "cluster_id"]
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "dev": "val", "test": "test", "testing": "test"}


def pick(df, given, candidates, what, required=True):
    if given:
        if given not in df.columns:
            raise SystemExit(f"Không có cột '{given}'. Các cột hiện có: {list(df.columns)}")
        return given
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    if required:
        raise SystemExit(f"Không tự nhận ra cột {what}. Các cột hiện có: {list(df.columns)}. "
                         f"Hãy chỉ định bằng tham số --{what}-col.")
    return None


def last_parts(p):
    """(thư mục lớp, tên file) từ một đường dẫn kiểu Windows hoặc POSIX."""
    parts = PureWindowsPath(str(p).replace("/", "\\")).parts
    return (parts[-2] if len(parts) >= 2 else None), parts[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="File CSV chia tập từ bên ngoài")
    ap.add_argument("--name", required=True, help="Tên cách chia, vd. group -> data/dataset_splits_group.csv")
    ap.add_argument("--path-col"); ap.add_argument("--split-col"); ap.add_argument("--group-col")
    args = ap.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name chỉ gồm chữ, số, '_' hoặc '-'.")

    ext = pd.read_csv(args.input)
    path_col = pick(ext, args.path_col, PATH_COLS, "path")
    split_col = pick(ext, args.split_col, SPLIT_COLS, "split")
    group_col = pick(ext, args.group_col, GROUP_COLS, "group", required=False)
    print(f"-> {len(ext):,} dòng; cột đường dẫn '{path_col}', cột tập '{split_col}'"
          + (f", cột nhóm '{group_col}'" if group_col else ""))

    ext["_split"] = ext[split_col].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
    bad = ext[ext["_split"].isna()][split_col].unique()
    if len(bad):
        raise SystemExit(f"Giá trị tập không nhận ra: {list(bad)[:10]} (cần train/val/test).")

    # Đối chiếu với toàn bộ ảnh của dự án theo tên file (tên file PlantVillage là duy nhất), kiểm tra thêm thư mục lớp
    ours = pd.read_csv(DATA_DIR / "dataset_splits.csv")[["rel_path", "class_name", "label_idx"]]
    ours["fname"] = ours["rel_path"].map(lambda p: p.rsplit("/", 1)[-1])
    assert ours["fname"].is_unique
    cls_file = ext[path_col].map(last_parts)
    ext["_cls"] = [c for c, _ in cls_file]
    ext["fname"] = [f for _, f in cls_file]
    if ext["fname"].duplicated().any():
        dup = ext[ext["fname"].duplicated(keep=False)]["fname"].head(5).tolist()
        raise SystemExit(f"Một ảnh xuất hiện nhiều lần trong file đầu vào, vd. {dup}")

    merged = ext.merge(ours, on="fname", how="left", indicator=True)
    unmatched = merged[merged["_merge"] == "left_only"]
    wrong_cls = merged[(merged["_merge"] == "both") & merged["_cls"].notna()
                       & (merged["_cls"] != merged["class_name"])]
    missing = ours[~ours["fname"].isin(ext["fname"])]
    print(f"-> Khớp {int((merged['_merge'] == 'both').sum()):,} ảnh; không khớp {len(unmatched):,}; "
          f"sai thư mục lớp {len(wrong_cls):,}; ảnh của dự án không có trong file {len(missing):,}")
    if len(unmatched):
        print("   vd. không khớp:", unmatched[path_col].head(3).tolist())
    if len(wrong_cls):
        raise SystemExit(f"Có ảnh nằm sai lớp so với dữ liệu dự án, vd. {wrong_cls[path_col].head(3).tolist()}")

    out = merged[merged["_merge"] == "both"][["rel_path", "class_name", "label_idx", "_split"]].rename(columns={"_split": "split"})
    if group_col:
        out["group"] = merged.loc[merged["_merge"] == "both", group_col].values
    out_path = DATA_DIR / f"dataset_splits_{args.name}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    counts = out["split"].value_counts()
    print(f"-> Đã lưu {out_path.name}: train {counts.get('train', 0):,} / val {counts.get('val', 0):,} / test {counts.get('test', 0):,}")
    per_split_classes = out.groupby("split")["class_name"].nunique().to_dict()
    n_classes = len(json.load(open(CLASS_MAPPING_PATH, encoding="utf-8"))["class_to_idx"])
    print(f"-> Số lớp mỗi tập: {per_split_classes} (cần {n_classes})")
    if group_col:
        leak = out.groupby("group")["split"].nunique()
        print(f"-> Nhóm nằm ở nhiều tập (rò rỉ): {int((leak > 1).sum())}")
    print(f"✅ Xong. Huấn luyện trên cách chia này: đặt PV_SPLIT={args.name} rồi chạy scripts/train_tier3.py")


if __name__ == "__main__":
    main()
