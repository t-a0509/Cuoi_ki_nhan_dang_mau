"""
Tái tạo đúng cách chia train/val/test theo nhóm ảnh gần trùng của phương án Random Forest
(notebook Phương án cơ sở/Nhận_dạng_mẫu/Plant_Disease_Detection.ipynb, mục 5.4–5.6), rồi ghi ra
data/dataset_splits_group.csv cùng định dạng với data/dataset_splits.csv.

Các hàm dưới đây chép nguyên logic của notebook:
  - thứ tự ảnh: lớp sắp theo tên, trong mỗi lớp ảnh color sắp theo tên (sorted(os.listdir))
  - pHash: ảnh BẢN GRAYSCALE của bộ dữ liệu, cv2.resize về 128×128 rồi 32×32, cv2.dct, khối 8×8,
    so với trung vị của 63 hệ số (bỏ hệ số DC), đóng gói 64 bit
  - nhóm: cặp ảnh cùng lớp có Hamming ≤ 4, Union-Find lấy chỉ số nhỏ nhất làm gốc
  - chia: StratifiedGroupKFold (shuffle, seed 42; thử 42, 7, 13, 2024, 99 nếu thiếu lớp), 80% → rồi 50/50

Để chắc chắn tái tạo đúng, script đối chiếu với các con số notebook đã in ra và dừng nếu lệch.
Lưu ý: kết quả chia phụ thuộc phiên bản scikit-learn. Notebook chạy scikit-learn 1.6.1; bản mới hơn
(vd. 1.9.1) chia ra khác. Vì vậy chạy hai bước:
  1. Gom nhóm (môi trường của dự án, cần OpenCV) → data/rf_groups.csv
       python scripts/reproduce_rf_split.py --variants-root "<thư mục chứa color/ và grayscale/>"
  2. Chia tập bằng scikit-learn 1.6.1 (pip install scikit-learn==1.6.1 numpy==2.1.3 pandas trong môi trường riêng)
       python scripts/reproduce_rf_split.py --from-groups
"""
import os
import sys
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = BASE_DIR / "data"
GROUPS_CSV = DATA_DIR / "rf_groups.csv"   # tương đương artifacts/groups.csv của notebook

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMG_SIZE = (128, 128)
HAMMING_THRESHOLD = 4
MAX_HAMMING = 12
# Các con số notebook đã in ra (mục 5.5, 5.6)
EXPECTED = {"pairs_le_12": 267561, "groups": 53393, "in_groups_ge2": 1317, "largest": 151,
            "train": 43238, "val": 5399, "test": 5668}


def _index_variant_dir(folder, suffix=""):
    by_stem, by_core = {}, {}
    if not folder or not os.path.isdir(folder):
        return by_stem, by_core
    for f in os.listdir(folder):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        if suffix and stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
        path = os.path.join(folder, f)
        by_stem.setdefault(stem.lower().strip(), path)
        by_core.setdefault(stem.split("___")[-1].lower().strip(), path)
    return by_stem, by_core


def build_variant_index(color_dir, gray_dir):
    gray_stem, gray_core = _index_variant_dir(gray_dir)
    mapping = {}
    for cf in sorted(os.listdir(color_dir)):
        if not cf.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(cf)[0]
        g = gray_stem.get(stem.lower().strip()) or gray_core.get(stem.split("___")[-1].lower().strip())
        mapping[cf] = g
    return mapping


def perceptual_hash(gray_path):
    import cv2
    img_gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    small = cv2.resize(img_gray, (32, 32)).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    med = np.median(dct[1:])
    return int.from_bytes(np.packbits((dct > med).astype(np.uint8)).tobytes(), "big")


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def split_by_group(idx, labels, groups, test_size):
    n_splits = round(1 / test_size)
    for seed in (42, 7, 13, 2024, 99):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(iter(splitter.split(idx, labels, groups[idx])))
        if len(np.unique(labels[test_idx])) == len(np.unique(labels)):
            return idx[train_idx], idx[test_idx]
    raise RuntimeError("Không chia được tập theo nhóm mà vẫn giữ đủ các lớp.")


def check(name, got):
    want = EXPECTED[name]
    mark = "khớp" if got == want else f"LỆCH (notebook: {want:,})"
    print(f"   {name:<14} {got:>8,}  {mark}")
    return got == want


def build_groups(variants_root):
    from joblib import Parallel, delayed
    from tqdm import tqdm
    color_root, gray_root = Path(variants_root) / "color", Path(variants_root) / "grayscale"
    paths, gray_paths, y = [], [], []
    for cls in sorted(os.listdir(color_root)):
        if not (color_root / cls).is_dir():
            continue
        for cf, g in build_variant_index(color_root / cls, gray_root / cls).items():
            if g is None:
                continue
            paths.append(f"{cls}/{cf}"); gray_paths.append(g); y.append(cls)
    y = np.array(y)
    print(f"-> {len(paths):,} ảnh có đủ bản grayscale")

    phashes = np.array(Parallel(n_jobs=-1)(delayed(perceptual_hash)(g) for g in tqdm(gray_paths, desc="pHash")),
                       dtype=np.uint64)
    near_pairs = []
    for cls in sorted(np.unique(y)):
        idx = np.flatnonzero(y == cls)
        hashes = phashes[idx]
        for start in range(0, len(hashes), 512):
            dist = np.bitwise_count(hashes[start:start + 512, None] ^ hashes[None, :])
            r, c = np.nonzero(dist <= MAX_HAMMING)
            keep = (r + start) < c
            near_pairs.extend(zip(idx[start + r[keep]], idx[c[keep]], dist[r[keep], c[keep]]))
    print("-> Đối chiếu gom nhóm với notebook:")
    check("pairs_le_12", len(near_pairs))

    uf = UnionFind(len(y))
    for i, j, d in near_pairs:
        if d <= HAMMING_THRESHOLD:
            uf.union(int(i), int(j))
    groups = np.array([uf.find(k) for k in range(len(y))])
    pd.DataFrame({"key": paths, "label": y, "group": groups}).to_csv(GROUPS_CSV, index=False, encoding="utf-8")
    print(f"-> Đã lưu nhóm: {GROUPS_CSV}")
    return paths, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-root", help="Thư mục chứa color/ và grayscale/ của PlantVillage")
    ap.add_argument("--from-groups", action="store_true", help=f"Bỏ qua pHash, đọc nhóm từ {GROUPS_CSV.name}")
    ap.add_argument("--name", default="group")
    args = ap.parse_args()

    if args.from_groups:
        g = pd.read_csv(GROUPS_CSV)
        paths, y, groups = g["key"].tolist(), g["label"].to_numpy(), g["group"].to_numpy()
    elif args.variants_root:
        paths, y, groups = build_groups(args.variants_root)
    else:
        raise SystemExit("Cần --variants-root hoặc --from-groups.")
    sizes = pd.Series(groups).value_counts()
    print(f"-> scikit-learn {sklearn.__version__} (notebook dùng 1.6.1)")

    classes = sorted(np.unique(y))
    y_enc = np.array([classes.index(c) for c in y], dtype=np.int32)
    all_idx = np.arange(len(y_enc))
    train_idx, rest = split_by_group(all_idx, y_enc, groups, test_size=0.20)
    val_idx, test_idx = split_by_group(rest, y_enc[rest], groups, test_size=0.50)

    print("-> Đối chiếu với notebook:")
    ok = all([check("groups", int(len(sizes))),
              check("in_groups_ge2", int(sizes[sizes > 1].sum())), check("largest", int(sizes.iloc[0])),
              check("train", len(train_idx)), check("val", len(val_idx)), check("test", len(test_idx))])

    split = np.empty(len(y), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    ours = pd.read_csv(DATA_DIR / "dataset_splits.csv")[["rel_path", "class_name", "label_idx"]]
    ours["key"] = ours["rel_path"].str.split("/color/").str[1]
    out = pd.DataFrame({"key": paths, "split": split, "group": groups}).merge(ours, on="key", how="inner")
    if len(out) != len(paths):
        raise SystemExit(f"Chỉ ghép được {len(out):,}/{len(paths):,} ảnh với dữ liệu dự án.")
    if not ok:
        raise SystemExit("Không tái tạo đúng cách chia của notebook, KHÔNG ghi file.")
    out_path = DATA_DIR / f"dataset_splits_{args.name}.csv"
    out[["rel_path", "class_name", "label_idx", "split", "group"]].to_csv(out_path, index=False, encoding="utf-8")
    leak = out.groupby("group")["split"].nunique()
    print(f"-> Nhóm nằm ở nhiều tập: {int((leak > 1).sum())}")
    print(f"✅ Tái tạo đúng. Đã lưu: {out_path}")


if __name__ == "__main__":
    main()
