"""
Đo mức rò rỉ ảnh gần trùng giữa train và test của một cách chia (mặc định: data/dataset_splits.csv).

pHash 64 bit giống thư viện imagehash: ảnh xám 32×32 → DCT 2 chiều → khối 8×8 tần số thấp → so với trung vị.
Hai ảnh CÙNG LỚP có khoảng cách Hamming ≤ ngưỡng (mặc định 4) được nối với nhau; các ảnh nối nhau (Union-Find)
tạo thành một nhóm. Một ảnh test "bị rò rỉ" nếu nhóm của nó có ảnh nằm trong tập train.

Đầu ra: results/near_duplicates.json, results/phash.csv (lưu hash để chạy lại nhanh).
Chạy: python scripts/near_duplicate_check.py [--threshold 4]
"""
import sys
import json
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
from PIL import Image
from scipy.fft import dct
from tqdm import tqdm

from src.config import RESULTS_DIR, SPLITS_CSV
from src.dataset import load_split_df

THRESHOLDS = [0, 2, 4, 6, 8, 10, 12]
HASH_CACHE = BASE_DIR / "results" / "phash.csv"


def phash(path):
    img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    d = dct(dct(np.asarray(img, dtype=np.float64), axis=0, norm=None), axis=1, norm=None)
    low = d[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def close_pairs(hashes, max_dist, chunk=1024):
    """Các cặp (i, j), i < j, có khoảng cách Hamming ≤ max_dist, kèm khoảng cách."""
    h = np.asarray(hashes, dtype=np.uint64)
    out = []
    for s in range(0, len(h), chunk):
        block = h[s:s + chunk, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= max_dist)
        keep = jj > ii + s
        out.append(np.stack([ii[keep] + s, jj[keep], dist[ii[keep], jj[keep]]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 3), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=4)
    args = ap.parse_args()

    df = load_split_df()
    if HASH_CACHE.exists():
        cache = pd.read_csv(HASH_CACHE)
        df = df.merge(cache, on="rel_path", how="left")
    if "phash" not in df or df["phash"].isna().any():
        todo = df["phash"].isna() if "phash" in df else pd.Series(True, index=df.index)
        df.loc[todo, "phash"] = [phash(p) for p in tqdm(df.loc[todo, "filepath"], desc="pHash")]
        df[["rel_path", "phash"]].astype({"phash": "uint64"}).to_csv(HASH_CACHE, index=False)
    df["phash"] = df["phash"].astype("uint64")

    # Cặp gần trùng trong từng lớp
    pairs = []
    for cls, g in df.groupby("class_name"):
        idx = g.index.to_numpy()
        for i, j, d in close_pairs(g["phash"].to_numpy(), max(THRESHOLDS)):
            pairs.append((idx[i], idx[j], d))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 3)

    by_threshold = []
    groups_at = None
    for t in THRESHOLDS:
        uf = UnionFind(len(df))
        for i, j, d in pairs[pairs[:, 2] <= t]:
            uf.union(int(i), int(j))
        roots = np.array([uf.find(i) for i in range(len(df))])
        sizes = pd.Series(roots).value_counts()
        by_threshold.append({"threshold": t, "groups": int(len(sizes)),
                             "images_in_groups_ge2": int(sizes[sizes >= 2].sum()), "largest_group": int(sizes.max())})
        if t == args.threshold:
            groups_at = roots

    df["group"] = groups_at
    train_groups = set(df.loc[df["split"] == "train", "group"])
    test = df[df["split"] == "test"].copy()
    test["leaked"] = test["group"].isin(train_groups)

    result = {"split_file": SPLITS_CSV.name, "threshold": args.threshold,
              "pairs_within_class_le_12": int(len(pairs)), "by_threshold": by_threshold,
              "test_images": int(len(test)), "test_leaked": int(test["leaked"].sum())}

    # Độ chính xác của ResNet50 trên ảnh test bị rò rỉ và không bị rò rỉ (nếu đã có file dự đoán)
    pred_path = RESULTS_DIR / "tier3_test_predictions.csv"
    if pred_path.exists():
        pr = pd.read_csv(pred_path)[["rel_path", "correct"]]
        t2 = test.merge(pr, on="rel_path", how="inner")
        for flag, name in [(True, "leaked"), (False, "not_leaked")]:
            part = t2[t2["leaked"] == flag]
            result[f"resnet_acc_{name}"] = float(part["correct"].mean()) if len(part) else None
            result[f"resnet_n_{name}"] = int(len(part))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "near_duplicates.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"-> Cặp ảnh cùng lớp có Hamming ≤ 12: {len(pairs):,}")
    for r in by_threshold:
        print(f"   ngưỡng {r['threshold']:>2}: {r['groups']:,} nhóm, {r['images_in_groups_ge2']:,} ảnh trong nhóm ≥2, nhóm lớn nhất {r['largest_group']:,}")
    print(f"-> Ảnh test có ảnh gần trùng trong train (ngưỡng {args.threshold}): {result['test_leaked']:,} / {result['test_images']:,}")
    if "resnet_acc_leaked" in result:
        print(f"-> ResNet50 accuracy: rò rỉ {result['resnet_acc_leaked']}, không rò rỉ {result['resnet_acc_not_leaked']:.4f}")
    print(f"✅ Đã lưu: {RESULTS_DIR / 'near_duplicates.json'}")


if __name__ == "__main__":
    main()
