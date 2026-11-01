"""
Thống kê mô tả ảnh của toàn bộ bộ dữ liệu (bản color, 54.305 ảnh), bổ sung cho 01_eda_and_split.py.

Với mỗi ảnh tính: kích thước, chế độ màu, trung bình R/G/B, trung bình S và V (HSV), và độ sáng trung bình của
4 góc 16×16 px (gần như toàn là nền). Kiểm định Kruskal-Wallis cho biết các đại lượng này có khác nhau giữa
38 lớp hay không; epsilon² đo mức độ khác nhau (0: không khác, 1: lớp quyết định hoàn toàn).

Đầu ra: results/eda/ gồm image_stats.csv, summary.json, brightness_by_class.png, healthy_vs_diseased.png
Chạy: python scripts/eda_stats.py
"""
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import kruskal, mannwhitneyu
from tqdm import tqdm

from src.config import CLASS_INFO
from src.dataset import load_split_df

OUT_DIR = BASE_DIR / "results" / "eda"
CORNER = 16


def image_stats(path):
    img = Image.open(path)
    w, h, mode = img.width, img.height, img.mode
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    c = CORNER
    corners = np.concatenate([gray[:c, :c].ravel(), gray[:c, -c:].ravel(), gray[-c:, :c].ravel(), gray[-c:, -c:].ravel()])
    return (w, h, mode, *rgb.reshape(-1, 3).mean(axis=0), hsv[..., 1].mean(), hsv[..., 2].mean(),
            corners.mean(), corners.std())


def epsilon_squared(h, n):
    return h / ((n * n - 1) / (n + 1))


def main():
    df = load_split_df()[["rel_path", "class_name", "filepath"]]
    with ProcessPoolExecutor() as ex:
        rows = list(tqdm(ex.map(image_stats, df["filepath"], chunksize=256), total=len(df), desc="Ảnh"))
    cols = ["width", "height", "mode", "mean_R", "mean_G", "mean_B", "mean_S", "mean_V", "corner_mean", "corner_std"]
    st = pd.concat([df[["rel_path", "class_name"]].reset_index(drop=True), pd.DataFrame(rows, columns=cols)], axis=1)
    st["healthy"] = st["class_name"].map(lambda c: CLASS_INFO.get(c, {}).get("status") == "healthy")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st.to_csv(OUT_DIR / "image_stats.csv", index=False, encoding="utf-8")

    n = len(st)
    summary = {"n_images": n,
               "sizes": {f"{w}x{h}": int(k) for (w, h), k in st.groupby(["width", "height"]).size().items()},
               "modes": {m: int(k) for m, k in st["mode"].value_counts().items()}}
    for col in ["mean_V", "mean_S", "corner_mean"]:
        h, p = kruskal(*[g[col].to_numpy() for _, g in st.groupby("class_name")])
        by_cls = st.groupby("class_name")[col].median()
        summary[col] = {"overall_median": float(st[col].median()), "kruskal_H": float(h), "p_value": float(p),
                        "epsilon_squared": float(epsilon_squared(h, n)),
                        "class_median_min": [by_cls.idxmin(), float(by_cls.min())],
                        "class_median_max": [by_cls.idxmax(), float(by_cls.max())]}
    for col in ["mean_S", "mean_V", "mean_G"]:
        a, b = st.loc[st["healthy"], col], st.loc[~st["healthy"], col]
        u, p = mannwhitneyu(a, b)
        summary[f"healthy_vs_diseased_{col}"] = {"median_healthy": float(a.median()), "median_diseased": float(b.median()),
                                                 "U": float(u), "p_value": float(p),
                                                 "auc": float(u / (len(a) * len(b)))}
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    vi = lambda c: CLASS_INFO.get(c, {}).get("vi", c)
    order = st.groupby("class_name")["corner_mean"].median().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True)
    for ax, col, title in [(axes[0], "corner_mean", "Độ sáng nền (4 góc 16×16 px)"),
                           (axes[1], "mean_V", "Độ sáng toàn ảnh (kênh V)")]:
        ax.boxplot([st.loc[st["class_name"] == c, col] for c in order], orientation="horizontal", showfliers=False,
                   widths=0.6, patch_artist=True, boxprops=dict(facecolor="#dbe7f7", color="#2563eb"),
                   medianprops=dict(color="#1e3a8a"))
        ax.set_title(f"{title}\nε² = {summary[col]['epsilon_squared']:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Giá trị 0–255")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(1, len(order) + 1), [vi(c) for c in order], fontsize=7.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "brightness_by_class.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, name in zip(axes, ["mean_G", "mean_S", "mean_V"], ["Trung bình kênh G", "Độ bão hòa S", "Độ sáng V"]):
        bins = np.linspace(st[col].quantile(0.005), st[col].quantile(0.995), 50)
        ax.hist(st.loc[st["healthy"], col], bins=bins, density=True, alpha=0.6, color="#16a34a", label="Lá khỏe")
        ax.hist(st.loc[~st["healthy"], col], bins=bins, density=True, alpha=0.6, color="#b45309", label="Lá bệnh")
        ax.set_title(f"{name} (AUC = {summary[f'healthy_vs_diseased_{col}']['auc']:.2f})", fontsize=10)
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "healthy_vs_diseased.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ Đã lưu: {OUT_DIR}")


if __name__ == "__main__":
    main()
