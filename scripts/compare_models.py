"""
So sánh hai mô hình trên CÙNG một tập test, ghép theo từng ảnh.

Đầu vào: hai file CSV dự đoán từng ảnh test.
  - Mô hình A (mặc định ResNet50): results/<cách chia>/tier3_test_predictions.csv do scripts/evaluate_tier3.py tạo ra.
  - Mô hình B (vd. Random Forest): CSV có cột đường dẫn ảnh, nhãn dự đoán, và (nên có) nhãn thật.
    Nhãn có thể là tên lớp (Tomato___Early_blight hoặc TomatoEarlyblight) hoặc chỉ số 0..37 theo thứ tự tên lớp.
Tên cột được tự nhận diện, hoặc chỉ định bằng --b-path-col / --b-true-col / --b-pred-col.

Đầu ra: results/comparison/<tag>/ gồm summary.md, metrics.csv, per_class.csv, per_class_f1.png.
Chạy:
  python scripts/compare_models.py --a results/group/tier3_test_predictions.csv --b rf_test_predictions.csv --tag group
"""
import re
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, precision_recall_fscore_support)

from src.config import CLASS_MAPPING_PATH, CLASS_INFO

PATH_COLS = ["rel_path", "filepath", "file_path", "path", "image_path", "img_path", "image", "file", "filename", "file_name"]
TRUE_COLS = ["class_name", "true", "y_true", "label", "true_label", "target", "actual"]
PRED_COLS = ["pred_class", "pred", "y_pred", "prediction", "predicted", "pred_label"]

METRICS = {
    "Accuracy": lambda y, p: accuracy_score(y, p),
    "Balanced accuracy": lambda y, p: balanced_accuracy_score(y, p),
    "Macro-F1": lambda y, p: f1_score(y, p, average="macro"),
    "Weighted-F1": lambda y, p: f1_score(y, p, average="weighted"),
    "Macro-precision": lambda y, p: precision_score(y, p, average="macro", zero_division=0),
    "Macro-recall": lambda y, p: recall_score(y, p, average="macro", zero_division=0),
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fname(p):
    return PureWindowsPath(str(p).replace("/", "\\")).parts[-1]


def pick(df, given, candidates, what):
    if given:
        return given
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def to_class(series, classes, what):
    """Đưa nhãn (tên lớp viết kiểu bất kỳ, hoặc chỉ số) về tên lớp chuẩn."""
    by_norm = {norm(c): c for c in classes}
    if pd.api.types.is_numeric_dtype(series):
        print(f"   ! Cột {what} là số: hiểu là chỉ số lớp theo thứ tự tên lớp (0 = {classes[0]}).")
        return series.astype(int).map(lambda i: classes[i])
    out = series.map(lambda s: by_norm.get(norm(s)))
    bad = series[out.isna()].unique()
    if len(bad):
        raise SystemExit(f"Không nhận ra nhãn {what}: {list(bad)[:10]}")
    return out


def load_b(path, classes, args):
    df = pd.read_csv(path)
    pc = pick(df, args.b_path_col, PATH_COLS, "path")
    tc = pick(df, args.b_true_col, TRUE_COLS, "true")
    pr = pick(df, args.b_pred_col, PRED_COLS, "pred")
    if pc is None or pr is None:
        raise SystemExit(f"File B cần cột đường dẫn và cột dự đoán. Các cột hiện có: {list(df.columns)}")
    out = pd.DataFrame({"fname": df[pc].map(fname), "pred_b": to_class(df[pr], classes, "dự đoán (B)")})
    if tc is not None:
        out["true_b"] = to_class(df[tc], classes, "thật (B)")
    return out


def species(c):
    return c.split("___")[0]


def vi(c):
    return CLASS_INFO.get(c, {}).get("vi", c)


def paired_bootstrap(y, pa, pb, n_boot, seed):
    rng = np.random.default_rng(seed)
    res = {k: {"a": [], "b": [], "d": []} for k in METRICS}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        for k, f in METRICS.items():
            va, vb = f(y[i], pa[i]), f(y[i], pb[i])
            res[k]["a"].append(va); res[k]["b"].append(vb); res[k]["d"].append(va - vb)
    return {k: {s: np.percentile(v, [2.5, 97.5]) for s, v in d.items()} for k, d in res.items()}


def top_pairs(y, p, n=5):
    err = pd.DataFrame({"true": y[y != p], "pred": p[y != p]})
    return err.groupby(["true", "pred"]).size().sort_values(ascending=False).head(n)


def plot_per_class(pc, name_a, name_b, save):
    pc = pc.sort_values("f1_b")
    y = np.arange(len(pc))
    plt.figure(figsize=(9, 11))
    plt.hlines(y, pc["f1_b"], pc["f1_a"], color="#cbd5e1", linewidth=2)
    plt.scatter(pc["f1_b"], y, color="#9ca3af", label=name_b, zorder=3)
    plt.scatter(pc["f1_a"], y, color="#2563eb", label=name_a, zorder=3)
    plt.yticks(y, [vi(c) for c in pc.index], fontsize=7.5)
    plt.xlabel("F1 trên tập test")
    plt.title(f"F1 từng lớp: {name_a} so với {name_b}", fontweight="bold", pad=28)
    plt.grid(axis="x", alpha=0.3)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="CSV dự đoán của mô hình A (định dạng tier3_test_predictions.csv)")
    ap.add_argument("--b", required=True, help="CSV dự đoán của mô hình B")
    ap.add_argument("--name-a", default="ResNet50"); ap.add_argument("--name-b", default="Random Forest")
    ap.add_argument("--b-path-col"); ap.add_argument("--b-true-col"); ap.add_argument("--b-pred-col")
    ap.add_argument("--tag", default="default", help="Tên thư mục kết quả trong results/comparison/")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    classes = [v for _, v in sorted(json.load(open(CLASS_MAPPING_PATH, encoding="utf-8"))["idx_to_class"].items(),
                                    key=lambda kv: int(kv[0]))]
    a = pd.read_csv(args.a)
    a = pd.DataFrame({"fname": a["rel_path"].map(fname), "true": a["class_name"], "pred_a": a["pred_class"]})
    b = load_b(args.b, classes, args)
    m = a.merge(b, on="fname", how="inner")
    print(f"-> A: {len(a):,} ảnh, B: {len(b):,} ảnh, ghép được {len(m):,} ảnh chung")
    if len(m) < max(len(a), len(b)):
        print(f"   ! {len(a) - len(m):,} ảnh của A và {len(b) - len(m):,} ảnh của B không có ở bên kia: "
              "hai mô hình chưa dùng cùng tập test, kết quả chỉ tính trên phần chung.")
    if "true_b" in m:
        diff = (m["true"] != m["true_b"]).sum()
        if diff:
            raise SystemExit(f"{diff} ảnh có nhãn thật khác nhau giữa hai file: kiểm tra lại cách ghép.")

    y, pa, pb = m["true"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    out = BASE_DIR / "results" / "comparison" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    point = {k: (f(y, pa), f(y, pb)) for k, f in METRICS.items()}
    ci = paired_bootstrap(y, pa, pb, args.n_boot, seed=0)
    rows = [{"metric": k, args.name_a: va, args.name_b: vb, "diff": va - vb,
             "ci_a_lo": ci[k]["a"][0], "ci_a_hi": ci[k]["a"][1], "ci_b_lo": ci[k]["b"][0], "ci_b_hi": ci[k]["b"][1],
             "ci_diff_lo": ci[k]["d"][0], "ci_diff_hi": ci[k]["d"][1]} for k, (va, vb) in point.items()]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")

    ca, cb = pa == y, pb == y
    only_a, only_b, both_wrong = int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum())
    p_mcnemar = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0

    pa_, ra, fa, n = precision_recall_fscore_support(y, pa, labels=classes, zero_division=0)
    pb_, rb, fb, _ = precision_recall_fscore_support(y, pb, labels=classes, zero_division=0)
    pc = pd.DataFrame({"recall_a": ra, "recall_b": rb, "f1_a": fa, "f1_b": fb, "n_test": n}, index=classes)
    pc["f1_gain"] = pc["f1_a"] - pc["f1_b"]
    pc.to_csv(out / "per_class.csv", encoding="utf-8-sig")
    plot_per_class(pc, args.name_a, args.name_b, out / "per_class_f1.png")

    same_sp = lambda p: float(np.mean([species(t) == species(q) for t, q in zip(y[p != y], p[p != y])])) if (p != y).any() else float("nan")
    A, B = args.name_a, args.name_b
    L = [f"# So sánh {A} và {B} trên cùng tập test", "", "*Sinh tự động bởi `scripts/compare_models.py`.*", "",
         f"Số ảnh test chung: **{len(m):,}**.", "", "## Chỉ số tổng hợp (khoảng tin cậy 95%, bootstrap ghép cặp)", "",
         f"| Chỉ số | {A} | {B} | Chênh lệch ({A} − {B}) |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['metric']} | {r[A]:.4f} [{r['ci_a_lo']:.4f}; {r['ci_a_hi']:.4f}] | "
                 f"{r[B]:.4f} [{r['ci_b_lo']:.4f}; {r['ci_b_hi']:.4f}] | "
                 f"{r['diff']:+.4f} [{r['ci_diff_lo']:+.4f}; {r['ci_diff_hi']:+.4f}] |")
    L += ["", "## Kiểm định McNemar", "",
          f"| | Số ảnh |", "|---|---|",
          f"| Chỉ {A} đúng | {only_a:,} |", f"| Chỉ {B} đúng | {only_b:,} |",
          f"| Cả hai cùng sai | {both_wrong:,} |", f"| Cả hai cùng đúng | {int((ca & cb).sum()):,} |", "",
          f"p-value (exact, hai phía) = **{p_mcnemar:.3g}**"
          + (" → chênh lệch có ý nghĩa thống kê ở mức 5%." if p_mcnemar < 0.05 else " → chưa đủ bằng chứng là hai mô hình khác nhau."), "",
          "## Lỗi đúng loài, sai bệnh", "",
          f"| | {A} | {B} |", "|---|---|---|",
          f"| Số ảnh sai | {int((~ca).sum()):,} | {int((~cb).sum()):,} |",
          f"| Tỷ lệ lỗi đúng loài cây | {same_sp(pa):.0%} | {same_sp(pb):.0%} |", "",
          f"## 5 lớp {B} có recall thấp nhất", "",
          f"| Lớp | Số ảnh test | Recall {B} | Recall {A} |", "|---|---|---|---|"]
    for c, r in pc.sort_values("recall_b").head(5).iterrows():
        L.append(f"| {vi(c)} | {int(r.n_test)} | {r.recall_b:.4f} | {r.recall_a:.4f} |")
    L += ["", f"## 5 lớp {A} có recall thấp nhất", "", f"| Lớp | Số ảnh test | Recall {A} | Recall {B} |", "|---|---|---|---|"]
    for c, r in pc.sort_values("recall_a").head(5).iterrows():
        L.append(f"| {vi(c)} | {int(r.n_test)} | {r.recall_a:.4f} | {r.recall_b:.4f} |")
    for name, p in [(B, pb), (A, pa)]:
        L += ["", f"## 5 cặp {name} nhầm nhiều nhất", "", f"| Thực tế | Dự đoán | {B} | {A} |", "|---|---|---|---|"]
        pairs_a = pd.DataFrame({"t": y, "p": pa})[pa != y].groupby(["t", "p"]).size()
        pairs_b = pd.DataFrame({"t": y, "p": pb})[pb != y].groupby(["t", "p"]).size()
        for (t, q), _ in top_pairs(y, p).items():
            L.append(f"| {vi(t)} | {vi(q)} | {pairs_b.get((t, q), 0)} | {pairs_a.get((t, q), 0)} |")
    L += ["", "## Hình", "", "- `per_class_f1.png`: F1 từng lớp của hai mô hình", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")

    print(f"-> Macro-F1: {A} {point['Macro-F1'][0]:.4f} | {B} {point['Macro-F1'][1]:.4f}; McNemar p = {p_mcnemar:.3g}")
    print(f"✅ Đã lưu: {out}")


if __name__ == "__main__":
    main()
