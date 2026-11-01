"""
So sánh ResNet50 với phương án Random Forest của nhóm trên CÙNG tập test 5.668 ảnh (cách chia theo nhóm ảnh gần trùng,
data/dataset_splits_group.csv do scripts/reproduce_rf_split.py tái tạo).

Số liệu Random Forest lấy thẳng từ kết quả notebook đã in ra (mục 6.5 của
../Phương án cơ sở/Plant_Disease_Detection.ipynb): 4 chỉ số tổng và báo cáo phân loại từng lớp.
Notebook không lưu dự đoán từng ảnh nên KHÔNG làm được kiểm định ghép cặp (McNemar); thay vào đó so khoảng tin cậy
95% của từng mô hình: bootstrap cho ResNet50, Wilson cho accuracy của Random Forest.

Đầu vào: results/group/tier3_test_predictions.csv (PV_SPLIT=group python scripts/evaluate_tier3.py)
Đầu ra: results/comparison/group/ gồm summary.md, metrics.csv, per_class.csv, per_class_f1.png
Chạy: python scripts/compare_with_notebook.py
"""
import re
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
sys.path.insert(0, str(BASE_DIR / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm as normal
from sklearn.metrics import precision_recall_fscore_support

from compare_models import METRICS, species, vi, top_pairs, plot_per_class

NOTEBOOK = BASE_DIR.parent / "Phương án cơ sở" / "Plant_Disease_Detection.ipynb"
PRED_CSV = BASE_DIR / "results" / "group" / "tier3_test_predictions.csv"
OUT_DIR = BASE_DIR / "results" / "comparison" / "group"
NAME_A, NAME_B = "ResNet50", "Random Forest"
N_BOOT = 1000


def notebook_rf_report():
    """Chỉ số tổng và bảng từng lớp của Random Forest trên tập test, đọc từ output của notebook."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            if "KẾT QUẢ TRÊN TẬP TEST" in text:
                break
        else:
            continue
        break
    else:
        raise SystemExit(f"Không tìm thấy kết quả tập test trong {NOTEBOOK.name}")

    overall = {k: float(v) for k, v in re.findall(r"^\s+(accuracy|macro_F1|weighted_F1|balanced_accuracy)\s+([\d.]+)$",
                                                  text, flags=re.M)}
    rows = re.findall(r"^\s*(\S.*?___\S.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$", text, flags=re.M)
    pc = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1", "support"]).set_index("class").astype(float)
    macro = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
    overall["macro_precision"], overall["macro_recall"] = float(macro.group(1)), float(macro.group(2))
    return overall, pc, int(macro.group(4))


def wilson(p, n, z=normal.ppf(0.975)):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    rf, rf_pc, rf_n = notebook_rf_report()
    df = pd.read_csv(PRED_CSV)
    y, p = df["class_name"].to_numpy(), df["pred_class"].to_numpy()
    if len(df) != rf_n:
        raise SystemExit(f"Tập test ResNet50 có {len(df):,} ảnh, Random Forest có {rf_n:,} ảnh: không cùng tập test.")
    sup = df["class_name"].value_counts()
    bad = [c for c in rf_pc.index if sup.get(c, 0) != rf_pc.loc[c, "support"]]
    if bad or len(rf_pc) != 38:
        raise SystemExit(f"Số ảnh từng lớp không khớp notebook: {bad}")
    print(f"-> Cùng tập test: {len(df):,} ảnh, số ảnh của cả 38 lớp khớp notebook")

    rng = np.random.default_rng(0)
    boot = {k: [] for k in METRICS}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for k, f in METRICS.items():
            boot[k].append(f(y[i], p[i]))
    rf_vals = {"Accuracy": rf["accuracy"], "Balanced accuracy": rf["balanced_accuracy"], "Macro-F1": rf["macro_F1"],
               "Weighted-F1": rf["weighted_F1"], "Macro-precision": rf["macro_precision"],
               "Macro-recall": rf["macro_recall"]}
    rows = []
    for k, f in METRICS.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"metric": k, NAME_A: f(y, p), f"{NAME_A}_ci_low": lo, f"{NAME_A}_ci_high": hi,
                     NAME_B: rf_vals[k], "diff_points": (f(y, p) - rf_vals[k]) * 100})
    met = pd.DataFrame(rows)
    rf_lo, rf_hi = wilson(rf["accuracy"], rf_n)

    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=rf_pc.index, zero_division=0)
    pc = pd.DataFrame({"support": rf_pc["support"].astype(int), "recall_a": rc, "recall_b": rf_pc["recall"],
                       "f1_a": f1, "f1_b": rf_pc["f1"]}, index=rf_pc.index)
    pc["f1_diff"] = pc["f1_a"] - pc["f1_b"]
    err = df[~df["correct"]]
    same_sp = float((err["class_name"].map(species) == err["pred_class"].map(species)).mean()) if len(err) else 0.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    met.to_csv(OUT_DIR / "metrics.csv", index=False, encoding="utf-8")
    pc.to_csv(OUT_DIR / "per_class.csv", encoding="utf-8")
    plot_per_class(pc, NAME_A, NAME_B, OUT_DIR / "per_class_f1.png")

    lines = [f"# So sánh {NAME_A} và {NAME_B} trên cùng tập test", "",
             f"- Tập test: {len(df):,} ảnh, cách chia theo nhóm ảnh gần trùng của notebook (data/dataset_splits_group.csv);"
             " số ảnh của cả 38 lớp khớp notebook.",
             f"- {NAME_B}: số liệu notebook in ra ở mục 6.5 (500 cây, đặc trưng thủ công, huấn luyện trên train + val).",
             f"- {NAME_A}: huấn luyện trên train, chọn checkpoint theo val, chấm trên test.", "",
             f"| Chỉ số | {NAME_A} (KTC 95%) | {NAME_B} | Chênh lệch |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r[NAME_A]:.4f} ({r[f'{NAME_A}_ci_low']:.4f}–{r[f'{NAME_A}_ci_high']:.4f}) "
                     f"| {r[NAME_B]:.4f} | +{r['diff_points']:.2f} điểm |")
    lines += ["", f"Accuracy {NAME_B}: KTC 95% (Wilson) {rf_lo:.4f}–{rf_hi:.4f}. Hai khoảng tin cậy không chồng lên nhau.",
              "Macro-precision và macro-recall của Random Forest chỉ có 2 chữ số thập phân trong notebook.",
              "Notebook không lưu dự đoán từng ảnh nên không làm kiểm định McNemar.", "",
              f"## Từng lớp", "",
              # F1 của Random Forest làm tròn 2 chữ số nên chênh dưới 0,005 coi là ngang nhau
              f"- {NAME_A} có F1 cao hơn ở {int((pc['f1_diff'] >= 0.005).sum())}/38 lớp, "
              f"thấp hơn ở {int((pc['f1_diff'] <= -0.005).sum())} lớp, ngang nhau ở "
              f"{int((pc['f1_diff'].abs() < 0.005).sum())} lớp"
              + "".join(f" ({vi(c)}: {r['f1_a']:.3f} so với {r['f1_b']:.2f})"
                        for c, r in pc[pc['f1_diff'].abs() < 0.005].iterrows())
              + ".",
              f"- F1 thấp nhất: {NAME_A} {pc['f1_a'].min():.3f} ({vi(pc['f1_a'].idxmin())}); "
              f"{NAME_B} {pc['f1_b'].min():.2f} ({vi(pc['f1_b'].idxmin())}).",
              f"- Số lớp có F1 < 0,80: {NAME_A} {int((pc['f1_a'] < 0.8).sum())}, {NAME_B} {int((pc['f1_b'] < 0.8).sum())}.",
              "", "Năm lớp Random Forest yếu nhất:", "", f"| Lớp | Số ảnh | F1 {NAME_B} | F1 {NAME_A} |", "|---|---|---|---|"]
    for c, r in pc.sort_values("f1_b").head(5).iterrows():
        lines.append(f"| {vi(c)} | {int(r['support'])} | {r['f1_b']:.2f} | {r['f1_a']:.3f} |")
    lines += ["", f"## Lỗi của {NAME_A}", "",
              f"- {len(err)} ảnh sai / {len(df):,}; {same_sp:.0%} số lỗi nhầm sang bệnh khác trên cùng loài cây.", "",
              "| Lớp thật | Dự đoán | Số ảnh |", "|---|---|---|"]
    for (t, q), n in top_pairs(y, p).items():
        lines.append(f"| {vi(t)} | {vi(q)} | {n} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(met.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"-> {NAME_B} accuracy KTC 95% (Wilson): {rf_lo:.4f}–{rf_hi:.4f}")
    print(f"✅ Đã lưu: {OUT_DIR}")


if __name__ == "__main__":
    main()
