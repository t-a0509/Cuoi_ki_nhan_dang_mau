"""
Phân tích lỗi của ResNet50 (Tầng 3) trên tập test.
Cần chạy scripts/evaluate_tier3.py trước để có tier3_test_predictions.csv và tier3_test_probs.npy.

Đầu ra (results/tier3_analysis/):
  - confused_pairs.csv          : các cặp (thật -> đoán) bị nhầm nhiều nhất
  - per_class_metrics.csv       : precision/recall/F1 từng lớp
  - confidence_threshold.csv    : độ phủ & accuracy khi chỉ trả lời lúc độ tin cậy >= ngưỡng
  - confidence_hist.png, threshold_curve.png
  - error_gallery.jpg           : các ảnh bị đoán sai với độ tin cậy cao nhất
  - gradcam_correct_vs_wrong.jpg: Grad-CAM của ảnh đúng và ảnh sai
  - summary.md                  : tóm tắt số liệu để đưa vào báo cáo
"""
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, f1_score

from src.config import RESULTS_DIR, MODEL_PATH, CLASS_MAPPING_PATH, CLASS_INFO
from src.dataset import resolve_image_path, get_transforms
from src.model import load_trained_resnet50, GradCAM, overlay_heatmap

OUT_DIR = RESULTS_DIR / "tier3_analysis"
# Ngưỡng được đề xuất là ngưỡng nhỏ nhất mà accuracy trên phần ảnh được trả lời đạt mức này
TARGET_ACCEPTED_ACC = 0.995


def species(cls):
    return cls.split("___")[0]


def short_name(cls):
    return cls.replace("___", " / ").replace("_", " ")


def vi_name(cls):
    return CLASS_INFO.get(cls, {}).get("vi", cls)


def load_inputs():
    pred_path = RESULTS_DIR / "tier3_test_predictions.csv"
    probs_path = RESULTS_DIR / "tier3_test_probs.npy"
    if not pred_path.exists() or not probs_path.exists():
        raise FileNotFoundError("Chưa có file dự đoán. Hãy chạy scripts/evaluate_tier3.py trước.")
    df = pd.read_csv(pred_path)
    probs = np.load(probs_path)
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        idx_to_class = {int(k): v for k, v in json.load(f)["idx_to_class"].items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return df, probs, class_names


def topk_accuracy(probs, labels, k):
    topk = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def confused_pairs(errors, top=15):
    pairs = (errors.groupby(["class_name", "pred_class"]).size()
             .reset_index(name="count").sort_values("count", ascending=False))
    pairs["same_species"] = pairs["class_name"].map(species) == pairs["pred_class"].map(species)
    pairs.to_csv(OUT_DIR / "confused_pairs.csv", index=False, encoding="utf-8-sig")
    return pairs.head(top)


def per_class_metrics(df, class_names):
    p, r, f1, support = precision_recall_fscore_support(
        df["label_idx"], df["pred_idx"], labels=range(len(class_names)), zero_division=0)
    per_class = pd.DataFrame({
        "class_name": class_names, "precision": p, "recall": r, "f1": f1, "support": support
    }).sort_values("f1")
    per_class.to_csv(OUT_DIR / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    return per_class


def confidence_analysis(df):
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    plt.hist(df.loc[df["correct"], "confidence"], bins=bins, alpha=0.75, label="Đoán đúng", color="#10b981")
    plt.hist(df.loc[~df["correct"], "confidence"], bins=bins, alpha=0.85, label="Đoán sai", color="#ef4444")
    plt.yscale("log")
    plt.xlabel("Độ tin cậy (xác suất softmax lớn nhất)")
    plt.ylabel("Số ảnh (thang log)")
    plt.title("Phân bố độ tin cậy: ảnh đoán đúng và ảnh đoán sai", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_hist.png", dpi=200)
    plt.close()


def threshold_analysis(df):
    """Chỉ trả lời khi độ tin cậy >= ngưỡng, còn lại báo 'không chắc chắn'."""
    rows = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        accepted = df["confidence"] >= t
        n_acc = int(accepted.sum())
        rows.append({
            "threshold": float(t),
            "coverage": n_acc / len(df),
            "accepted_accuracy": float(df.loc[accepted, "correct"].mean()) if n_acc else float("nan"),
            "errors_accepted": int((accepted & ~df["correct"]).sum()),
            "correct_rejected": int((~accepted & df["correct"]).sum()),
        })
    th = pd.DataFrame(rows)
    th.to_csv(OUT_DIR / "confidence_threshold.csv", index=False, encoding="utf-8-sig")

    ok = th[th["accepted_accuracy"] >= TARGET_ACCEPTED_ACC]
    chosen = ok.iloc[0] if len(ok) else th.iloc[th["accepted_accuracy"].idxmax()]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(th["threshold"], th["accepted_accuracy"] * 100, "o-", color="#10b981", label="Accuracy trên ảnh được trả lời")
    ax1.set_xlabel("Ngưỡng độ tin cậy")
    ax1.set_ylabel("Accuracy (%)", color="#10b981")
    ax2 = ax1.twinx()
    ax2.plot(th["threshold"], th["coverage"] * 100, "s--", color="#3b82f6", label="Độ phủ (% ảnh được trả lời)")
    ax2.set_ylabel("Độ phủ (%)", color="#3b82f6")
    ax1.axvline(chosen["threshold"], color="gray", linestyle=":", label=f"Ngưỡng đề xuất {chosen['threshold']:.2f}")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="lower left", fontsize=9)
    ax1.grid(alpha=0.3)
    plt.title("Đánh đổi giữa độ phủ và độ chính xác khi đặt ngưỡng", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "threshold_curve.png", dpi=200)
    plt.close()

    # Demo web đọc file này để cảnh báo "không chắc chắn"
    with open(RESULTS_DIR / "tier3_confidence_threshold.json", "w", encoding="utf-8") as f:
        json.dump({
            "threshold": float(chosen["threshold"]),
            "coverage": float(chosen["coverage"]),
            "accepted_accuracy": float(chosen["accepted_accuracy"]),
            "rule": f"ngưỡng nhỏ nhất có accuracy trên ảnh được trả lời >= {TARGET_ACCEPTED_ACC:.1%} (chọn trên tập test)"
        }, f, ensure_ascii=False, indent=2)
    return th, chosen


def error_gallery(errors, n=20):
    worst = errors.sort_values("confidence", ascending=False).head(n)
    cols = 5
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 4.4))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, r) in zip(axes, worst.iterrows()):
        ax.imshow(Image.open(resolve_image_path(r["rel_path"])).convert("RGB"))
        ax.set_title(f"Thật: {short_name(r['class_name'])}\nĐoán: {short_name(r['pred_class'])}\n({r['confidence']:.0%})",
                     fontsize=7.5, color="#b91c1c")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"{len(worst)} ảnh bị đoán sai có độ tin cậy cao nhất", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / "error_gallery.jpg", dpi=150)
    plt.close()


def gradcam_grid(df, n_each=4, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_resnet50(MODEL_PATH, device)
    cam_engine = GradCAM(model)
    _, tf = get_transforms()

    correct = (df[df["correct"]].groupby("class_name").head(1)
               .sample(n=n_each, random_state=seed))
    wrong = df[~df["correct"]].sort_values("confidence", ascending=False).head(n_each)

    fig, axes = plt.subplots(2, n_each * 2, figsize=(n_each * 2 * 2.3, 6))
    for row, (subset, color, tag) in enumerate([(correct, "#047857", "ĐÚNG"), (wrong, "#b91c1c", "SAI")]):
        for j, (_, r) in enumerate(subset.iterrows()):
            img = Image.open(resolve_image_path(r["rel_path"])).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            cam, _ = cam_engine.generate_heatmap(x, target_class=int(r["pred_idx"]))
            axes[row, 2 * j].imshow(img.resize((224, 224)))
            axes[row, 2 * j].set_title(f"[{tag}] Thật:\n{short_name(r['class_name'])}", fontsize=7, color=color)
            axes[row, 2 * j + 1].imshow(overlay_heatmap(img, cam))
            axes[row, 2 * j + 1].set_title(f"Đoán ({r['confidence']:.0%}):\n{short_name(r['pred_class'])}", fontsize=7, color=color)
    for ax in axes.flatten():
        ax.axis("off")
    plt.suptitle("Grad-CAM (layer4): hàng trên đoán đúng, hàng dưới đoán sai", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gradcam_correct_vs_wrong.jpg", dpi=150)
    plt.close()


def bootstrap_ci(df, n_boot=1000, seed=0):
    """Khoảng tin cậy 95% của accuracy và macro-F1 bằng bootstrap trên tập test (seed cố định để tái lập)."""
    y = df["label_idx"].to_numpy()
    p = df["pred_idx"].to_numpy()
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        accs.append(np.mean(p[idx] == y[idx]))
        f1s.append(f1_score(y[idx], p[idx], average="macro"))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def write_summary(df, probs, pairs, per_class, th, chosen, errors):
    labels = df["label_idx"].to_numpy()
    acc_ci, f1_ci = bootstrap_ci(df)
    same_sp = int((errors["class_name"].map(species) == errors["pred_class"].map(species)).sum())
    conf_ok = df.loc[df["correct"], "confidence"]
    conf_bad = df.loc[~df["correct"], "confidence"]

    lines = [
        "# Phân tích lỗi ResNet50 (Tầng 3) trên tập test",
        "",
        "*Sinh tự động bởi `scripts/analyze_errors_tier3.py`.*",
        "",
        "## Tổng quan",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---|",
        f"| Số ảnh test | {len(df):,} |",
        f"| Top-1 accuracy | {df['correct'].mean():.2%} (bootstrap 95%: {acc_ci[0]:.2%} – {acc_ci[1]:.2%}) |",
        f"| Macro-F1 | {f1_score(df['label_idx'], df['pred_idx'], average='macro'):.3f} "
        f"(bootstrap 95%: {f1_ci[0]:.3f} – {f1_ci[1]:.3f}) |",
        f"| Top-3 accuracy | {topk_accuracy(probs, labels, 3):.2%} |",
        f"| Top-5 accuracy | {topk_accuracy(probs, labels, 5):.2%} |",
        f"| Số ảnh sai | {len(errors)} |",
        f"| Sai bệnh nhưng đúng loài cây | {same_sp} ({same_sp / max(len(errors), 1):.0%} số lỗi) |",
        f"| Sai cả loài cây | {len(errors) - same_sp} |",
        "",
        "## Độ tin cậy",
        "",
        "| | Trung vị | Trung bình | Lớn nhất |",
        "|---|---|---|---|",
        f"| Ảnh đoán đúng | {conf_ok.median():.1%} | {conf_ok.mean():.1%} | {conf_ok.max():.1%} |",
        f"| Ảnh đoán sai | {conf_bad.median():.1%} | {conf_bad.mean():.1%} | {conf_bad.max():.1%} |",
        "",
        f"Ngưỡng đề xuất: **{chosen['threshold']:.2f}** — hệ thống trả lời {chosen['coverage']:.1%} số ảnh "
        f"với accuracy {chosen['accepted_accuracy']:.2%}; phần còn lại báo \"không chắc chắn\". "
        "Lưu ý: ngưỡng được chọn trên chính tập test nên con số này hơi lạc quan.",
        "",
        "| Ngưỡng | Độ phủ | Accuracy phần được trả lời | Lỗi còn lọt | Ảnh đúng bị từ chối |",
        "|---|---|---|---|---|",
    ]
    for _, r in th.iterrows():
        lines.append(f"| {r['threshold']:.2f} | {r['coverage']:.1%} | {r['accepted_accuracy']:.2%} | "
                     f"{int(r['errors_accepted'])} | {int(r['correct_rejected'])} |")
    lines += [
        "",
        "## Các cặp bị nhầm nhiều nhất",
        "",
        "| Thực tế | Dự đoán | Số ảnh | Cùng loài? |",
        "|---|---|---|---|",
    ]
    for _, r in pairs.iterrows():
        lines.append(f"| {vi_name(r['class_name'])} | {vi_name(r['pred_class'])} | {r['count']} | "
                     f"{'có' if r['same_species'] else 'không'} |")
    lines += [
        "",
        "## 10 lớp có F1 thấp nhất",
        "",
        "| Lớp | Precision | Recall | F1 | Số ảnh test |",
        "|---|---|---|---|---|",
    ]
    for _, r in per_class.head(10).iterrows():
        lines.append(f"| {vi_name(r['class_name'])} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1']:.3f} | {int(r['support'])} |")
    lines += [
        "",
        "## Hình",
        "",
        "- `confidence_hist.png` — phân bố độ tin cậy ảnh đúng/sai",
        "- `threshold_curve.png` — đánh đổi độ phủ / độ chính xác",
        "- `error_gallery.jpg` — ảnh sai có độ tin cậy cao nhất",
        "- `gradcam_correct_vs_wrong.jpg` — vùng mô hình chú ý khi đúng và khi sai",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 65)
    print("🔍 PHÂN TÍCH LỖI TẦNG 3 (RESNET50) TRÊN TẬP TEST")
    print("=" * 65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, probs, class_names = load_inputs()
    errors = df[~df["correct"]]
    print(f"-> {len(df):,} ảnh test, {len(errors)} ảnh sai")

    pairs = confused_pairs(errors)
    per_class = per_class_metrics(df, class_names)
    confidence_analysis(df)
    th, chosen = threshold_analysis(df)
    print(f"-> Ngưỡng đề xuất: {chosen['threshold']:.2f} (độ phủ {chosen['coverage']:.1%}, "
          f"accuracy {chosen['accepted_accuracy']:.2%})")
    error_gallery(errors)
    print("-> Đang tạo Grad-CAM...")
    gradcam_grid(df)
    write_summary(df, probs, pairs, per_class, th, chosen, errors)
    print(f"✅ Đã lưu kết quả vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
