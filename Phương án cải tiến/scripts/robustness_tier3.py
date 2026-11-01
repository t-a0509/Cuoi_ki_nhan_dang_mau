"""
Kiểm tra độ bền của ResNet50 (Tầng 3): đánh giá lại tập test sau khi biến đổi ảnh
(ảnh xám, làm mờ, giảm sáng, giảm độ phân giải, nhiễu). Các phép biến đổi nằm trong src/corruptions.py.

Đầu ra: results/tier3_robustness.csv, results/tier3_robustness.png
Chạy: python scripts/robustness_tier3.py [--only gray blur]
"""
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
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.config import RESULTS_DIR, MODEL_PATH, NUM_WORKERS, USE_AMP
from src.dataset import PlantVillageDataset, get_transforms
from src.model import load_trained_resnet50
from src.corruptions import CORRUPTIONS, Corrupt


def predict(model, loader, device):
    preds, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=USE_AMP and device.type == "cuda"):
                out = model(images)
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def plot(results, save_path):
    names = results["display_name"].tolist()
    accs = (results["accuracy"] * 100).tolist()
    colors = ["#2e7d4f" if k == "none" else "#9ca3af" for k in results["key"]]
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(names, accs, color=colors)
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 1, f"{a:.1f}%", ha="center", fontsize=10)
    plt.ylim(0, 108)
    plt.ylabel("Accuracy (%)")
    plt.title("ResNet50 fine-tuned: accuracy trên tập test sau khi biến đổi ảnh", fontweight="bold")
    plt.xticks(fontsize=9)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(CORRUPTIONS), help="Chỉ chạy một số phép biến đổi")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 65)
    print("🧪 KIỂM TRA ĐỘ BỀN TẦNG 3 (RESNET50) VỚI ẢNH BỊ BIẾN ĐỔI")
    print("=" * 65)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_resnet50(MODEL_PATH, device)
    _, val_tf = get_transforms()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in args.only:
        tf = transforms.Compose([Corrupt(key)] + list(val_tf.transforms))
        ds = PlantVillageDataset("test", transform=tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
        y, p = predict(model, loader, device)
        acc = accuracy_score(y, p)
        mf1 = f1_score(y, p, average="macro")
        rows.append({"key": key, "display_name": CORRUPTIONS[key][0], "accuracy": acc, "macro_f1": mf1})
        print(f"   * {CORRUPTIONS[key][0]:<28} Accuracy {acc:.2%} | Macro-F1 {mf1:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "tier3_robustness.csv", index=False, encoding="utf-8-sig")
    plot(results, RESULTS_DIR / "tier3_robustness.png")
    print(f"✅ Đã lưu: {RESULTS_DIR / 'tier3_robustness.csv'} và tier3_robustness.png")


if __name__ == "__main__":
    main()
