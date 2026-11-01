"""
Kiểm tra thiên lệch nền của PlantVillage (không phụ thuộc mô hình).
Chỉ dùng màu của 4 góc ảnh 16×16 px (gần như toàn là nền) để huấn luyện Random Forest.
Nếu accuracy cao hơn hẳn mức ngẫu nhiên, nghĩa là nền/ánh sáng khác nhau theo lớp -> mô hình
có thể học "lối tắt" từ nền thay vì vết bệnh (xem Noyan, arXiv:2206.04374).

Đầu ra: results/background_bias.json
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
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.config import RESULTS_DIR
from src.dataset import load_split_df

PATCH = 16


def corner_features(path):
    """Mean + std từng kênh RGB của 4 góc ảnh -> 24 chiều."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    corners = [arr[:PATCH, :PATCH], arr[:PATCH, w - PATCH:], arr[h - PATCH:, :PATCH], arr[h - PATCH:, w - PATCH:]]
    feats = []
    for c in corners:
        flat = c.reshape(-1, 3)
        feats.extend(flat.mean(0))
        feats.extend(flat.std(0))
    return feats


def extract(df, desc):
    X = np.array([corner_features(p) for p in tqdm(df["filepath"], desc=desc)], dtype=np.float32)
    return X, df["label_idx"].to_numpy()


def main():
    print("=" * 65)
    print("🎯 KIỂM TRA THIÊN LỆCH NỀN: CHỈ NHÌN 4 GÓC ẢNH")
    print("=" * 65)
    train_df = load_split_df("train")
    test_df = load_split_df("test")
    X_tr, y_tr = extract(train_df, "Train")
    X_te, y_te = extract(test_df, "Test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    majority = np.bincount(y_tr).argmax()
    result = {
        "features": f"mean+std RGB của 4 góc {PATCH}x{PATCH} px (24 chiều)",
        "classifier": "RandomForest(n_estimators=300)",
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "test_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "random_guess_accuracy": 1.0 / len(np.unique(y_tr)),
        "majority_class_accuracy": float(np.mean(y_te == majority)),
    }
    with open(RESULTS_DIR / "background_bias.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"   * Accuracy chỉ nhìn 4 góc:  {result['test_accuracy']:.2%}")
    print(f"   * Đoán ngẫu nhiên:          {result['random_guess_accuracy']:.2%}")
    print(f"   * Luôn đoán lớp đông nhất:  {result['majority_class_accuracy']:.2%}")
    print(f"✅ Đã lưu: {RESULTS_DIR / 'background_bias.json'}")


if __name__ == "__main__":
    main()
