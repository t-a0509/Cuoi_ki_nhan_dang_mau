import os
import io
import sys
import json
import time
import base64
import threading
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

# Thêm thư mục gốc vào path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    MODELS_DIR, RESULTS_DIR, MODEL_PATH, CLASS_MAPPING_PATH,
    NUM_CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_INFO
)
from src.model import load_trained_resnet50, GradCAM, overlay_heatmap
from src.corruptions import CORRUPTIONS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Cấu hình tải mô hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIER3_MODEL = None
TIER3_CAM = None
CLASS_MAPPING = None
# Flask chạy đa luồng; Grad-CAM lưu activations/gradients trong hook nên phải khóa khi suy luận
INFER_LOCK = threading.Lock()

# Ngưỡng "không chắc chắn": lấy từ scripts/analyze_errors_tier3.py nếu đã chạy, không thì dùng mặc định
DEFAULT_THRESHOLD = 0.6


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def get_threshold():
    data = read_json(RESULTS_DIR / "tier3_confidence_threshold.json", {})
    return float(data.get("threshold", DEFAULT_THRESHOLD))


def load_tier3_model():
    """Nạp mô hình Tầng 3 vào bộ nhớ."""
    global TIER3_MODEL, TIER3_CAM, CLASS_MAPPING
    model_path = MODEL_PATH
    mapping_path = CLASS_MAPPING_PATH

    if mapping_path.exists():
        mapping = read_json(mapping_path)
        CLASS_MAPPING = {int(k): v for k, v in mapping["idx_to_class"].items()}

    if model_path.exists():
        try:
            print(f"-> Đang nạp mô hình Tầng 3 từ {model_path} lên {DEVICE}...")
            TIER3_MODEL = load_trained_resnet50(model_path, DEVICE)
            TIER3_CAM = GradCAM(TIER3_MODEL)
            # Chạy thử một lần để lần dự đoán đầu tiên không bị chậm do khởi tạo CUDA/cuDNN
            with torch.no_grad():
                TIER3_MODEL(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE))
            print("-> Tầng 3 đã sẵn sàng phục vụ!")
        except Exception as e:
            print(f"Lỗi nạp mô hình Tầng 3: {e}")
            TIER3_MODEL = None
            TIER3_CAM = None

# Transform ảnh cho inference
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def to_data_url(img, fmt="JPEG"):
    """PIL Image hoặc mảng numpy RGB -> data URL base64 để hiển thị trực tiếp trên trình duyệt."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def class_info(raw_class):
    return CLASS_INFO.get(raw_class, {
        "vi": raw_class.replace("___", " - ").replace("_", " "),
        "status": "disease" if "healthy" not in raw_class.lower() else "healthy",
        "treatment": "Cần theo dõi sát sao biểu hiện của lá cây."
    })


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    """Kiểm tra trạng thái các mô hình."""
    tier1_ready = (MODELS_DIR / "tier1_svm.pkl").exists()
    tier2_ready = (MODELS_DIR / "tier2_svm.pkl").exists()
    tier3_ready = MODEL_PATH.exists()

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Mode"
    metrics = read_json(RESULTS_DIR / "tier3_metrics.json", {})

    return jsonify({
        "gpu": {
            "available": gpu_available,
            "device": gpu_name
        },
        "models": {
            "tier1": {"name": "Tầng 1: Handcrafted + SVM", "ready": tier1_ready},
            "tier2": {"name": "Tầng 2: Deep Features + SVM", "ready": tier2_ready},
            "tier3": {"name": "Tầng 3: ResNet50 Fine-tuned", "ready": tier3_ready}
        },
        "tier3_test_accuracy": metrics.get("accuracy"),
        "tier3_test_macro_f1": metrics.get("macro_f1"),
        "threshold": get_threshold(),
        "filters": {k: v[0] for k, v in CORRUPTIONS.items()}
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """API dự đoán bệnh cây trồng. Form fields: image (file), filter (tùy chọn, xem src/corruptions.py)."""
    global TIER3_MODEL, CLASS_MAPPING

    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh upload"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    filter_key = request.form.get("filter", "none")
    if filter_key not in CORRUPTIONS:
        return jsonify({"error": f"Bộ lọc không hợp lệ: {filter_key}"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Không thể đọc file ảnh: {str(e)}"}), 400

    # Áp dụng bộ lọc trước khi đưa vào mô hình (giống hệt scripts/robustness_tier3.py)
    image = CORRUPTIONS[filter_key][1](image)

    results = {}

    # Chạy Tầng 3 nếu có
    if TIER3_MODEL is None:
        load_tier3_model()

    if TIER3_MODEL is not None and CLASS_MAPPING is not None:
        img_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        with INFER_LOCK:
            t0 = time.time()
            with torch.no_grad():
                outputs = TIER3_MODEL(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.time() - t0) * 1000

            top1_idx = int(np.argmax(probs))
            gradcam_url = None
            try:
                cam, _ = TIER3_CAM.generate_heatmap(img_tensor, target_class=top1_idx)
                gradcam_url = to_data_url(overlay_heatmap(image, cam))
            except Exception as e:
                print(f"Lỗi Grad-CAM: {e}")

        # Lấy Top 5 dự đoán
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = []
        for idx in top5_indices:
            raw_class = CLASS_MAPPING.get(int(idx), f"Class_{idx}")
            info = class_info(raw_class)
            top5.append({
                "raw_class": raw_class,
                "display_name": info["vi"],
                "status": info["status"],
                "confidence": round(float(probs[idx]) * 100, 2),
                "treatment": info["treatment"]
            })

        best = top5[0]
        threshold = get_threshold()
        results["tier3"] = {
            "available": True,
            "latency_ms": round(latency_ms, 1),
            "prediction": best["display_name"],
            "raw_class": best["raw_class"],
            "status": best["status"],
            "confidence": best["confidence"],
            "treatment": best["treatment"],
            "top5": top5,
            "uncertain": best["confidence"] < threshold * 100,
            "threshold": round(threshold * 100, 1),
            "filter": filter_key,
            "filter_name": CORRUPTIONS[filter_key][0],
            "input_image": to_data_url(image.resize((IMAGE_SIZE, IMAGE_SIZE))),
            "gradcam": gradcam_url
        }
    else:
        results["tier3"] = {
            "available": False,
            "message": "Chưa tìm thấy models/tier3_resnet50.pth. Hãy chạy scripts/train_tier3.py trước."
        }

    # Tầng 1 & Tầng 2: phương án cơ sở, sẽ bổ sung sau
    results["tier1"] = {
        "available": (MODELS_DIR / "tier1_svm.pkl").exists(),
        "message": "Mô hình Tầng 1 (models/tier1_svm.pkl) chưa được bổ sung."
    }
    results["tier2"] = {
        "available": (MODELS_DIR / "tier2_svm.pkl").exists(),
        "message": "Mô hình Tầng 2 (models/tier2_svm.pkl) chưa được bổ sung."
    }

    return jsonify(results)

if __name__ == "__main__":
    load_tier3_model()
    print("🌿 Khởi chạy Web Demo tại: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
