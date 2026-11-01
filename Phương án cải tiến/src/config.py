import os
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "archive" / "plantvillage dataset" / "color"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORRUPT_CSV = DATA_DIR / "corrupt_images.csv"   # do scripts/check_images.py tạo ra

# Cách chia tập. Mặc định: chia ngẫu nhiên phân tầng theo ảnh (data/dataset_splits.csv).
# Đặt biến môi trường PV_SPLIT=<tên> để dùng data/dataset_splits_<tên>.csv (vd. cách chia theo nhóm ảnh gần trùng
# do scripts/import_external_split.py tạo ra). Khi đó mô hình và kết quả lưu riêng, không ghi đè bản mặc định.
SPLIT_NAME = os.environ.get("PV_SPLIT", "").strip()
SPLITS_CSV = DATA_DIR / (f"dataset_splits_{SPLIT_NAME}.csv" if SPLIT_NAME else "dataset_splits.csv")

# Thư mục models và results
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / SPLIT_NAME if SPLIT_NAME else BASE_DIR / "results"
MODEL_PATH = MODELS_DIR / (f"tier3_resnet50_{SPLIT_NAME}.pth" if SPLIT_NAME else "tier3_resnet50.pth")
CLASS_MAPPING_PATH = MODELS_DIR / "tier3_class_mapping.json"   # 38 lớp sắp theo tên, dùng chung cho mọi cách chia

# Cấu hình huấn luyện (Tối ưu cho GPU GTX 1060 3GB VRAM)
NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 16          # 16 để tránh tràn VRAM 3GB
ACCUMULATION_STEPS = 2   # Tích lũy 2 batches -> Effective Batch Size = 32
USE_AMP = True           # FP16 Mixed Precision giúp giảm 50% bộ nhớ activations
NUM_WORKERS = 2          # Phù hợp Windows CPU đa luồng

# Learning rates & Epochs
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
EPOCHS_HEAD = 8          # Giai đoạn 1: Chỉ train Classifier Head
EPOCHS_FINETUNE = 4      # Giai đoạn 2: Fine-tune Layer 4

# Chuẩn hóa ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Thông tin 38 lớp: Tên tiếng Việt và Biện pháp điều trị
CLASS_INFO = {
    "Apple___Apple_scab": {
        "vi": "Táo: Bệnh vảy (Apple Scab)",
        "status": "disease",
        "treatment": "Cắt tỉa cành thông thoáng, dọn sạch lá rụng. Phun thuốc gốc đồng hoặc Difenoconazole / Mancozeb khi búp mới nở."
    },
    "Apple___Black_rot": {
        "vi": "Táo: Bệnh thối đen (Black Rot)",
        "status": "disease",
        "treatment": "Cắt bỏ cành khô và quả bị bệnh héo đen đem tiêu hủy. Phun thuốc trừ nấm Captan hoặc Thiophanate-methyl."
    },
    "Apple___Cedar_apple_rust": {
        "vi": "Táo: Bệnh rỉ sắt tuyết tùng (Cedar Apple Rust)",
        "status": "disease",
        "treatment": "Loại bỏ cây bách xù lân cận trong bán kính 1-2km nếu có thể. Phun Myclobutanil hoặc Mancozeb vào đầu mùa xuân."
    },
    "Apple___healthy": {
        "vi": "Táo: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây phát triển tốt, không phát hiện bệnh. Tiếp tục duy trì tưới nước và bón phân cân đối N-P-K."
    },
    "Blueberry___healthy": {
        "vi": "Việt quất: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Lá bóng mượt, phát triển bình thường. Duy trì độ pH đất chua thích hợp (4.5 - 5.5)."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "vi": "Anh đào: Bệnh phấn trắng (Powdery Mildew)",
        "status": "disease",
        "treatment": "Phun lưu huỳnh dạng bột thấm nước (Wettable Sulfur) hoặc thuốc nhóm Triazole khi xuất hiện lớp phấn trắng đầu tiên."
    },
    "Cherry_(including_sour)___healthy": {
        "vi": "Anh đào: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây khỏe mạnh. Đảm bảo ánh sáng mặt trời đầy đủ và bón lót phân hữu cơ hoai mục định kỳ."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "vi": "Ngô: Bệnh đốm xám (Gray Leaf Spot)",
        "status": "disease",
        "treatment": "Luân canh cây trồng (không trồng ngô liên tiếp). Phun thuốc chứa Azoxystrobin hoặc Pyraclostrobin khi chớm bệnh."
    },
    "Corn_(maize)___Common_rust_": {
        "vi": "Ngô: Bệnh rỉ sắt thông thường (Common Rust)",
        "status": "disease",
        "treatment": "Chọn giống ngô kháng bệnh. Phun thuốc trừ nấm Mancozeb hoặc Propiconazole khi có nhiều ổ bào tử màu nâu đỏ."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "vi": "Ngô: Bệnh cháy lá phương Bắc (Northern Leaf Blight)",
        "status": "disease",
        "treatment": "Cày vùi tàn dư rơm rạ sau vụ thu hoạch. Phun thuốc bảo vệ thực vật hoạt chất Azoxystrobin + Difenoconazole."
    },
    "Corn_(maize)___healthy": {
        "vi": "Ngô: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Ngô sinh trưởng mạnh mẽ, lá xanh đậm. Tiếp tục kiểm soát độ ẩm và làm cỏ định kỳ."
    },
    "Grape___Black_rot": {
        "vi": "Nho: Bệnh thối đen (Black Rot)",
        "status": "disease",
        "treatment": "Cắt tỉa quả thối, giữ giàn nho thông thoáng. Phun Mancozeb, Ziram hoặc Myclobutanil từ giai đoạn đâm chồi đến sau đậu quả."
    },
    "Grape___Esca_(Black_Measles)": {
        "vi": "Nho: Bệnh sởi đen / Esca (Black Measles)",
        "status": "disease",
        "treatment": "Khử trùng dụng cụ tỉa cành. Cắt bỏ thân/cành bị hoại tử, quét sơn bảo vệ vết cắt. Tránh tưới nước ứ đọng gốc."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "vi": "Nho: Bệnh đốm cháy lá (Leaf Blight)",
        "status": "disease",
        "treatment": "Dọn dẹp lá rụng dưới gốc. Phun thuốc gốc đồng Copper Hydroxide hoặc thuốc trừ nấm Phosetyl-Al."
    },
    "Grape___healthy": {
        "vi": "Nho: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Lá nho xanh tươi, quang hợp tốt. Định kỳ buộc cành, tỉa chồi nách để đón ánh sáng tối đa."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "vi": "Cam: Bệnh vàng lá gân xanh (Huanglongbing / Greening)",
        "status": "disease",
        "treatment": "Bệnh do rầy chổng cánh truyền vi khuẩn. Phun trừ rầy triệt để, chặt bỏ cây bệnh nặng để tránh lây lan, bổ sung vi lượng kẽm-sắt."
    },
    "Peach___Bacterial_spot": {
        "vi": "Đào: Bệnh đốm vi khuẩn (Bacterial Spot)",
        "status": "disease",
        "treatment": "Phun Oxytetracycline hoặc hợp chất đồng liều lượng nhẹ trong giai đoạn ngủ đông và khi mới đâm lộc."
    },
    "Peach___healthy": {
        "vi": "Đào: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây phát triển tốt. Chú ý thoát nước tốt quanh gốc trong mùa mưa bão."
    },
    "Pepper,_bell___Bacterial_spot": {
        "vi": "Ớt chuông: Bệnh đốm vi khuẩn (Bacterial Spot)",
        "status": "disease",
        "treatment": "Sử dụng hạt giống sạch bệnh. Phun phối hợp thuốc gốc đồng + Mancozeb để ức chế vi khuẩn lây lan."
    },
    "Pepper,_bell___healthy": {
        "vi": "Ớt chuông: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây ớt khỏe, lá xanh bóng, sẵn sàng ra hoa kết trái. Cung cấp đầy đủ Canxi để tránh thối đít trái."
    },
    "Potato___Early_blight": {
        "vi": "Khoai tây: Bệnh cháy sớm (Early Blight)",
        "status": "disease",
        "treatment": "Tạo khoảng cách cây thông thoáng. Phun thuốc trừ nấm Chlorothalonil hoặc Mancozeb định kỳ 7-10 ngày khi trời ẩm."
    },
    "Potato___Late_blight": {
        "vi": "Khoai tây: Bệnh mốc sương / Cháy muộn (Late Blight)",
        "status": "disease",
        "treatment": "Bệnh lây lan cực nhanh do Phytophthora. Phun phòng trừ ngay bằng Metalaxyl, Dimethomorph hoặc Cymoxanil khi trời rét ẩm."
    },
    "Potato___healthy": {
        "vi": "Khoai tây: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Tán lá dày, củ phát triển tốt. Vun xới gốc và tránh để củ tiếp xúc trực tiếp với ánh nắng."
    },
    "Raspberry___healthy": {
        "vi": "Mâm xôi: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây sinh trưởng bình thường. Giữ ẩm gốc bằng rơm rạ hoặc màng phủ nông nghiệp."
    },
    "Soybean___healthy": {
        "vi": "Đậu nành: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Lá xanh đều, rễ nốt sần cố định đạm tốt. Duy trì mật độ gieo trồng hợp lý."
    },
    "Squash___Powdery_mildew": {
        "vi": "Bí ngô: Bệnh phấn trắng (Powdery Mildew)",
        "status": "disease",
        "treatment": "Tưới nước sát gốc (tránh ướt mặt lá). Phun chế phẩm sinh học dầu Neem hoặc thuốc trừ nấm gốc Azoxystrobin."
    },
    "Strawberry___Leaf_scorch": {
        "vi": "Dâu tây: Bệnh cháy lá đốm đỏ (Leaf Scorch)",
        "status": "disease",
        "treatment": "Cắt bỏ toàn bộ lá già, lá bị đốm tím. Phun thuốc ngừa nấm Captan hoặc Thiophanate-methyl sau khi thu hoạch."
    },
    "Strawberry___healthy": {
        "vi": "Dâu tây: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Lá xanh hình quạt cân đối, hoa quả phát triển đẹp. Giữ quả khô ráo trên màng nilong phủ."
    },
    "Tomato___Bacterial_spot": {
        "vi": "Cà chua: Bệnh đốm vi khuẩn (Bacterial Spot)",
        "status": "disease",
        "treatment": "Tránh tưới phun mưa lên tán lá. Phun luân phiên thuốc gốc đồng kết hợp Kasugamycin hoặc Streptomycin."
    },
    "Tomato___Early_blight": {
        "vi": "Cà chua: Bệnh đốm vòng / Cháy sớm (Early Blight)",
        "status": "disease",
        "treatment": "Đặc trưng bởi các đốm nâu có vòng đồng tâm. Cắt tỉa lá gốc sát mặt đất. Phun Mancozeb, Difenoconazole hoặc Chlorothalonil."
    },
    "Tomato___Late_blight": {
        "vi": "Cà chua: Bệnh sương mai / Cháy muộn (Late Blight)",
        "status": "disease",
        "treatment": "Gây thối nhũn nâu đen trên lá và thân rất nhanh. Phun dập dịch khẩn cấp bằng Ridomil Gold (Metalaxyl + Mancozeb) hoặc Fosetyl-Al."
    },
    "Tomato___Leaf_Mold": {
        "vi": "Cà chua: Bệnh mốc lá (Leaf Mold)",
        "status": "disease",
        "treatment": "Thường gặp trong nhà màng ẩm ướt. Tăng cường thông gió, giảm độ ẩm không khí. Phun thuốc chứa Difenoconazole."
    },
    "Tomato___Septoria_leaf_spot": {
        "vi": "Cà chua: Bệnh đốm lá Septoria (Septoria Leaf Spot)",
        "status": "disease",
        "treatment": "Nhiều đốm nhỏ tròn viền nâu sẫm tâm xám. Cắt bỏ lá bệnh ở tầng dưới, phun thuốc bảo vệ Chlorothalonil."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "vi": "Cà chua: Nhện đỏ hai đốm (Spider Mites)",
        "status": "disease",
        "treatment": "Lá có chấm vàng lấm tấm và màng tơ mỏng mặt dưới. Phun thuốc đặc trị nhện đỏ như Abamectin, Propargite hoặc dầu khoáng."
    },
    "Tomato___Target_Spot": {
        "vi": "Cà chua: Bệnh đốm mắt chim (Target Spot)",
        "status": "disease",
        "treatment": "Đốm hoại tử tròn như bia bắn. Cải thiện lưu thông không khí giữa các luống, phun Azoxystrobin hoặc Boscalid."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "vi": "Cà chua: Virus xoăn vàng lá (TYLCV)",
        "status": "disease",
        "treatment": "Do bọ phấn trắng truyền virus. Không có thuốc trị virus, phải nhổ tiêu hủy cây bệnh và phun trừ bọ phấn (Imidacloprid/Dinotefuran)."
    },
    "Tomato___Tomato_mosaic_virus": {
        "vi": "Cà chua: Virus khảm lá (Tobacco/Tomato Mosaic Virus)",
        "status": "disease",
        "treatment": "Lá biến dạng loang lổ xanh vàng. Lây qua tiếp xúc tay và dụng cụ. Nhổ bỏ cây bệnh, sát trùng tay và kéo tỉa cành bằng xà phòng."
    },
    "Tomato___healthy": {
        "vi": "Cà chua: Khỏe mạnh (Healthy)",
        "status": "healthy",
        "treatment": "Cây sinh trưởng rất tốt, lá xanh đậm, phân nhánh đều. Chú ý tỉa chồi nách và cắm cọc đỡ giàn chắc chắn."
    }
}
