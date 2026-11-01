# Nhận dạng bệnh cây trồng qua ảnh lá (PlantVillage, 38 lớp)

Bài tập cuối kỳ môn Nhận dạng mẫu. Đầu vào là ảnh một chiếc lá; đầu ra là 1 trong 38 lớp (14 loài cây, gồm các bệnh và trạng thái khỏe).

| Phương án | Mã nguồn | Kết quả trên tập test 5.668 ảnh |
|---|---|---|
| Cơ sở: 1.977 đặc trưng thủ công + Random Forest 500 cây | `Phương án cơ sở/Nhận_dạng_mẫu/Plant_Disease_Detection.ipynb` | Accuracy 82,07%, macro-F1 0,7931 |
| **Cải tiến: ResNet50 học chuyển giao** | `src/`, `scripts/`, `demo/` | **Accuracy 98,57%, macro-F1 0,9769** |

Hai phương án dùng chung một cách chia tập (theo nhóm ảnh gần trùng) và được chấm trên cùng tập test.
Báo cáo LaTeX ở `Nhận_dạng_bệnh_cây_trồng_qua_ảnh_lá/` (`main.tex`, biên dịch bằng pdfLaTeX hoặc XeLaTeX, ví dụ trên Overleaf).

## Chạy nhanh demo

```bash
pip install -r requirements.txt
python -m demo.app            # trước đó đặt PV_SPLIT=group (PowerShell: $env:PV_SPLIT="group")
```

Mở <http://localhost:5000>. Trên Windows chỉ cần nhấp đúp `run_web.bat` (tự đặt `PV_SPLIT=group`).
Demo cần file trọng số `models/tier3_resnet50_group.pth` (98,6 MB). File này không có trong repo: chạy bước huấn luyện
bên dưới, hoặc lấy từ gói dự án của nhóm rồi đặt vào `models/`.

**Repo này chỉ gồm mã nguồn và file chia tập.** Không gồm dữ liệu ảnh, trọng số mô hình, thư mục `results/` và báo cáo
LaTeX; các thư mục đó được tạo ra khi chạy các script dưới đây.

## Dữ liệu

- **PlantVillage** (54.305 ảnh, 3 biến thể `color`, `grayscale`, `segmented`):
  <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset>
  (bản gốc: <https://github.com/spMohanty/PlantVillage-Dataset>; Hughes & Salathé, arXiv:1511.08060, 2015).
  Giải nén sao cho có `data/raw/archive/plantvillage dataset/color/<tên lớp>/*.JPG`.
  ResNet50 dùng bản `color`; Random Forest dùng `segmented` (đặc trưng màu) và `grayscale` (HOG, Canny).
- `data/dataset_splits_group.csv`: **cách chia chính**, theo nhóm ảnh gần trùng của notebook Random Forest.
  pHash 64 bit, nối các ảnh cùng lớp có khoảng cách Hamming ≤ 4 (53.393 nhóm), rồi `StratifiedGroupKFold`
  80/10/10. Kết quả: train 43.238 / val 5.399 / test 5.668 ảnh; không nhóm nào nằm ở hai tập.
- `data/rf_groups.csv`: nhóm của từng ảnh theo đúng thứ tự notebook (dùng cho `reproduce_rf_split.py --from-groups`).
- `data/dataset_splits.csv`: cách chia ngẫu nhiên phân tầng 80/10/10, seed 42 (43.444 / 5.430 / 5.431 ảnh), dùng trước khi
  thống nhất với phương án cơ sở. Chỉ còn dùng để đối chiếu (mục "Ảnh hưởng của cách chia tập").
- `data/corrupt_images.csv`: danh sách ảnh không đọc được, do `scripts/check_images.py` tạo ra; hiện **rỗng**.
  Lần giải nén đầu từng có 352 file toàn byte 0; nhóm đã tải lại từ Kaggle và thay đủ. Nên giải nén bằng 7-Zip rồi chạy
  `scripts/check_images.py`.
- `demo/static/samples/`: 16 ảnh mẫu của demo, đều thuộc **tập test của cách chia theo nhóm** (nguồn ghi trong `SOURCES.csv`).

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Nếu có GPU NVIDIA, cài torch/torchvision bản CUDA trước theo <https://pytorch.org/get-started/locally/>
(nhóm dùng `torch 2.6.0+cu124`). Không có GPU vẫn chạy được, chỉ chậm hơn.

## Tái lập kết quả

Biến môi trường `PV_SPLIT` chọn cách chia: `PV_SPLIT=group` dùng `data/dataset_splits_group.csv`, lưu mô hình vào
`models/tier3_resnet50_group.pth` và kết quả vào `results/group/`. Không đặt biến thì dùng cách chia ngẫu nhiên và `results/`.
Chạy các lệnh từ thư mục gốc (PowerShell: `$env:PV_SPLIT="group"` thay cho `export`). Thời gian đo trên GTX 1060 3 GB.

| Bước | Lệnh | Đầu ra chính | Thời gian |
|---|---|---|---|
| 1. Dò ảnh hỏng | `python scripts/check_images.py` | `data/corrupt_images.csv` | ~2 phút |
| 2. Tái tạo cách chia theo nhóm | `python scripts/reproduce_rf_split.py --variants-root "<thư mục chứa color/, grayscale/>"` | `data/rf_groups.csv`, `data/dataset_splits_group.csv` | ~5 phút |
| 3. Phân bố lớp, ảnh mẫu | `PV_SPLIT=group python scripts/01_eda_and_split.py` | `results/group/eda_*` | ~1 phút |
| 4. Thống kê màu, độ sáng | `python scripts/eda_stats.py` | `results/eda/` | ~3 phút |
| 5. Huấn luyện | `PV_SPLIT=group python scripts/train_tier3.py` | `models/tier3_resnet50_group.pth` | ~80 phút |
| 6. Đánh giá test | `PV_SPLIT=group python scripts/evaluate_tier3.py` | `results/group/tier3_*` | ~1 phút |
| 7. Phân tích lỗi | `PV_SPLIT=group python scripts/analyze_errors_tier3.py` (cần bước 6) | `results/group/tier3_analysis/` | < 1 phút |
| 8. Độ bền | `PV_SPLIT=group python scripts/robustness_tier3.py` | `results/group/tier3_robustness.*` | ~5 phút |
| 9. Thiên lệch nền | `PV_SPLIT=group python scripts/background_bias_test.py` | `results/group/background_bias.json` | ~3 phút |
| 10. So sánh với Random Forest | `python scripts/compare_with_notebook.py` (cần bước 6) | `results/comparison/group/` | ~2 phút |

**Lưu ý bước 2:** kết quả chia phụ thuộc phiên bản scikit-learn. Notebook chạy scikit-learn 1.6.1; bản mới hơn (vd. 1.9.1)
chia ra khác. Bước gom nhóm chạy được trong môi trường của dự án; bước chia cần môi trường riêng có
`scikit-learn==1.6.1 numpy==2.1.3 pandas` rồi chạy `python scripts/reproduce_rf_split.py --from-groups`.
Script đối chiếu với các số notebook in ra và **không ghi file nếu lệch**. File `data/dataset_splits_group.csv` đã có sẵn,
nên thường không cần chạy lại bước này.

**Cấu hình huấn luyện ResNet50** (`src/config.py`, `scripts/train_tier3.py`): ResNet50 pretrained ImageNet, thay lớp FC bằng
`Dropout(0,4) → Linear(2048,512) → ReLU → BN → Dropout(0,2) → Linear(512,38)` (tổng 24,6 triệu tham số).
- Giai đoạn 1 (8 epoch): đóng băng backbone, chỉ học phần phân loại; AdamW lr 1e-3, ReduceLROnPlateau.
- Giai đoạn 2 (4 epoch): mở `layer4` (lr 1e-5) và phần phân loại (lr 1e-4), lịch học cosine.
- Ảnh 224×224, lật, xoay ±15°, ColorJitter 0,15; label smoothing 0,05; batch hiệu dụng 32 (16 × tích lũy 2), FP16, seed 42.
  Giữ checkpoint có val accuracy cao nhất (98,48% ở epoch 12).

**Cấu hình Random Forest** (notebook, mục 6): 500 cây, `max_depth=20`, `min_samples_split=8`, `min_samples_leaf=2`,
`max_features=0.2`, `criterion="entropy"`, `class_weight="balanced_subsample"`, `ccp_alpha=0.001`; chọn qua tìm kiếm 3 tầng
với 5 fold theo nhóm (macro-F1 CV 0,7961 ± 0,0062), học lại trên train + val rồi chấm một lần trên test.

## Kết quả (cách chia theo nhóm, tập test 5.668 ảnh)

| Chỉ số | Random Forest | ResNet50 (bootstrap 95%) |
|---|---|---|
| Accuracy | 0,8207 (Wilson 95%: 0,8105–0,8305) | **0,9857** (0,9827–0,9887) |
| Balanced accuracy | 0,8249 | 0,9764 (0,9693–0,9825) |
| Macro-F1 | 0,7931 | **0,9769** (0,9704–0,9821) |
| Weighted-F1 | 0,8216 | 0,9856 (0,9827–0,9887) |
| Macro-precision / macro-recall | 0,77 / 0,82 | 0,9777 / 0,9764 |
| Số lớp có F1 < 0,80 | 16 | 0 |

- ResNet50 có F1 cao hơn ở 37/38 lớp; lớp còn lại (Ngô rỉ sắt) hai mô hình ngang nhau (0,996 so với 1,00 đã làm tròn).
- Lớp yếu nhất của Random Forest là Cà chua cháy sớm (F1 0,45); ResNet50 đạt 0,910 ở lớp này.
- Số liệu Random Forest lấy từ output notebook (mục 6.5). Notebook không lưu dự đoán từng ảnh nên không làm được kiểm định
  McNemar; thay vào đó so khoảng tin cậy của hai mô hình (không chồng lên nhau).
- Chi tiết: `results/comparison/group/summary.md` (tạo bởi `scripts/compare_with_notebook.py`).

**ResNet50, thêm:**

| Chỉ số | Giá trị |
|---|---|
| Top-3 / Top-5 accuracy | 99,89% / 99,96% |
| Số ảnh sai | 81, trong đó 62 (77%) sai bệnh nhưng đúng loài cây |
| Cặp hay nhầm nhất | Ngô đốm xám ↔ Ngô cháy lá phương Bắc (15); Cà chua sương mai → Khoai tây mốc sương (5) |
| Độ tin cậy trung vị (ảnh đúng / ảnh sai) | 97,1% / 56,5% |
| Ngưỡng "không chắc chắn" 0,65 | Trả lời 96,5% số ảnh với accuracy 99,54%; còn 5 lỗi có độ tin cậy ≥ 0,90 |
| Tốc độ suy luận (GTX 1060, batch 32, FP16) | 6,12 ms/ảnh |

**Độ bền** của ResNet50 khi biến đổi ảnh test:

| Ảnh gốc | Giảm sáng 50% | Ảnh xám | Nhiễu σ=25 | Làm mờ r=2 | Độ phân giải 64 px |
|---|---|---|---|---|---|
| 98,57% | 98,01% | 84,46% | 76,22% | 70,54% | 65,91% |

**Thiên lệch nền:** một Random Forest chỉ nhìn màu của 4 góc ảnh 16×16 px (gần như toàn là nền) đạt **65,46%** trên tập test,
trong khi đoán ngẫu nhiên chỉ được 2,63% và luôn đoán lớp đông nhất được 9,84%. `eda_stats.py` cũng cho thấy độ sáng nền khác
nhau rõ giữa các lớp (Kruskal–Wallis, ε² = 0,57; trung vị từ 18,6 ở Ngô rỉ sắt đến 177,6 ở Đậu nành khỏe).

**Ảnh hưởng của cách chia tập:** trên cách chia ngẫu nhiên (`results/`, 5.431 ảnh test), mô hình huấn luyện trước đó đạt
98,40% / macro-F1 0,976. 323/5.431 ảnh test có ảnh gần trùng trong train (`scripts/near_duplicate_check.py`), và rò rỉ này chỉ
làm accuracy tăng khoảng 0,1 điểm %. Mô hình đó không có trong gói gửi nhóm; mọi số liệu báo cáo dùng cách chia theo nhóm.

## Demo web

- Tải ảnh lên (hoặc kéo thả), hoặc chọn 1 trong 16 ảnh mẫu của tập test.
- Xem top-5 dự đoán, tên bệnh tiếng Việt và gợi ý xử lý.
- **Grad-CAM**: xem mô hình chú ý vào vùng nào của lá.
- **Bộ lọc**: ảnh xám, làm mờ, giảm sáng, độ phân giải 64 px, nhiễu. Đổi bộ lọc thì kết quả tự cập nhật.
- **Cảnh báo "không chắc chắn"** khi độ tin cậy thấp hơn ngưỡng trong `results/group/tier3_confidence_threshold.json` (0,65).
  Ảnh mẫu "Khoai tây mốc sương" có độ tin cậy 0,54 nên minh họa được cảnh báo này.

## Cấu trúc thư mục

```
src/       config.py (cấu hình, PV_SPLIT), dataset.py, model.py (ResNet50, Grad-CAM), split_data.py, corruptions.py
scripts/   check_images.py, reproduce_rf_split.py, 01_eda_and_split.py, eda_stats.py, train_tier3.py,
           evaluate_tier3.py, analyze_errors_tier3.py, robustness_tier3.py, background_bias_test.py,
           compare_with_notebook.py, compare_models.py, near_duplicate_check.py, import_external_split.py
demo/      app.py (Flask), templates/, static/
models/    tier3_resnet50_group.pth, tier3_class_mapping.json
results/   group/ (kết quả ResNet50 trên cách chia theo nhóm), eda/ (thống kê ảnh), comparison/group/ (so sánh),
           các file ở gốc: kết quả trên cách chia ngẫu nhiên
data/      dataset_splits_group.csv, rf_groups.csv, dataset_splits.csv, corrupt_images.csv, raw/ (tự tải về)
Phương án cơ sở/Nhận_dạng_mẫu/Plant_Disease_Detection.ipynb   notebook phương án cơ sở (Random Forest)
Nhận_dạng_bệnh_cây_trồng_qua_ảnh_lá/   báo cáo LaTeX: main.tex, bia.tex (trang bìa), figures/
```

`compare_models.py` dùng khi có file dự đoán từng ảnh của cả hai mô hình (kèm kiểm định McNemar);
`compare_with_notebook.py` dùng số liệu notebook in ra. Báo cáo dùng hình từ `results/` và từ output notebook; nếu chạy lại
các script, hãy copy hình mới vào `figures/` của báo cáo.

## Hạn chế

- Ảnh chụp trong phòng thí nghiệm (một lá, nền đồng màu), và nền mang thông tin về nhãn; kết quả chưa phản ánh ảnh ngoài đồng.
- ResNet50 giảm mạnh với ảnh mờ, nhiễu, độ phân giải thấp vì các biến đổi này không có trong tăng cường dữ liệu.
- Chỉ huấn luyện một lần (seed 42), nên khoảng tin cậy chỉ phản ánh dao động của tập test, chưa phản ánh dao động giữa các lần chạy.
- pHash ngưỡng 4 chỉ gom được 2,43% số ảnh; các ảnh cùng một chiếc lá chụp ở góc khác vẫn có thể nằm ở hai tập.
- Ngưỡng "không chắc chắn" được chọn trên tập test nên con số sau khi lọc hơi lạc quan, và không bắt được lỗi tự tin.
- Ảnh không phải lá, hoặc lá của loài ngoài 14 loài, vẫn bị gán vào một trong 38 lớp.

## Tài liệu và mã nguồn tham khảo

- PyTorch / torchvision (ResNet50 pretrained): <https://pytorch.org/vision/stable/models.html>; scikit-learn, OpenCV, scikit-image.
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- L. Breiman, *Random Forests*, Machine Learning 45(1), 2001.
- R. R. Selvaraju et al., *Grad-CAM*, ICCV 2017.
- S. P. Mohanty et al., *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science, 2016.
- M. A. Noyan, *Uncovering bias in the PlantVillage dataset*, arXiv:2206.04374, 2022.
- Tham khảo cách tổ chức pipeline: <https://github.com/CharlesWang185/plantvillage-disease-classifier>,
  <https://github.com/haxybaxy/PlantDiseaseDetection>, <https://github.com/shivaharish1804/Plant-Disease-Detection>.
