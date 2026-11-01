/**
 * AgriVision AI — Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const btnChangeImage = document.getElementById("btnChangeImage");
    const btnPredict = document.getElementById("btnPredict");
    const btnSpinner = document.getElementById("btnSpinner");
    
    // Status Elements
    const gpuStatusPill = document.getElementById("gpuStatusPill");
    const gpuStatusText = document.getElementById("gpuStatusText");
    
    // Results Elements
    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");
    const fallbackNotice = document.getElementById("fallbackNotice");
    const diagnosisBanner = document.getElementById("diagnosisBanner");
    const statusTag = document.getElementById("statusTag");
    const diagnosisName = document.getElementById("diagnosisName");
    const diagnosisRaw = document.getElementById("diagnosisRaw");
    const confidenceVal = document.getElementById("confidenceVal");
    const latencyBadge = document.getElementById("latencyBadge");
    const predictionBars = document.getElementById("predictionBars");
    const treatmentText = document.getElementById("treatmentText");
    const uncertainNotice = document.getElementById("uncertainNotice");
    const uncertainText = document.getElementById("uncertainText");
    const accuracyBadge = document.getElementById("accuracyBadge");
    const filterBadge = document.getElementById("filterBadge");
    const inputImage = document.getElementById("inputImage");
    const gradcamImage = document.getElementById("gradcamImage");

    // Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const filterChips = document.querySelectorAll(".filter-chip");
    let currentTier = "tier3";
    let currentFilter = "none";
    let lastPredictionData = null;
    let selectedFile = null;

    // 1. Kiểm tra trạng thái GPU và Mô hình
    async function checkSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            if (data.gpu && data.gpu.available) {
                gpuStatusText.innerHTML = `🟢 <strong>GPU:</strong> ${data.gpu.device}`;
                gpuStatusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
            } else {
                gpuStatusText.innerHTML = `⚪ <strong>CPU Mode</strong>`;
            }
            if (data.tier3_test_accuracy != null) {
                accuracyBadge.innerHTML = `🎯 Accuracy trên tập test: <strong>${data.tier3_test_accuracy.toFixed(2)}%</strong>`;
            }
        } catch (e) {
            gpuStatusText.innerHTML = `⚠️ Không thể kiểm tra GPU`;
        }
    }
    checkSystemStatus();

    // 2. Xử lý Drag & Drop và File Selection
    dropzone.addEventListener("click", (e) => {
        if (e.target !== btnChangeImage) {
            fileInput.click();
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    btnChangeImage.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFileSelection(file) {
        if (!file.type.startsWith("image/")) {
            alert("Vui lòng chọn file hình ảnh hợp lệ (JPG, PNG)!");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            btnPredict.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        btnPredict.disabled = true;
    }

    // 3. Xử lý Quick Sample Chips (Nạp ảnh lá cây thực tế)
    const sampleChips = document.querySelectorAll(".chip");
    sampleChips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const dataSrc = chip.getAttribute("data-src");
            if (!dataSrc) return;
            
            try {
                const res = await fetch(dataSrc);
                const blob = await res.blob();
                const filename = dataSrc.split("/").pop();
                const realFile = new File([blob], filename, { type: "image/jpeg" });
                handleFileSelection(realFile);
            } catch (err) {
                console.error("Lỗi khi tải ảnh mẫu:", err);
            }
        });
    });

    // 3b. Chọn bộ lọc ảnh; nếu đã có kết quả thì chạy lại ngay để so sánh
    filterChips.forEach(chip => {
        chip.addEventListener("click", () => {
            filterChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            currentFilter = chip.getAttribute("data-filter");
            if (selectedFile && lastPredictionData) {
                runPrediction();
            }
        });
    });

    // 4. Gửi ảnh phân tích (API /predict)
    btnPredict.addEventListener("click", runPrediction);

    async function runPrediction() {
        if (!selectedFile) return;

        // Bật animation quét laser và spinner
        dropzone.classList.add("scanning");
        btnSpinner.classList.remove("hidden");
        btnPredict.disabled = true;

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("filter", currentFilter);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Lỗi khi phân tích ảnh từ máy chủ");
            }
            lastPredictionData = data;
            renderResultsForTier(currentTier);

        } catch (err) {
            alert(`Lỗi: ${err.message}`);
        } finally {
            dropzone.classList.remove("scanning");
            btnSpinner.classList.add("hidden");
            btnPredict.disabled = false;
        }
    }

    // 5. Chuyển Tab giữa các Tầng
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentTier = btn.getAttribute("data-tier");
            if (lastPredictionData) {
                renderResultsForTier(currentTier);
            }
        });
    });

    // 6. Hiển thị kết quả của tầng được chọn
    function renderResultsForTier(tier) {
        if (!lastPredictionData) return;
        
        emptyState.classList.add("hidden");
        const tierData = lastPredictionData[tier];

        if (!tierData || !tierData.available || !tierData.top5) {
            resultsContent.classList.add("hidden");
            fallbackNotice.classList.remove("hidden");
            
            const tierNames = {
                "tier1": "Tầng 1 (ML truyền thống: Handcrafted + SVM)",
                "tier2": "Tầng 2 (Deep Features + SVM)",
                "tier3": "Tầng 3 (ResNet50 Fine-tuned)"
            };
            document.getElementById("fallbackTitle").innerText = `${tierNames[tier]} chưa sẵn sàng`;
            document.getElementById("fallbackDesc").innerText = tierData ? tierData.message : "Chờ nạp file mô hình.";
            return;
        }

        fallbackNotice.classList.add("hidden");
        resultsContent.classList.remove("hidden");

        // Cập nhật Top Banner
        const isDisease = tierData.status === "disease";
        const isUncertain = Boolean(tierData.uncertain);
        diagnosisBanner.className = `diagnosis-banner ${isUncertain ? "uncertain" : (isDisease ? "disease" : "healthy")}`;
        statusTag.innerText = isUncertain ? "⚠️ Không Chắc Chắn"
            : (isDisease ? "🚨 Phát Hiện Bệnh" : "✅ Lá Cây Khỏe Mạnh");
        diagnosisName.innerText = tierData.prediction;
        diagnosisRaw.innerText = tierData.raw_class;
        confidenceVal.innerText = `${tierData.confidence}%`;

        if (isUncertain) {
            uncertainText.innerText = `Độ tin cậy ${tierData.confidence}% thấp hơn ngưỡng ${tierData.threshold}%. ` +
                "Kết quả dễ sai — nên chụp lại ảnh rõ hơn, một lá, đủ sáng, hoặc xem thêm các khả năng trong Top 5.";
            uncertainNotice.classList.remove("hidden");
        } else {
            uncertainNotice.classList.add("hidden");
        }

        if (tierData.latency_ms) {
            latencyBadge.innerHTML = `⚡ Tốc độ suy luận: <strong>${tierData.latency_ms} ms</strong>`;
        }
        if (tierData.filter_name) {
            filterBadge.innerHTML = `🧪 Bộ lọc: <strong>${tierData.filter_name}</strong>`;
        }
        if (tierData.input_image) {
            inputImage.src = tierData.input_image;
        }
        if (tierData.gradcam) {
            gradcamImage.src = tierData.gradcam;
        }

        // Cập nhật Top 5 Bars
        predictionBars.innerHTML = "";
        tierData.top5.forEach((item, index) => {
            const barItem = document.createElement("div");
            barItem.className = "bar-item";
            
            const isTop = index === 0;
            barItem.innerHTML = `
                <div class="bar-meta">
                    <span class="bar-label">${index + 1}. ${item.display_name}</span>
                    <span class="bar-percent">${item.confidence}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill ${isTop ? "" : "secondary"}" style="width: 0%"></div>
                </div>
            `;
            predictionBars.appendChild(barItem);

            // Animate width sau khi render
            setTimeout(() => {
                barItem.querySelector(".bar-fill").style.width = `${item.confidence}%`;
            }, 50 * index);
        });

        // Cập nhật Hướng dẫn điều trị
        treatmentText.innerText = tierData.treatment || "Theo dõi tình trạng phát triển của cây thường xuyên.";
    }
});
