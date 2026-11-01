import os
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    SPLITS_CSV, SPLIT_NAME, MODELS_DIR, RESULTS_DIR, MODEL_PATH, NUM_CLASSES,
    BATCH_SIZE, ACCUMULATION_STEPS, USE_AMP,
    LR_HEAD, LR_FINETUNE, EPOCHS_HEAD, EPOCHS_FINETUNE
)
from src.split_data import create_dataset_splits
from src.dataset import get_data_loaders
from src.model import build_resnet50, unfreeze_layer4

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed Precision forward pass
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / accumulation_steps
            
        # Backward pass với GradScaler
        if device.type == "cuda" and use_amp:
            scaler.scale(loss_scaled).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_scaled.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
        # Update progress bar
        current_acc = (correct / total) * 100
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.1f}%"})
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def main():
    print("=" * 65)
    print("🌿 TẦNG 3: HUẤN LUYỆN FINE-TUNING RESNET50 (PLANTVILLAGE 38 LỚP)")
    print("=" * 65)
    torch.manual_seed(42)   # cố định thứ tự trộn batch và tăng cường dữ liệu (cuDNN vẫn có thể dao động nhẹ)
    
    # Thiết lập thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Thiết bị đang dùng: {device.type.upper()}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   * GPU Model: {gpu_name} ({vram_total:.1f} GB VRAM)")
        print(f"   * Mixed Precision FP16: BẬT (Tiết kiệm VRAM & tăng tốc)")
        print(f"   * Gradient Accumulation Steps: {ACCUMULATION_STEPS} (Effective Batch = {BATCH_SIZE * ACCUMULATION_STEPS})")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("   * Cảnh báo: Đang chạy trên CPU. Huấn luyện sẽ lâu hơn GPU.")
        
    # Đảm bảo dữ liệu đã được chia
    if not SPLITS_CSV.exists():
        if SPLIT_NAME:
            raise FileNotFoundError(f"Chưa có {SPLITS_CSV}. Hãy tạo bằng scripts/import_external_split.py trước.")
        print("-> Chưa có file chia dữ liệu. Bắt đầu phân chia Stratified...")
        create_dataset_splits()
    print(f"-> File chia tập: {SPLITS_CSV.name}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
    # Tạo DataLoaders
    print("-> Đang khởi tạo DataLoader...")
    train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)
    print(f"   * Số batch Train: {len(train_loader)} (Batch size: {BATCH_SIZE})")
    print(f"   * Số batch Val:   {len(val_loader)}")
    
    # Khởi tạo mô hình
    print("-> Đang tải pretrained ResNet50 và gắn Classifier Head mới (38 lớp)...")
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # Loss function với Label Smoothing nhẹ để chống overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))
    
    # Lưu vết lịch sử huấn luyện
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    
    best_val_acc = 0.0
    best_model_path = MODEL_PATH
    
    start_total_time = time.time()
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: Train Classifier Head (Backbone Frozen)
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"📌 GIAI ĐOẠN 1: Huấn luyện Classifier Head ({EPOCHS_HEAD} Epochs)")
    print("   (Chỉ cập nhật lớp phân loại cuối, giữ nguyên backbone)")
    print("-" * 65)
    
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    
    for epoch in range(1, EPOCHS_HEAD + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ACCUMULATION_STEPS, USE_AMP
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, USE_AMP)
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS_HEAD:02d} [{elapsed:.0f}s] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.1e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES
            }, best_model_path)
            print(f"   ⭐ Đã lưu model tốt nhất mới! (Val Acc: {val_acc:.2f}%)")
            
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: Fine-Tuning Layer 4 (Mở khóa block sâu nhất)
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"📌 GIAI ĐOẠN 2: Fine-Tuning Layer 4 ({EPOCHS_FINETUNE} Epochs)")
    print("   (Mở khóa Layer 4 để tinh chỉnh đặc trưng sâu, lr nhỏ)")
    print("-" * 65)
    
    # Load lại best model từ Giai đoạn 1 trước khi fine-tune
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    unfreeze_layer4(model)
    
    # Thiết lập learning rate khác biệt: layer4 học rất chậm, fc học vừa phải
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": LR_FINETUNE * 0.1},
        {"params": model.fc.parameters(), "lr": LR_FINETUNE}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE, eta_min=1e-6)
    
    for epoch in range(1, EPOCHS_FINETUNE + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ACCUMULATION_STEPS, USE_AMP
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, USE_AMP)
        scheduler.step()
        
        current_lr = optimizer.param_groups[1]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        elapsed = time.time() - t0
        total_epoch_num = EPOCHS_HEAD + epoch
        print(f"Epoch {total_epoch_num:02d}/{(EPOCHS_HEAD + EPOCHS_FINETUNE):02d} [{elapsed:.0f}s] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.1e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": total_epoch_num,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES
            }, best_model_path)
            print(f"   ⭐ Đã lưu model tốt nhất mới! (Val Acc: {val_acc:.2f}%)")
            
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
    total_time = (time.time() - start_total_time) / 60
    print("\n" + "=" * 65)
    print(f"🎉 HUẤN LUYỆN HOÀN TẤT TRONG {total_time:.1f} PHÚT!")
    print(f"   * Độ chính xác Validation cao nhất: {best_val_acc:.2f}%")
    print(f"   * File mô hình: {best_model_path}")
    
    # Lưu lịch sử huấn luyện
    history_path = RESULTS_DIR / "tier3_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"   * Lịch sử train đã lưu: {history_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
