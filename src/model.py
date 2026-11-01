import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from PIL import Image
from src.config import NUM_CLASSES

def build_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    """
    Khởi tạo mô hình ResNet50 với Classifier Head mới 38 lớp.
    - freeze_backbone=True: Đóng băng feature extractor cho Giai đoạn 1.
    """
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)
        
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Thay thế Fully Connected (FC) layer bằng Classifier Head 2 tầng
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    return model

def load_trained_resnet50(model_path, device, num_classes=NUM_CLASSES):
    """Nạp checkpoint do scripts/train_tier3.py lưu ra, đặt ở chế độ eval."""
    model = build_resnet50(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()

def unfreeze_layer4(model):
    """Mở unfreeze layer4 (block sâu nhất của ResNet50) cho Giai đoạn 2 (Fine-tuning nhẹ)."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    print("-> Đã unfreeze layer4 để Fine-tuning!")
    return model

class GradCAM:
    """Kỹ thuật Explainable AI (XAI) Grad-CAM để trực quan hóa vùng lá bị bệnh mô hình quan sát."""
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.gradients = None
        self.activations = None
        
        # Hook lưu activations và gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        input_tensor: Tensor kích thước (1, 3, 224, 224)
        Trả về: heatmap numpy array (224, 224)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Tính trọng số alpha dựa trên average pooling của gradient
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Tích chập tuyến tính giữa weights và activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        
        # Áp dụng ReLU (chỉ giữ lại những nét ảnh hưởng tích cực)
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)
            
        # Resize về kích thước ảnh gốc
        cam_resized = cv2.resize(cam, (224, 224))
        return cam_resized, target_class

def overlay_heatmap(pil_img, cam, alpha=0.45, size=224):
    """Chồng heatmap Grad-CAM (giá trị 0..1) lên ảnh gốc. Trả về mảng RGB uint8 (size, size, 3)."""
    base = np.asarray(pil_img.convert("RGB").resize((size, size)), dtype=np.float32)
    cam = cv2.resize(cam, (size, size))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.uint8(np.clip((1 - alpha) * base + alpha * heat, 0, 255))
