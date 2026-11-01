import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from src.config import (
    SPLITS_CSV, CORRUPT_CSV, DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD
)

def resolve_image_path(rel_path):
    """Chuyển đường dẫn tương đối (so với data/) trong CSV thành đường dẫn tuyệt đối."""
    return str(DATA_DIR / rel_path)

def load_split_df(split=None, drop_corrupt=True):
    """
    Đọc file chia tập; thêm cột 'filepath' tuyệt đối. split=None trả về toàn bộ.
    drop_corrupt=True: bỏ các ảnh hỏng liệt kê trong data/corrupt_images.csv (scripts/check_images.py).
    """
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"Chưa tìm thấy file chia dữ liệu: {SPLITS_CSV}. Hãy chạy src/split_data.py trước.")
    df = pd.read_csv(SPLITS_CSV)
    if split is not None:
        df = df[df["split"] == split]
    if drop_corrupt and CORRUPT_CSV.exists():
        corrupt = set(pd.read_csv(CORRUPT_CSV)["rel_path"])
        df = df[~df["rel_path"].isin(corrupt)]
    df = df.reset_index(drop=True)
    df["filepath"] = df["rel_path"].map(resolve_image_path)
    return df

class PlantVillageDataset(Dataset):
    """Dataset tùy biến đọc trực tiếp từ metadata CSV mà không cần copy di dời ảnh."""
    def __init__(self, df_or_split, transform=None):
        if isinstance(df_or_split, str):
            self.df = load_split_df(df_or_split)
        else:
            self.df = df_or_split.reset_index(drop=True)
            if "filepath" not in self.df.columns:
                self.df["filepath"] = self.df["rel_path"].map(resolve_image_path)

        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["label_idx"].tolist()
        self.classes = self.df["class_name"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Không thay bằng ảnh đen: ảnh đen mang nhãn thật sẽ thành nhiễu nhãn và làm sai kết quả đánh giá
            raise RuntimeError(f"Không đọc được ảnh {path}. Hãy chạy scripts/check_images.py để loại ảnh hỏng.") from e

        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def get_transforms():
    """Tạo bộ biến đổi dữ liệu với Data Augmentation hợp lý cho cây trồng."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transform, val_test_transform

def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True):
    """Khởi tạo DataLoader cho 3 tập Train, Val, Test."""
    train_tf, val_test_tf = get_transforms()
    
    train_ds = PlantVillageDataset("train", transform=train_tf)
    val_ds = PlantVillageDataset("val", transform=val_test_tf)
    test_ds = PlantVillageDataset("test", transform=val_test_tf)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
