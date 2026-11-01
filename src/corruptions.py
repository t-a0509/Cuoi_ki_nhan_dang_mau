"""Các phép biến đổi ảnh dùng chung cho kiểm tra độ bền (scripts/robustness_tier3.py) và demo web."""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def to_gray(img):
    return img.convert("L").convert("RGB")


def blur(img, radius=2.0):
    return img.filter(ImageFilter.GaussianBlur(radius))


def darken(img, factor=0.5):
    return ImageEnhance.Brightness(img).enhance(factor)


def low_res(img, size=64):
    """Thu nhỏ về size×size rồi phóng lại kích thước cũ (mô phỏng ảnh chụp xa / độ phân giải thấp)."""
    w, h = img.size
    return img.resize((size, size), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def gaussian_noise(img, sigma=25.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# key -> (tên hiển thị, hàm biến đổi)
CORRUPTIONS = {
    "none": ("Ảnh gốc", lambda img: img),
    "gray": ("Ảnh xám", to_gray),
    "blur": ("Làm mờ (Gaussian r=2)", blur),
    "dark": ("Giảm sáng 50%", darken),
    "lowres": ("Độ phân giải 64×64", low_res),
    "noise": ("Nhiễu Gauss σ=25", gaussian_noise),
}


class Corrupt:
    """Transform picklable (cần cho DataLoader nhiều worker trên Windows)."""
    def __init__(self, name):
        if name not in CORRUPTIONS:
            raise ValueError(f"Không có phép biến đổi '{name}'. Chọn một trong: {list(CORRUPTIONS)}")
        self.name = name

    def __call__(self, img):
        return CORRUPTIONS[self.name][1](img)
