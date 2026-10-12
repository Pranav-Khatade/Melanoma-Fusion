import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def patient_level_split(df, group_col, label_col, val=0.15, test=0.15, seed=42):
    """70/15/15 split with NO patient appearing in more than one split
    (image-level splits leak patients and inflate metrics)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, df[label_col], df[group_col]))
    trainval, test_df = df.iloc[trainval_idx], df.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    tr_idx, va_idx = next(gss2.split(trainval, trainval[label_col], trainval[group_col]))
    return trainval.iloc[tr_idx], trainval.iloc[va_idx], test_df


def build_meta_preprocessor(age_col, sex_col, site_col):
    """z-score age, one-hot sex and anatomical site, impute missing values.
    Fit on the TRAIN split only."""
    return ColumnTransformer(
        [
            ("age", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), [age_col]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             [sex_col, site_col]),
        ],
        sparse_threshold=0.0,
    )


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def remove_hair(img_rgb: np.ndarray) -> np.ndarray:
    """Black-hat morphology + inpainting (classic DullRazor-style)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_rgb, mask, 1, cv2.INPAINT_TELEA)


def preprocess_image(pil: Image.Image, size: int, clahe=True, hair=True) -> Image.Image:
    arr = np.array(pil.convert("RGB"))
    if hair:
        arr = remove_hair(arr)
    if clahe:
        arr = apply_clahe(arr)
    return Image.fromarray(arr).resize((size, size))


def get_transforms(size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, img_col, label_col, meta, transform, size=224,
                 ext=".jpg", clahe=True, hair=True):
        self.df, self.img_dir, self.img_col = df.reset_index(drop=True), img_dir, img_col
        self.labels = self.df[label_col].values.astype(np.float32)
        self.meta = meta.astype(np.float32)
        self.transform, self.size, self.ext = transform, size, ext
        self.clahe, self.hair = clahe, hair

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = f"{self.img_dir}/{self.df.loc[i, self.img_col]}{self.ext}"
        img = preprocess_image(Image.open(path), self.size, self.clahe, self.hair)
        return (self.transform(img), torch.from_numpy(self.meta[i]),
                torch.tensor(self.labels[i]))


def save_preprocessor(prep, path):
    joblib.dump(prep, path)


def load_preprocessor(path):
    return joblib.load(path)
