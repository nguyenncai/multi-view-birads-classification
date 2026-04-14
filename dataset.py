"""
VinDr Mammography Dataset for Multi-View, Multi-Task Learning
Supports 4-view input (R_CC, R_MLO, L_CC, L_MLO) with diagnosis and BI-RADS labels
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from annotation_schema import load_vindr_annotations

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAMMOCLIP_MEAN = [0.3089279, 0.3089279, 0.3089279]
MAMMOCLIP_STD = [0.25053555408335154, 0.25053555408335154, 0.25053555408335154]


class VinDrMammographyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        split='training',
        transform=None,
        img_size=512,
        augmentation_profile='light',
        normalization_profile='imagenet',
        preprocess_profile='none',
    ):
        """
        Args:
            csv_path: Path to cleaned_label.csv
            img_dir: Path to cleaned_images directory
            split: 'training' or 'test'
            transform: Optional transform to be applied on images
            img_size: Target image size (default 512x512)
        """
        self.img_dir = img_dir
        self.split = split
        self.img_size = img_size
        self.augmentation_profile = augmentation_profile
        self.normalization_profile = normalization_profile
        self.preprocess_profile = preprocess_profile
        
        # Load and filter data from either supported annotation schema.
        df = load_vindr_annotations(csv_path)
        df = df[df['split'] == split].copy()
        df['resolved_img_path'] = df.apply(
            lambda row: self._resolve_image_path(row['image_id'], row['patient_id']),
            axis=1
        )
        
        # Group by patient to get 4-view sets
        self.patient_data = self._group_by_patient(df)
        self.patient_ids = list(self.patient_data.keys())
        
        # Setup transforms
        if transform is None:
            if split == 'training':
                if augmentation_profile == 'heavy':
                    self.transform = self._get_heavy_train_transform(img_size)
                else:
                    self.transform = self._get_light_train_transform(img_size)
            else:
                self.transform = self._get_test_transform(img_size)
        else:
            self.transform = transform

    def _get_normalization_stats(self):
        if self.normalization_profile == 'mammoclip':
            return MAMMOCLIP_MEAN, MAMMOCLIP_STD
        return IMAGENET_MEAN, IMAGENET_STD

    def _get_light_train_transform(self, img_size):
        mean, std = self._get_normalization_stats()
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _get_heavy_train_transform(self, img_size):
        mean, std = self._get_normalization_stats()
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _get_test_transform(self, img_size):
        mean, std = self._get_normalization_stats()
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def _group_by_patient(self, df):
        """Group images by patient_id to create 4-view sets"""
        patient_data = {}
        
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id]
            
            # Initialize view dictionary
            views = {'R_CC': None, 'R_MLO': None, 'L_CC': None, 'L_MLO': None}
            labels = {}
            
            for _, row in patient_df.iterrows():
                view_key = f"{row['laterality']}_{row['view']}"
                views[view_key] = {
                    'image_id': row['image_id'],
                    'path': row['resolved_img_path'],
                }
                
                # Store labels per side
                side = row['laterality']
                if side not in labels:
                    cancer = row['cancer'] if 'cancer' in row and not pd.isna(row['cancer']) else -1
                    labels[side] = {
                        'cancer': int(cancer) if cancer != -1 else -1,
                        'birads': int(row['BIRADS']) if not pd.isna(row['BIRADS']) else -1
                    }
            
            patient_data[patient_id] = {
                'views': views,
                'labels': labels
            }
        
        return patient_data
    
    def _resolve_image_path(self, image_id, patient_id):
        primary = os.path.join(self.img_dir, f"{patient_id}_{image_id}.png")
        if os.path.exists(primary):
            return primary
        fallback = os.path.join(self.img_dir, f"{image_id}.png")
        return fallback

    def _preprocess_mammoclip_style(self, raw_img):
        from mammoclip.utils import extract_breast

        img = np.array(raw_img)
        if img.ndim == 3:
            if img.shape[2] >= 3:
                if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
                    img = img[:, :, 0]
                else:
                    img = np.round(
                        0.299 * img[:, :, 0]
                        + 0.587 * img[:, :, 1]
                        + 0.114 * img[:, :, 2]
                    ).astype(np.float32)
            else:
                img = img[:, :, 0]

        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            img = (img * 255.0).astype(np.uint8)

        cropped = extract_breast(img)
        if cropped.size == 0:
            cropped = img
        return Image.fromarray(cropped).convert('RGB')

    def _preprocess_mammodino_style(self, raw_img):
        import cv2

        img = np.array(raw_img)
        if img.ndim == 3:
            if img.shape[2] >= 3:
                if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2]):
                    img = img[:, :, 0]
                else:
                    img = np.round(
                        0.299 * img[:, :, 0]
                        + 0.587 * img[:, :, 1]
                        + 0.114 * img[:, :, 2]
                    )
            else:
                img = img[:, :, 0]

        img = img.astype(np.float32)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return Image.fromarray(img).convert('RGB')

    def _load_image(self, image_id, patient_id, resolved_path=None):
        """Load and preprocess single image"""
        if image_id is None:
            # Return zero tensor for missing views
            return torch.zeros(3, self.img_size, self.img_size)
        
        img_path = resolved_path if resolved_path is not None else self._resolve_image_path(image_id, patient_id)
        
        try:
            with Image.open(img_path) as raw_img:
                if self.preprocess_profile == 'mammoclip':
                    img = self._preprocess_mammoclip_style(raw_img)
                elif self.preprocess_profile == 'mammodino':
                    img = self._preprocess_mammodino_style(raw_img)
                else:
                    img = raw_img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size)
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = self.patient_data[patient_id]
        
        # Load 4 views
        views = data['views']
        images = {
            'R_CC': self._load_image(views['R_CC']['image_id'] if views['R_CC'] else None, patient_id, resolved_path=views['R_CC']['path'] if views['R_CC'] else None),
            'R_MLO': self._load_image(views['R_MLO']['image_id'] if views['R_MLO'] else None, patient_id, resolved_path=views['R_MLO']['path'] if views['R_MLO'] else None),
            'L_CC': self._load_image(views['L_CC']['image_id'] if views['L_CC'] else None, patient_id, resolved_path=views['L_CC']['path'] if views['L_CC'] else None),
            'L_MLO': self._load_image(views['L_MLO']['image_id'] if views['L_MLO'] else None, patient_id, resolved_path=views['L_MLO']['path'] if views['L_MLO'] else None)
        }
        
        # Prepare labels
        labels = data['labels']
        
        # Right side labels
        right_cancer = labels.get('R', {}).get('cancer', -1)
        right_birads = labels.get('R', {}).get('birads', -1)
        
        # Left side labels
        left_cancer = labels.get('L', {}).get('cancer', -1)
        left_birads = labels.get('L', {}).get('birads', -1)
        
        # Convert BI-RADS to 0-indexed (1-5 -> 0-4)
        if right_birads != -1:
            right_birads = right_birads - 1
        if left_birads != -1:
            left_birads = left_birads - 1
        
        return {
            'images': images,
            'patient_id': patient_id,
            'labels': {
                'right_cancer': torch.tensor(right_cancer, dtype=torch.long),
                'left_cancer': torch.tensor(left_cancer, dtype=torch.long),
                'right_birads': torch.tensor(right_birads, dtype=torch.long),
                'left_birads': torch.tensor(left_birads, dtype=torch.long)
            }
        }


def get_class_weights(csv_path, split='training', method='inverse', deduplicate_breasts=False):
    """Calculate class weights for handling imbalance."""
    df = load_vindr_annotations(csv_path)
    df = df[df['split'] == split].copy()

    if deduplicate_breasts:
        df = df.drop_duplicates(subset=['patient_id', 'laterality'])

    # Cancer class weights are optional because breast-level BI-RADS files
    # may not carry cancer labels.
    if 'cancer' in df.columns and df['cancer'].notna().any():
        cancer_counts = df['cancer'].astype(int).value_counts()
        n_max_cancer = cancer_counts.max()
        cancer_weights = []
        for label in [0, 1]:
            count = cancer_counts.get(label, 0)
            if count <= 0:
                weight = 1.0
            elif method == 'inverse':
                weight = n_max_cancer / count
            elif method == 'inverse_sqrt':
                weight = (n_max_cancer / count) ** 0.5
            elif method == 'effective_num':
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, count)
                weight = (1.0 - beta) / effective_num
            else:
                weight = 1.0
            cancer_weights.append(float(weight))
        cancer_tensor = torch.tensor(cancer_weights, dtype=torch.float32)
        cancer_tensor = cancer_tensor / cancer_tensor.sum() * len(cancer_tensor)
    else:
        cancer_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32)

    birads_df = df[df['BIRADS'].notna()].copy()
    birads_df['BIRADS'] = birads_df['BIRADS'].astype(int)
    birads_counts = birads_df['BIRADS'].value_counts()
    n_max_birads = birads_counts.max() if len(birads_counts) > 0 else 1

    birads_weights = []
    for label in range(1, 6):
        count = birads_counts.get(label, 0)
        if count <= 0:
            weight = 1.0
        elif method == 'inverse':
            weight = n_max_birads / count
        elif method == 'inverse_sqrt':
            weight = (n_max_birads / count) ** 0.5
        elif method == 'effective_num':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, count)
            weight = (1.0 - beta) / effective_num
        else:
            weight = 1.0
        birads_weights.append(float(weight))
    birads_tensor = torch.tensor(birads_weights, dtype=torch.float32)
    birads_tensor = birads_tensor / birads_tensor.sum() * len(birads_tensor)

    return {
        'cancer': cancer_tensor,
        'birads': birads_tensor,
    }
