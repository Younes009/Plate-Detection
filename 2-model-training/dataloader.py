"""
DataLoader pour SSD - Lecture Parquet avec images binaires et targets pré-calculés

Supporte les fichiers Parquet générés par Spark grâce à PySpark.
"""

import io
import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as T


# ============================================================================
# config
# ============================================================================
IMG_SIZE = 256
MAX_OBJECTS = 20  # nbr max d'objets par image (pour padding)


# ============================================================================
# PARQUET READER (Compatible Spark)
# ============================================================================
def read_spark_parquet(path: str) -> pd.DataFrame:
    """
    Lit un fichier Parquet généré par Spark.
    Utilise PySpark si PyArrow échoue (compatibilité Spark 4.x).

    Args:
        path: Chemin vers le fichier .parquet

    Returns:
        DataFrame pandas
    """
    # Méthode 1: PyArrow standard
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        table = pf.read()
        return table.to_pandas()
    except Exception:
        pass

    # Méthode 2: pandas direct
    try:
        return pd.read_parquet(path)
    except Exception:
        pass

    # Méthode 3: PySpark (pour fichiers Spark 4.x)
    try:
        from pyspark.sql import SparkSession

        # Créer une session Spark légère
        spark = SparkSession.builder \
            .appName("DataLoader") \
            .master("local[1]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.ui.enabled", "false") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        # Lire et convertir en pandas
        df_spark = spark.read.parquet(path)
        df_pandas = df_spark.toPandas()

        spark.stop()

        return df_pandas
    except Exception as e:
        raise RuntimeError(f"Cannot read parquet file: {path}. Error: {e}")


# ============================================================================
# DATASET
# ============================================================================
class PlateDataset(Dataset):
    """
    Dataset pour plaques d'immatriculation.

    Lit un fichier Parquet contenant:
    - image_name: string
    - images: binary (JPEG 256x256)
    - cls_targets: array<float> (classes pour chaque bbox)
    - reg_targets: array<array<float>> (cx, cy, w, h normalisés)
    - pos_mask: array<float> (1.0 si bbox valide, 0.0 sinon)
    """

    def __init__(self, parquet_path: str, transform: Optional[T.Compose] = None):
        """
        Args:
            parquet_path: Chemin vers le fichier .parquet
            transform: Transformations torchvision (optionnel)
        """
        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # Charger le Parquet
        self.df = read_spark_parquet(str(parquet_path))

        # Transform par défaut
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"   ✅ Loaded {len(self.df)} samples from {parquet_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor [3, H, W]
            cls_targets: Tensor [MAX_OBJECTS]
            reg_targets: Tensor [MAX_OBJECTS, 4]
            pos_mask: Tensor [MAX_OBJECTS]
        """
        row = self.df.iloc[idx]

        # 1. Image binary -> tensor
        image_bytes = row['images']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image)

        # 2. Récupérer les targets (listes depuis Parquet)
        cls_list = row['cls_targets'] if row['cls_targets'] is not None else []
        reg_list = row['reg_targets'] if row['reg_targets'] is not None else []
        mask_list = row['pos_mask'] if row['pos_mask'] is not None else []

        # Convertir en listes si nécessaire (numpy arrays)
        if hasattr(cls_list, 'tolist'):
            cls_list = cls_list.tolist()
        if hasattr(reg_list, 'tolist'):
            reg_list = reg_list.tolist()
        if hasattr(mask_list, 'tolist'):
            mask_list = mask_list.tolist()

        num_objects = len(cls_list)

        # 3. Créer tensors avec padding
        cls_targets = torch.zeros(MAX_OBJECTS, dtype=torch.float32)
        reg_targets = torch.zeros(MAX_OBJECTS, 4, dtype=torch.float32)
        pos_mask = torch.zeros(MAX_OBJECTS, dtype=torch.float32)

        # Remplir avec les vraies valeurs
        for i in range(min(num_objects, MAX_OBJECTS)):
            cls_targets[i] = float(cls_list[i])
            pos_mask[i] = float(mask_list[i]) if i < len(mask_list) else 1.0

            if i < len(reg_list) and reg_list[i] is not None:
                bbox = reg_list[i]
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                if len(bbox) >= 4:
                    reg_targets[i] = torch.tensor(bbox[:4], dtype=torch.float32)

        return image, cls_targets, reg_targets, pos_mask

    def get_image_name(self, idx: int) -> str:
        """Retourne le nom de l'image (utile pour l'inférence)"""
        return self.df.iloc[idx]['image_name']


# ============================================================================
# COLLATE FUNCTION (pour batch)
# ============================================================================
def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function pour batching.
    Déjà paddé dans __getitem__, donc simple stack.
    """
    images, cls_targets, reg_targets, pos_masks = zip(*batch)

    images = torch.stack(images, dim=0)
    cls_targets = torch.stack(cls_targets, dim=0)
    reg_targets = torch.stack(reg_targets, dim=0)
    pos_masks = torch.stack(pos_masks, dim=0)

    return images, cls_targets, reg_targets, pos_masks


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    train_file: str = "train.parquet",
    valid_file: str = "valid.parquet",
    test_file: str = "test.parquet"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour train, valid et test.

    Args:
        data_root: Répertoire contenant les fichiers Parquet
        batch_size: Taille du batch
        num_workers: nbr de workers pour le chargement
        train_file, valid_file, test_file: Noms des fichiers

    Returns:
        train_loader, val_loader, test_loader
    """
    data_root = Path(data_root)

    print("📂 Loading datasets...")

    # Datasets
    train_dataset = PlateDataset(data_root / train_file)
    val_dataset = PlateDataset(data_root / valid_file)
    test_dataset = PlateDataset(data_root / test_file)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )

    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Valid: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    # Test rapide
    data_root = "../1-spark-preprocessing/data/processed"

    print("🧪 Testing DataLoader...")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0
    )

    # Vérifier un batch
    print("\n🔍 Checking one batch...")
    for images, cls_targets, reg_targets, pos_masks in train_loader:
        print(f"   Images shape: {images.shape}")
        print(f"   Cls targets shape: {cls_targets.shape}")
        print(f"   Reg targets shape: {reg_targets.shape}")
        print(f"   Pos masks shape: {pos_masks.shape}")
        print(f"   First image stats: min={images[0].min():.3f}, max={images[0].max():.3f}")
        break

    print("\n✅ DataLoader OK!")
