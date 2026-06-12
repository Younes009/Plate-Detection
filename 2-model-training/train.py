"""
Script d'entraînement complet pour SSD - Détection de plaques d'immatriculation

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 32
    python train.py --resume checkpoints/best_model.pt
    python train.py --device cuda
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import create_dataloaders, MAX_OBJECTS
from model import create_model, predict, NUM_CLASSES, CLASS_NAMES
from loss import create_ssd_loss


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_CONFIG = {
    # Paths
    'data_root': '../1-spark-preprocessing/data/processed',
    'output_dir': './checkpoints',
    'log_dir': './logs',

    # Training
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,

    # Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'reg_weight': 1.0,

    # Scheduler
    'lr_patience': 5,
    'lr_factor': 0.5,
    'min_lr': 1e-6,

    # Early stopping
    'early_stop_patience': 10,

    # Device
    'device': 'cpu',  # Changer en 'cuda' pour GPU
    'num_workers': 0
}


# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    """Classe d'entraînement complète"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])

        # Créer les dossiers
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

        # Timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(config['log_dir']) / f"training_{self.timestamp}.csv"

        # Affichage
        self._print_header()

        # Initialisation
        self._init_model()
        self._init_data()
        self._init_training()

        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_history = []
        self.val_history = []
        self.current_epoch = 0

    def _print_header(self):
        """Affiche l'en-tête"""
        print("=" * 60)
        print("🚀 SSD PLATE DETECTION - Training")
        print("=" * 60)
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Data: {self.config['data_root']}")
        print("=" * 60)

    def _init_model(self):
        """Initialise le modèle"""
        print("\n📦 Initializing model...")
        self.model = create_model(num_classes=NUM_CLASSES)
        self.model.to(self.device)

    def _init_data(self):
        """Initialise les dataloaders"""
        print("\n📂 Loading data...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )

    def _init_training(self):
        """Initialise optimizer, scheduler, loss"""
        print("\n⚙️ Initializing training components...")

        # Loss
        self.criterion = create_ssd_loss(
            num_classes=NUM_CLASSES,
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma'],
            reg_weight=self.config['reg_weight']
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience'],
            min_lr=self.config['min_lr']
            #verbose=True
        )

        print("   ✅ Loss: Focal Loss + Smooth L1")
        print("   ✅ Optimizer: Adam")
        print("   ✅ Scheduler: ReduceLROnPlateau")

    def train_epoch(self) -> Dict:
        """Entraîne pour une epoch"""
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for batch_idx, (images, cls_targets, reg_targets, pos_mask) in enumerate(self.train_loader):
            # Vers device
            images = images.to(self.device)
            cls_targets = cls_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)
            pos_mask = pos_mask.to(self.device)

            anchors = self.model.anchors.to(self.device)

            num_pos = pos_mask.sum().item()
            if batch_idx % 20 == 0:
                print(f"      [DEBUG] Batch {batch_idx}: Objects matched to anchors: {num_pos}")

            # Forward
            cls_preds, reg_preds = self.model(images)

            # Loss
            loss, cls_loss, reg_loss = self.criterion(
                cls_preds, reg_preds, cls_targets, reg_targets, pos_mask, anchors
                )
            

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            # Accumulation
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1

            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"      Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f} (cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f})")

        return {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'reg_loss': total_reg_loss / num_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        """Valide le modèle"""
        self.model.eval()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for images, cls_targets, reg_targets, pos_mask in self.val_loader:
            images = images.to(self.device)
            cls_targets = cls_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)
            pos_mask = pos_mask.to(self.device)

            cls_preds, reg_preds = self.model(images)

            loss, cls_loss, reg_loss = self.criterion(
                cls_preds, reg_preds, cls_targets, reg_targets, pos_mask
            )

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'cls_loss': total_cls_loss / max(num_batches, 1),
            'reg_loss': total_reg_loss / max(num_batches, 1)
        }

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = Path(self.config['output_dir']) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Saved best model to {best_path}")

    def load_checkpoint(self, path: str, resume_training: bool = True):
        """Charge un checkpoint"""
        print(f"\n🔄 Loading checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.train_history = checkpoint.get('train_history', [])
            self.val_history = checkpoint.get('val_history', [])

            print(f"   ✅ Resumed from epoch {self.current_epoch}")
            print(f"   📊 Best val loss: {self.best_val_loss:.4f}")
        else:
            print(f"   ✅ Loaded model weights only")

    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log les métriques dans un CSV"""
        # Créer le fichier si nécessaire
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_cls_loss', 'train_reg_loss',
                    'val_loss', 'val_cls_loss', 'val_reg_loss', 'lr', 'timestamp'
                ])

        # Ajouter la ligne
        current_lr = self.optimizer.param_groups[0]['lr']
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['cls_loss']:.6f}",
                f"{train_metrics['reg_loss']:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['cls_loss']:.6f}",
                f"{val_metrics['reg_loss']:.6f}",
                f"{current_lr:.8f}",
                datetime.now().isoformat()
            ])

    def train(self) -> float:
        """Boucle d'entraînement principale"""
        print("\n" + "=" * 60)
        print("🏋️ Starting training...")
        print("=" * 60)

        start_epoch = self.current_epoch
        total_time = 0

        for epoch in range(start_epoch, self.config['epochs']):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\n📅 Epoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 40)

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # Update scheduler
            self.scheduler.step(val_metrics['loss'])

            # Log
            self.log_metrics(epoch + 1, train_metrics, val_metrics)

            # Print
            epoch_time = time.time() - epoch_start
            total_time += epoch_time
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\n   📊 Train Loss: {train_metrics['loss']:.4f} "
                  f"(cls={train_metrics['cls_loss']:.4f}, reg={train_metrics['reg_loss']:.4f})")
            print(f"   📊 Val Loss:   {val_metrics['loss']:.4f} "
                  f"(cls={val_metrics['cls_loss']:.4f}, reg={val_metrics['reg_loss']:.4f})")
            print(f"   📈 LR: {current_lr:.6f}")
            print(f"   ⏱️ Time: {epoch_time:.1f}s")

            # Check improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                print(f"   🎉 New best validation loss!")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            checkpoint_path = Path(self.config['output_dir']) / f"checkpoint_epoch{epoch + 1}.pt"
            self.save_checkpoint(str(checkpoint_path), is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stop_patience']:
                print(f"\n⛔ Early stopping after {epoch + 1} epochs")
                break

        # Summary
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print(f"   Log file: {self.log_file}")
        print("=" * 60)

        # Save final
        final_path = Path(self.config['output_dir']) / "final_model.pt"
        self.save_checkpoint(str(final_path))
        print(f"   💾 Final model saved to {final_path}")

        return self.best_val_loss

    @torch.no_grad()
    def test(self) -> Dict:
        """Test final sur le test set"""
        print("\n" + "=" * 60)
        print("🧪 Testing on test set...")
        print("=" * 60)

        self.model.eval()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0

        for images, cls_targets, reg_targets, pos_mask in self.test_loader:
            images = images.to(self.device)
            cls_targets = cls_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)
            pos_mask = pos_mask.to(self.device)

            cls_preds, reg_preds = self.model(images)
            anchors = self.model.anchors.to(self.device)

            loss, cls_loss, reg_loss = self.criterion(
                cls_preds, reg_preds, cls_targets, reg_targets, pos_mask, anchors
            )

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1

        test_metrics = {
            'loss': total_loss / max(num_batches, 1),
            'cls_loss': total_cls_loss / max(num_batches, 1),
            'reg_loss': total_reg_loss / max(num_batches, 1)
        }

        print(f"\n   📊 Test Loss: {test_metrics['loss']:.4f}")
        print(f"      Classification: {test_metrics['cls_loss']:.4f}")
        print(f"      Regression: {test_metrics['reg_loss']:.4f}")

        return test_metrics


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================
def load_model_for_inference(checkpoint_path: str, device: str = 'cpu'):
    """
    Charge un modèle pour l'inférence.

    Args:
        checkpoint_path: Chemin vers le checkpoint
        device: 'cpu' ou 'cuda'

    Returns:
        model: Modèle prêt pour inférence
    """
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = create_model(num_classes=checkpoint.get('num_classes', NUM_CLASSES))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Device: {device}")

    return model


def predict_and_save_csv(
    model,
    images: torch.Tensor,
    image_names: List[str],
    output_path: str,
    score_threshold: float = 0.5
):
    """
    Fait des prédictions et sauvegarde en CSV.

    Args:
        model: Modèle SSD
        images: Tensor [B, 3, 256, 256]
        image_names: Liste des noms d'images
        output_path: Chemin du fichier CSV
        score_threshold: Seuil de confiance

    Format CSV:
        image_name, predicted_boxes, predicted_classes, scores, timestamp
    """
    results = predict(model, images, score_threshold=score_threshold)

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Header si fichier vide
        if f.tell() == 0:
            writer.writerow(['image_name', 'predicted_boxes', 'predicted_classes', 'scores', 'timestamp'])

        timestamp = datetime.now().isoformat()

        for i, (name, det) in enumerate(zip(image_names, results)):
            boxes = det['boxes'].cpu().numpy().tolist()
            classes = det['classes'].cpu().numpy().tolist()
            scores = det['scores'].cpu().numpy().tolist()

            writer.writerow([
                name,
                str(boxes),
                str(classes),
                str(scores),
                timestamp
            ])

    print(f"✅ Predictions saved to {output_path}")


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SSD for license plate detection'
    )

    # Paths
    parser.add_argument('--data-root', type=str, default=DEFAULT_CONFIG['data_root'],
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_CONFIG['output_dir'],
                        help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_CONFIG['log_dir'],
                        help='Directory to save logs')

    # Training
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')

    # Device
    parser.add_argument('--device', type=str, default=DEFAULT_CONFIG['device'],
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=DEFAULT_CONFIG['num_workers'],
                        help='Number of data loading workers')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')

    # Test only
    parser.add_argument('--test-only', action='store_true',
                        help='Only run test evaluation')

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()

    # Configuration
    config = DEFAULT_CONFIG.copy()
    config['data_root'] = args.data_root
    config['output_dir'] = args.output_dir
    config['log_dir'] = args.log_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['device'] = args.device
    config['num_workers'] = args.num_workers

    # Auto-detect CUDA
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        config['device'] = 'cpu'

    if config['device'] == 'cuda':
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")

    # Créer le trainer
    trainer = Trainer(config)

    # Resume si spécifié
    if args.resume:
        trainer.load_checkpoint(args.resume, resume_training=not args.test_only)

    # Test only mode
    if args.test_only:
        trainer.test()
        return

    # Training
    trainer.train()

    # Test final
    trainer.test()


if __name__ == "__main__":
    main()
