"""
Training Script for Siamese Trajectory Network

Trains the Siamese network on trajectory fragment pairs
and evaluates performance on validation set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from siamese_dataset import TrajectoryPairDataset, EnhancedTrajectoryPairDataset, collate_fn, ENHANCED_PIPELINE_AVAILABLE
from siamese_model import SiameseTrajectoryNetwork, CombinedLoss, count_parameters


class SiameseTrainer:
    """Training manager for Siamese Network"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 output_dir: Path,
                 scheduler: optim.lr_scheduler._LRScheduler = None):
        """
        Args:
            model: Siamese network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory to save outputs
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.scheduler = scheduler

        # Training history
        self.history = {
            'train_loss': [],
            'train_bce': [],
            'train_contrastive': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_bce': [],
            'val_contrastive': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_contrastive = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (seq_a, len_a, seq_b, len_b, endpoint_features, labels) in enumerate(pbar):
            # Move to device
            seq_a = seq_a.to(self.device)
            len_a = len_a.to(self.device)
            seq_b = seq_b.to(self.device)
            len_b = len_b.to(self.device)
            endpoint_features = endpoint_features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            similarity, emb_a, emb_b = self.model(seq_a, len_a, seq_b, len_b, endpoint_features)

            # Compute loss
            loss, bce, contrastive = self.criterion(similarity, emb_a, emb_b, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_contrastive += contrastive.item()

            # Accuracy
            predictions = (similarity > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        # Average metrics
        metrics = {
            'loss': epoch_loss / len(self.train_loader),
            'bce': epoch_bce / len(self.train_loader),
            'contrastive': epoch_contrastive / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }

        return metrics

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        """Validate for one epoch"""
        self.model.eval()

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_contrastive = 0.0
        correct = 0
        total = 0

        all_similarities = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc="Validation")
        for seq_a, len_a, seq_b, len_b, endpoint_features, labels in pbar:
            # Move to device
            seq_a = seq_a.to(self.device)
            len_a = len_a.to(self.device)
            seq_b = seq_b.to(self.device)
            len_b = len_b.to(self.device)
            endpoint_features = endpoint_features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            similarity, emb_a, emb_b = self.model(seq_a, len_a, seq_b, len_b, endpoint_features)

            # Compute loss
            loss, bce, contrastive = self.criterion(similarity, emb_a, emb_b, labels)

            # Statistics
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_contrastive += contrastive.item()

            # Accuracy
            predictions = (similarity > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Store for ROC computation
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

        # Average metrics
        metrics = {
            'loss': epoch_loss / len(self.val_loader),
            'bce': epoch_bce / len(self.val_loader),
            'contrastive': epoch_contrastive / len(self.val_loader),
            'accuracy': 100.0 * correct / total,
            'similarities': np.array(all_similarities),
            'labels': np.array(all_labels)
        }

        return metrics

    def train(self, num_epochs: int):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "="*60)
        print(f"STARTING TRAINING - {num_epochs} EPOCHS")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_bce'].append(train_metrics['bce'])
            self.history['train_contrastive'].append(train_metrics['contrastive'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])

            # Validate
            val_metrics = self.validate_epoch()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_bce'].append(val_metrics['bce'])
            self.history['val_contrastive'].append(val_metrics['contrastive'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.2f}% | Val Acc:  {val_metrics['accuracy']:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_loss.pth', epoch, val_metrics)
                print(f"  ✓ Best loss model saved!")

            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.save_checkpoint('best_accuracy.pth', epoch, val_metrics)
                print(f"  ✓ Best accuracy model saved!")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.2f}%")

        # Save final model and history
        self.save_checkpoint('final_model.pth', num_epochs, val_metrics)
        self.save_history()
        self.plot_training_curves()

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: v for k, v in metrics.items() if k not in ['similarities', 'labels']},
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)

    def save_history(self):
        """Save training history to JSON"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_accuracy'], label='Train Acc', linewidth=2)
        ax.plot(epochs, self.history['val_accuracy'], label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Loss components
        ax = axes[1, 0]
        ax.plot(epochs, self.history['train_bce'], label='Train BCE', linewidth=2)
        ax.plot(epochs, self.history['train_contrastive'], label='Train Contrastive', linewidth=2)
        ax.plot(epochs, self.history['val_bce'], label='Val BCE', linewidth=2, linestyle='--')
        ax.plot(epochs, self.history['val_contrastive'], label='Val Contrastive', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Component', fontsize=12)
        ax.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history['learning_rate'], linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {self.output_dir / 'training_curves.png'}")


def main():
    """Main training function"""
    # Configuration
    config = {
        'dataset_names': ['i', 'ii', 'iii'],  # All three scenarios
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'train_val_split': 0.8,
        'model': {
            'input_size': 4,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'similarity_hidden_dim': 64,
            'endpoint_feature_dim': 4,
            'use_endpoint_features': True
        },
        'use_enhanced_dataset': True,  # Use trajectory masking + hard negative mining
        'loss': {
            'margin': 2.0,
            'alpha': 0.1  # Weight for contrastive loss (BCE is primary for inference)
        },
        'scheduler': {
            'step_size': 15,
            'gamma': 0.5
        }
    }

    # Setup - use relative path for portability to Sol
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    # Use enhanced dataset if available and configured
    if config.get('use_enhanced_dataset', False) and ENHANCED_PIPELINE_AVAILABLE:
        print("Using EnhancedTrajectoryPairDataset with masking + hard negative mining")
        full_dataset = EnhancedTrajectoryPairDataset(
            dataset_names=config['dataset_names'],
            normalize=True,
            use_masking=True,
            use_hard_negatives=True,
            augment=False,  # Disable augmentation during initial training
            pairs_per_trajectory=5,
            hard_negative_ratio=0.7
        )
    else:
        print("Using basic TrajectoryPairDataset")
        full_dataset = TrajectoryPairDataset(
            dataset_names=config['dataset_names'],
            normalize=True
        )

    # Train/val split
    train_size = int(config['train_val_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)

    model = SiameseTrajectoryNetwork(**config['model'])
    model = model.to(device)

    print(f"\nModel parameters: {count_parameters(model):,}")

    # Create loss function
    criterion = CombinedLoss(**config['loss'])

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        **config['scheduler']
    )

    # Create trainer
    trainer = SiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        scheduler=scheduler
    )

    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
