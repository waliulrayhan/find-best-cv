"""
Utility functions for the CV screening project
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_json(data: Dict[str, Any], path: Path):
    """Save data as JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data"""
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(data: Any, path: Path):
    """Save data as pickle"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: Path) -> Any:
    """Load pickle data"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_training_curves(metrics_history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(len(metrics_history['train_loss']))
    
    # Loss
    axes[0, 0].plot(epochs, metrics_history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs, metrics_history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, metrics_history['train_accuracy'], label='Train Acc')
    axes[0, 1].plot(epochs, metrics_history['val_accuracy'], label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(epochs, metrics_history['val_f1'], label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(epochs, metrics_history['learning_rate'], label='LR')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Optional[Path] = None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_size(path: Path) -> str:
    """Get human readable file size"""
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def print_model_summary(model, input_shape: tuple = None):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    print("="*80)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }