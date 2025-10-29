"""
Enhanced Training Module for Hybrid CV Screening Model
Implements training with early stopping, learning rate scheduling, and experiment tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.models.hybrid_model import EnhancedHybridModel, CVDataset, EarlyStopping
from src.data.data_processor import EnhancedDataProcessor
from config.config import *

# Set up logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and HARDWARE_CONFIG["use_cuda"] else 'cpu')
logger.info(f"Using device: {device}")

class ExperimentTracker:
    """Track experiments and results for research purposes"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = RESULTS_DIR / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        self.config = {
            'bert_config': BERT_CONFIG,
            'cnn_config': CNN_CONFIG,
            'lstm_config': LSTM_CONFIG,
            'traditional_config': TRADITIONAL_ML_CONFIG,
            'hybrid_config': HYBRID_CONFIG,
            'training_config': TRAINING_CONFIG,
            'dataset_config': DATASET_CONFIG
        }
        
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Save intermediate results
        self.save_results()
        
    def log_final_results(self, test_metrics: Dict[str, float], model_info: Dict[str, Any]):
        """Log final test results and model information"""
        final_results = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_history': self.metrics_history,
            'test_metrics': test_metrics,
            'model_info': model_info
        }
        
        # Save comprehensive results
        with open(self.results_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Generate plots
        self.generate_plots()
        
        logger.info(f"Final results saved to {self.results_dir}")
        
    def save_results(self):
        """Save current state of results"""
        current_results = {
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        with open(self.results_dir / 'training_progress.json', 'w') as f:
            json.dump(current_results, f, indent=2, default=str)
    
    def generate_plots(self):
        """Generate training plots"""
        # Training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        epochs = range(len(self.metrics_history['train_loss']))
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.metrics_history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(epochs, self.metrics_history['val_f1'], label='Validation F1')
        axes[1, 0].plot(epochs, self.metrics_history['val_precision'], label='Validation Precision')
        axes[1, 0].plot(epochs, self.metrics_history['val_recall'], label='Validation Recall')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(epochs, self.metrics_history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

class HybridModelTrainer:
    """Enhanced trainer for the hybrid CV screening model"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_tracker = ExperimentTracker(experiment_name)
        self.data_processor = EnhancedDataProcessor()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if TRAINING_CONFIG["use_mixed_precision"] else None
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Any]:
        """Prepare data loaders"""
        logger.info("Preparing data...")
        
        # Load and preprocess data
        try:
            # Try to load processed data first
            processed_df, metadata = self.data_processor.load_processed_data()
            logger.info("Loaded previously processed data")
        except:
            # Process raw data
            raw_df = self.data_processor.load_data(DATASET_CONFIG["raw_dataset_path"])
            processed_df, metadata = self.data_processor.preprocess_data(raw_df)
            self.data_processor.save_processed_data(processed_df, metadata)
            logger.info("Processed and saved new data")
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.create_training_data(processed_df)
        
        # Prepare traditional features
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self.data_processor.prepare_features(X_train, X_val, X_test)
        
        # Create datasets
        train_dataset = CVDataset(X_train, y_train, self.data_processor.tokenizer)
        val_dataset = CVDataset(X_val, y_val, self.data_processor.tokenizer)
        test_dataset = CVDataset(X_test, y_test, self.data_processor.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=HARDWARE_CONFIG["num_workers"],
            pin_memory=HARDWARE_CONFIG["pin_memory"]
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=HARDWARE_CONFIG["num_workers"],
            pin_memory=HARDWARE_CONFIG["pin_memory"]
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=HARDWARE_CONFIG["num_workers"],
            pin_memory=HARDWARE_CONFIG["pin_memory"]
        )
        
        return train_loader, val_loader, test_loader, (X_train_tfidf, X_val_tfidf, X_test_tfidf), metadata
    
    def create_model(self, metadata: Dict[str, Any], tfidf_dim: int) -> EnhancedHybridModel:
        """Create and initialize the hybrid model"""
        logger.info("Creating hybrid model...")
        
        num_classes = len(metadata["categories"]) if metadata["categories"] else 2
        vocab_size = 30000  # Standard vocabulary size
        
        model = EnhancedHybridModel(
            num_classes=num_classes,
            vocab_size=vocab_size,
            tfidf_dim=tfidf_dim
        )
        
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def setup_training(self, model: EnhancedHybridModel, train_loader: DataLoader):
        """Setup optimizer and scheduler"""
        # Optimizer with different learning rates for different components
        bert_param_ids = set(id(p) for p in model.bert_component.parameters())
        other_params = [p for p in model.parameters() if id(p) not in bert_param_ids]
        bert_params = list(model.bert_component.parameters())
        
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': TRAINING_CONFIG["learning_rate"]},
            {'params': other_params, 'lr': TRAINING_CONFIG["learning_rate"] * 2}
        ], weight_decay=TRAINING_CONFIG["weight_decay"])
        
        # Learning rate scheduler
        total_steps = len(train_loader) * TRAINING_CONFIG["max_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: EnhancedHybridModel, train_loader: DataLoader, 
                   tfidf_data: torch.Tensor, optimizer, scheduler) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Get corresponding TF-IDF features
            batch_size = input_ids.size(0)
            start_idx = batch_idx * TRAINING_CONFIG["batch_size"]
            end_idx = start_idx + batch_size
            batch_tfidf = tfidf_data[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            if self.scaler and TRAINING_CONFIG["use_mixed_precision"]:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, batch_tfidf)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if TRAINING_CONFIG["gradient_clip_value"] > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_value"])
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(input_ids, attention_mask, batch_tfidf)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                loss.backward()
                
                if TRAINING_CONFIG["gradient_clip_value"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_value"])
                
                optimizer.step()
            
            scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': scheduler.get_last_lr()[0]
        }
    
    def validate_epoch(self, model: EnhancedHybridModel, val_loader: DataLoader, 
                      tfidf_data: torch.Tensor) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Get corresponding TF-IDF features
                batch_size = input_ids.size(0)
                start_idx = batch_idx * TRAINING_CONFIG["batch_size"]
                end_idx = start_idx + batch_size
                batch_tfidf = tfidf_data[start_idx:end_idx].to(device)
                
                outputs = model(input_ids, attention_mask, batch_tfidf)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self) -> EnhancedHybridModel:
        """Main training loop"""
        logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader, test_loader, tfidf_data, metadata = self.prepare_data()
        X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_data
        
        # Convert TF-IDF to tensors
        X_train_tfidf = torch.FloatTensor(X_train_tfidf.toarray())
        X_val_tfidf = torch.FloatTensor(X_val_tfidf.toarray())
        X_test_tfidf = torch.FloatTensor(X_test_tfidf.toarray())
        
        # Create model
        self.model = self.create_model(metadata, X_train_tfidf.shape[1])
        
        # Setup training
        self.optimizer, self.scheduler = self.setup_training(self.model, train_loader)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG["patience"],
            min_delta=TRAINING_CONFIG["min_delta"]
        )
        
        best_val_f1 = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(TRAINING_CONFIG["max_epochs"]):
            logger.info(f"Epoch {epoch+1}/{TRAINING_CONFIG['max_epochs']}")
            
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, X_train_tfidf, 
                                           self.optimizer, self.scheduler)
            
            # Validate
            val_metrics = self.validate_epoch(self.model, val_loader, X_val_tfidf)
            
            # Log metrics
            epoch_metrics = {
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'learning_rate': train_metrics['learning_rate']
            }
            
            self.experiment_tracker.log_epoch(epoch, epoch_metrics)
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
            
            # Check early stopping
            if early_stopping(val_metrics['f1']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Log progress
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        test_metrics = self.validate_epoch(self.model, test_loader, X_test_tfidf)
        
        # Log final results
        model_info = {
            'feature_weights': self.model.get_feature_weights(),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        self.experiment_tracker.log_final_results(test_metrics, model_info)
        
        # Save model
        self.save_model(metadata)
        
        logger.info(f"Training complete! Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Target accuracy achieved: {test_metrics['accuracy'] >= PERFORMANCE_TARGET['accuracy']}")
        
        return self.model
    
    def save_model(self, metadata: Dict[str, Any]):
        """Save the trained model and associated files"""
        logger.info("Saving model...")
        
        # Save model state
        torch.save(self.model.state_dict(), MODEL_PATHS["hybrid_model"])
        
        # Save model metadata
        model_metadata = {
            'model_architecture': 'EnhancedHybridModel',
            'num_classes': len(metadata["categories"]) if metadata["categories"] else 2,
            'classes': metadata["categories"],
            'vocab_size': 30000,
            'feature_weights': self.model.get_feature_weights(),
            'config': self.experiment_tracker.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(MODEL_PATHS["model_metadata"], 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        # Save tokenizer
        self.model.tokenizer.save_pretrained(MODEL_PATHS["tokenizer"])
        
        # Save vectorizers
        vectorizers = {
            'tfidf_vectorizer': self.data_processor.tfidf_vectorizer,
            'label_encoder': self.data_processor.label_encoder
        }
        
        with open(MODEL_PATHS["vectorizers"], 'wb') as f:
            pickle.dump(vectorizers, f)
        
        logger.info(f"Model saved to {MODELS_DIR}")

def main():
    """Main training function"""
    trainer = HybridModelTrainer(experiment_name="enhanced_hybrid_cv_screening")
    model = trainer.train()
    return model

if __name__ == "__main__":
    main()