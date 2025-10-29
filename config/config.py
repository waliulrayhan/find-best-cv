"""
Configuration file for CV Screening Hybrid Model
Contains all hyperparameters, paths, and model settings
"""

import os
import logging
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "raw_dataset_path": RAW_DATA_DIR / "Resume.csv",
    "processed_dataset_path": PROCESSED_DATA_DIR / "processed_dataset.csv",
    "metadata_path": PROCESSED_DATA_DIR / "metadata.json",
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_state": 42,
    "min_text_length": 50,
    "max_text_length": 2000
}

# Model configurations
BERT_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "hidden_size": 768,
    "dropout": 0.4,  # Increased from 0.3 to reduce overfitting
    "num_layers": 2,
    "freeze_bert": False
}

CNN_CONFIG = {
    "num_filters": [100, 100, 100],
    "filter_sizes": [3, 4, 5],
    "dropout": 0.5,
    "activation": "relu"
}

LSTM_CONFIG = {
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.4,  # Increased from 0.3 to reduce overfitting
    "bidirectional": True
}

TRADITIONAL_ML_CONFIG = {
    "tfidf_max_features": 10000,
    "tfidf_ngram_range": (1, 3),
    "use_word2vec": True,
    "word2vec_dim": 300,
    "use_skill_features": True,
    "use_experience_features": True
}

# Hybrid model configuration
HYBRID_CONFIG = {
    "bert_weight": 0.4,
    "cnn_weight": 0.25,
    "lstm_weight": 0.25,
    "traditional_weight": 0.1,
    "use_attention": True,
    "attention_dim": 128,
    "final_dropout": 0.3  # Increased from 0.2 to reduce overfitting
}

# Training configuration - Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM)
TRAINING_CONFIG = {
    "batch_size": 8,  # Reduced for 4GB VRAM
    "learning_rate": 2e-5,
    "weight_decay": 0.05,  # Increased from 0.01 for stronger L2 regularization
    "max_epochs": 20,
    "patience": 5,  # Reduced from 10 to stop earlier and prevent overfitting
    "min_delta": 0.001,
    "warmup_steps": 500,
    "gradient_clip_value": 1.0,
    "accumulation_steps": 2,  # Simulate batch_size=16 with gradient accumulation
    "use_mixed_precision": True  # Enable for faster training on GPU
}

# Evaluation configuration
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
    "cross_validation_folds": 5,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    "project_name": "cv_screening_hybrid",
    "experiment_name": None,  # Will be generated automatically
    "track_gradients": True,
    "track_parameters": True,
    "track_metrics": True,
    "save_model_checkpoints": True,
    "checkpoint_frequency": 5  # Save every N epochs
}

# Hardware configuration - Optimized for NVIDIA RTX 3050 Laptop GPU
HARDWARE_CONFIG = {
    "use_cuda": True,  # Enable GPU
    "cuda_device": 0,
    "num_workers": 2,  # Reduced for laptop GPU
    "pin_memory": True,  # Faster data transfer to GPU
    "persistent_workers": False  # Changed for laptop stability
}

# Logging configuration  
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_FILE = LOGS_DIR / "training.log"

# Model paths
MODEL_PATHS = {
    "hybrid_model": MODELS_DIR / "hybrid_model.pth",
    "bert_model": MODELS_DIR / "bert_model.pth",
    "cnn_model": MODELS_DIR / "cnn_model.pth",
    "lstm_model": MODELS_DIR / "lstm_model.pth",
    "traditional_model": MODELS_DIR / "traditional_model.pkl",
    "tokenizer": MODELS_DIR / "tokenizer",
    "vectorizers": MODELS_DIR / "vectorizers.pkl",
    "model_metadata": MODELS_DIR / "model_metadata.json"
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 1,
    "upload_max_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": [".pdf", ".docx", ".doc", ".txt"],
    "cors_origins": ["*"]
}