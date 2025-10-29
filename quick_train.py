"""
Quick Start Training Script for CV Screening Model
Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM)
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("="*80)
    logger.info("CHECKING PREREQUISITES")
    logger.info("="*80)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is not available!")
        logger.error("Please install PyTorch with CUDA support:")
        logger.error("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    logger.info(f"✅ CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"✅ CUDA Version: {torch.version.cuda}")
    
    # Check data
    from config.config import DATASET_CONFIG
    if not DATASET_CONFIG["raw_dataset_path"].exists():
        logger.error(f"❌ Dataset not found at: {DATASET_CONFIG['raw_dataset_path']}")
        logger.error("Please place Resume.csv in data/raw/ directory")
        return False
    
    logger.info(f"✅ Dataset found: {DATASET_CONFIG['raw_dataset_path']}")
    
    # Check dependencies
    try:
        import transformers
        import sklearn
        import nltk
        logger.info("✅ All dependencies installed")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    return True

def quick_train():
    """Quick training with optimal settings for RTX 3050"""
    logger.info("\n" + "="*80)
    logger.info("STARTING QUICK TRAINING")
    logger.info("="*80)
    
    # Import after prerequisites check
    from src.models.trainer import HybridModelTrainer
    from config.config import TRAINING_CONFIG, HARDWARE_CONFIG
    import pandas as pd
    
    # Display configuration
    logger.info("\n📋 Training Configuration:")
    logger.info(f"   Batch Size: {TRAINING_CONFIG['batch_size']}")
    logger.info(f"   Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    logger.info(f"   Max Epochs: {TRAINING_CONFIG['max_epochs']}")
    logger.info(f"   Mixed Precision: {TRAINING_CONFIG['use_mixed_precision']}")
    logger.info(f"   GPU Enabled: {HARDWARE_CONFIG['use_cuda']}")
    
    # Create experiment name
    experiment_name = f"gpu_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize trainer
    logger.info("\n🔧 Initializing trainer...")
    trainer = HybridModelTrainer(experiment_name=experiment_name)
    
    # Load data
    logger.info("\n📊 Loading and processing data...")
    trainer.load_data()
    
    # Prepare dataloaders
    logger.info("\n⚙️ Preparing dataloaders...")
    trainer.prepare_dataloaders()
    
    # Build model
    logger.info("\n🏗️ Building hybrid model...")
    trainer.build_model()
    
    # Display GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        logger.info(f"   GPU Memory Allocated: {allocated:.2f} GB")
    
    # Train model
    logger.info("\n🚀 Starting training on GPU...")
    logger.info("   This may take 30-60 minutes depending on your GPU...")
    
    try:
        results = trainer.train()
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"\n📊 Final Results:")
        logger.info(f"   Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"   Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"   Test Precision: {results['test_precision']:.4f}")
        logger.info(f"   Test Recall: {results['test_recall']:.4f}")
        logger.info(f"   Test F1-Score: {results['test_f1']:.4f}")
        
        logger.info(f"\n💾 Model saved to: {trainer.save_paths['model']}")
        logger.info(f"📈 Results saved to: {trainer.tracker.results_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.error("If you see CUDA Out of Memory error:")
        logger.error("  1. Reduce batch_size in config/config.py (try 4 or 6)")
        logger.error("  2. Increase accumulation_steps to simulate larger batches")
        return False

def main():
    """Main entry point"""
    logger.info("="*80)
    logger.info("CV SCREENING MODEL - QUICK START TRAINING")
    logger.info("Optimized for NVIDIA RTX 3050 Laptop GPU")
    logger.info("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run training
    success = quick_train()
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("🎉 ALL DONE!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  1. Test the model: python -c \"from src.models.predictor import HybridModelPredictor; p = HybridModelPredictor()\"")
        logger.info("  2. Start the API: cd backend && python main.py")
        logger.info("  3. Start the frontend: cd frontend && npm run dev")
        sys.exit(0)
    else:
        logger.error("\n❌ Training failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
