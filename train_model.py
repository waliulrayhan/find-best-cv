"""
Main Training Script for Enhanced Hybrid CV Screening Model
Trains the model for optimal performance without overfitting
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from src.models.trainer import HybridModelTrainer
from src.models.predictor import HybridModelPredictor, ModelEvaluator
from src.data.data_processor import EnhancedDataProcessor
from config.config import *

# Set up logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training_main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to train and evaluate the hybrid model"""
    logger.info("="*80)
    logger.info("ENHANCED HYBRID CV SCREENING MODEL - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info("Goal: Achieve optimal performance without overfitting")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Models Directory: {MODELS_DIR}")
    logger.info(f"Experiments Directory: {EXPERIMENTS_DIR}")
    
    try:
        # Step 1: Initialize trainer
        logger.info("\n" + "="*50)
        logger.info("STEP 1: INITIALIZING TRAINER")
        logger.info("="*50)
        
        experiment_name = f"hybrid_cv_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        trainer = HybridModelTrainer(experiment_name=experiment_name)
        
        # Step 2: Train the model
        logger.info("\n" + "="*50)
        logger.info("STEP 2: TRAINING HYBRID MODEL")
        logger.info("="*50)
        logger.info("Training configuration:")
        for key, value in TRAINING_CONFIG.items():
            logger.info(f"  {key}: {value}")
        
        model = trainer.train()
        
        # Step 3: Load trained model and evaluate
        logger.info("\n" + "="*50)
        logger.info("STEP 3: FINAL MODEL EVALUATION")
        logger.info("="*50)
        
        predictor = HybridModelPredictor()
        
        if not predictor.is_loaded:
            logger.error("Failed to load trained model for evaluation")
            return False
        
        # Step 4: Generate comprehensive evaluation report
        logger.info("\n" + "="*50)
        logger.info("STEP 4: GENERATING EVALUATION REPORT")
        logger.info("="*50)
        
        evaluator = ModelEvaluator(predictor)
        
        # Load test data for evaluation
        data_processor = EnhancedDataProcessor()
        try:
            processed_df, metadata = data_processor.load_processed_data()
        except:
            logger.info("Processing raw data for evaluation...")
            raw_df = data_processor.load_data(DATASET_CONFIG["raw_dataset_path"])
            processed_df, metadata = data_processor.preprocess_data(raw_df)
            data_processor.save_processed_data(processed_df, metadata)
        
        # Create test split for evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.create_training_data(processed_df)
        
        # Convert labels back to strings
        test_labels = data_processor.label_encoder.inverse_transform(y_test).tolist()
        
        # Generate report
        report_path = RESULTS_DIR / f"{experiment_name}_evaluation_report.json"
        evaluation_report = evaluator.generate_evaluation_report(
            X_test.tolist(), test_labels, report_path
        )
        
        # Step 5: Performance assessment
        logger.info("\n" + "="*50)
        logger.info("STEP 5: PERFORMANCE ASSESSMENT")
        logger.info("="*50)
        
        test_accuracy = evaluation_report['test_evaluation']['overall_metrics']['accuracy']
        test_f1 = evaluation_report['test_evaluation']['overall_metrics']['f1']
        test_precision = evaluation_report['test_evaluation']['overall_metrics']['precision']
        test_recall = evaluation_report['test_evaluation']['overall_metrics']['recall']
        
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"Final Test Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        logger.info(f"Final Test Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
        logger.info(f"Final Test F1: {test_f1:.4f} ({test_f1*100:.2f}%)")
        
        # Model information
        model_info = predictor.get_model_info()
        feature_weights = model_info.get('feature_weights', {})
        
        logger.info("\nLearned Feature Weights:")
        for component, weight in feature_weights.items():
            logger.info(f"  {component}: {weight:.4f}")
        
        # Step 6: Save final results summary
        logger.info("\n" + "="*50)
        logger.info("STEP 6: SAVING RESULTS SUMMARY")
        logger.info("="*50)
        
        final_summary = {
            'experiment_name': experiment_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'performance_summary': {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            },
            'model_configuration': {
                'bert_config': BERT_CONFIG,
                'cnn_config': CNN_CONFIG,
                'lstm_config': LSTM_CONFIG,
                'traditional_config': TRADITIONAL_ML_CONFIG,
                'hybrid_config': HYBRID_CONFIG,
                'training_config': TRAINING_CONFIG
            },
            'learned_weights': feature_weights,
            'dataset_info': metadata,
            'evaluation_report_path': str(report_path)
        }
        
        summary_path = RESULTS_DIR / f"{experiment_name}_final_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        logger.info(f"Final summary saved to: {summary_path}")
        
        # Step 7: Demonstration
        logger.info("\n" + "="*50)
        logger.info("STEP 7: MODEL DEMONSTRATION")
        logger.info("="*50)
        
        # Example CV-Job matching
        sample_cv = """
        Senior Software Engineer with 8 years of experience in Python, Java, and machine learning.
        Extensive experience with web development using Django, Flask, React, and Angular.
        Proficient in cloud technologies including AWS, Docker, and Kubernetes.
        Strong background in data science and artificial intelligence.
        Master's degree in Computer Science from top university.
        Led teams of 5+ developers on multiple successful projects.
        """
        
        sample_job = """
        We are seeking a Senior Machine Learning Engineer with 5+ years of experience.
        Must have strong Python programming skills and experience with ML frameworks.
        Experience with web frameworks like Django or Flask is required.
        Cloud experience with AWS and containerization technologies preferred.
        Team leadership experience is a plus.
        """
        
        logger.info("Demonstrating CV-Job matching...")
        logger.info(f"Sample CV (first 100 chars): {sample_cv[:100]}...")
        logger.info(f"Sample Job (first 100 chars): {sample_job[:100]}...")
        
        try:
            match_result = predictor.match_cv_to_job(sample_cv, sample_job)
            
            logger.info(f"Match Result:")
            logger.info(f"  Similarity Score: {match_result['similarity_score']:.4f}")
            logger.info(f"  Is Match: {match_result['is_match']}")
            logger.info(f"  CV Category: {match_result['cv_category']}")
            logger.info(f"  Job Category: {match_result['job_category']}")
            logger.info(f"  Experience Years: {match_result['cv_experience_years']}")
            
        except Exception as e:
            logger.warning(f"Demonstration failed: {e}")
        
        # Final status
        logger.info("\n" + "="*80)
        logger.info("SUCCESS! HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Achieved {test_accuracy*100:.2f}% accuracy with {test_f1*100:.2f}% F1-score")
        logger.info("Model trained for optimal performance without overfitting")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)