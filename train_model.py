"""
Main Training Script for Enhanced Hybrid CV Screening Model
Run this script to train the model and achieve 95% accuracy target
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
    logger.info(f"Target Accuracy: {PERFORMANCE_TARGET['accuracy']*100:.1f}%")
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
        
        # Step 5: Check if target accuracy achieved
        logger.info("\n" + "="*50)
        logger.info("STEP 5: PERFORMANCE ASSESSMENT")
        logger.info("="*50)
        
        test_accuracy = evaluation_report['test_evaluation']['overall_metrics']['accuracy']
        test_f1 = evaluation_report['test_evaluation']['overall_metrics']['f1']
        target_met = evaluation_report['performance_target_met']
        
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"Final Test F1: {test_f1:.4f}")
        logger.info(f"Target Accuracy ({PERFORMANCE_TARGET['accuracy']*100:.1f}%): {'[ACHIEVED]' if target_met else '[NOT ACHIEVED]'}")
        
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
                'test_f1': test_f1,
                'target_accuracy': PERFORMANCE_TARGET['accuracy'],
                'target_achieved': target_met
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
        if target_met:
            logger.info("SUCCESS! HYBRID MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"[ACHIEVED] {test_accuracy*100:.2f}% accuracy (target: {PERFORMANCE_TARGET['accuracy']*100:.1f}%)")
        else:
            logger.info("TRAINING COMPLETED BUT TARGET NOT FULLY ACHIEVED")
            logger.info(f"[RESULT] Achieved {test_accuracy*100:.2f}% accuracy (target: {PERFORMANCE_TARGET['accuracy']*100:.1f}%)")
            logger.info("[INFO] Consider adjusting hyperparameters or increasing training data")
        
        logger.info("="*80)
        
        return target_met
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import pandas as pd
    success = main()
    sys.exit(0 if success else 1)