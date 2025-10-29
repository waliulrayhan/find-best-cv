"""
Enhanced Inference Module for Hybrid CV Screening Model
Handles model loading, prediction, and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

from src.models.hybrid_model import EnhancedHybridModel, CVDataset
from src.data.data_processor import EnhancedDataProcessor
from config.config import *

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

class HybridModelPredictor:
    """Enhanced predictor for the hybrid CV screening model"""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the predictor"""
        self.model = None
        self.tokenizer = None
        self.data_processor = None
        self.model_metadata = None
        self.is_loaded = False
        
        if model_path or MODEL_PATHS["hybrid_model"].exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load the trained model and associated components"""
        try:
            logger.info("Loading hybrid model...")
            
            # Load model metadata
            with open(MODEL_PATHS["model_metadata"], 'r') as f:
                self.model_metadata = json.load(f)
            
            # Initialize data processor
            self.data_processor = EnhancedDataProcessor()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["tokenizer"])
            
            # Load vectorizers
            with open(MODEL_PATHS["vectorizers"], 'rb') as f:
                vectorizers = pickle.load(f)
                self.data_processor.tfidf_vectorizer = vectorizers['tfidf_vectorizer']
                self.data_processor.label_encoder = vectorizers['label_encoder']
            
            # Create and load model
            tfidf_dim = self.data_processor.tfidf_vectorizer.transform([""]).shape[1]
            
            self.model = EnhancedHybridModel(
                num_classes=self.model_metadata['num_classes'],
                vocab_size=self.model_metadata['vocab_size'],
                tfidf_dim=tfidf_dim
            )
            
            # Load model state
            model_path = model_path or MODEL_PATHS["hybrid_model"]
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            logger.info(f"Model classes: {self.model_metadata['classes']}")
            logger.info(f"Feature weights: {self.model_metadata['feature_weights']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess text for prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Clean text
        cleaned_text = self.data_processor.clean_text(text)
        
        # Tokenize for BERT
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=BERT_CONFIG["max_length"],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Create TF-IDF features
        tfidf_features = self.data_processor.tfidf_vectorizer.transform([cleaned_text])
        tfidf_tensor = torch.FloatTensor(tfidf_features.toarray()).to(device)
        
        return input_ids, attention_mask, tfidf_tensor
    
    def predict_single(self, text: str, return_probabilities: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
        """Predict category for a single text"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_ids, attention_mask, tfidf_features = self.preprocess_text(text)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, tfidf_features)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        # Convert prediction to label
        predicted_class_idx = prediction.item()
        predicted_class = self.data_processor.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        if return_probabilities:
            prob_dict = {}
            for i, class_name in enumerate(self.model_metadata['classes']):
                prob_dict[class_name] = probabilities[0][i].item()
            return predicted_class, prob_dict
        
        return predicted_class
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = False) -> Union[List[str], Tuple[List[str], List[Dict[str, float]]]]:
        """Predict categories for multiple texts"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = []
        probabilities_list = []
        
        # Process texts in batches
        batch_size = TRAINING_CONFIG["batch_size"]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            batch_input_ids = []
            batch_attention_masks = []
            batch_tfidf_features = []
            
            for text in batch_texts:
                input_ids, attention_mask, tfidf_features = self.preprocess_text(text)
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_tfidf_features.append(tfidf_features)
            
            # Stack tensors
            batch_input_ids = torch.cat(batch_input_ids, dim=0)
            batch_attention_masks = torch.cat(batch_attention_masks, dim=0)
            batch_tfidf_features = torch.cat(batch_tfidf_features, dim=0)
            
            # Predict
            with torch.no_grad():
                logits = self.model(batch_input_ids, batch_attention_masks, batch_tfidf_features)
                batch_probabilities = F.softmax(logits, dim=1)
                batch_predictions = torch.argmax(batch_probabilities, dim=1)
            
            # Convert to labels
            for j in range(len(batch_texts)):
                pred_idx = batch_predictions[j].item()
                predicted_class = self.data_processor.label_encoder.inverse_transform([pred_idx])[0]
                predictions.append(predicted_class)
                
                if return_probabilities:
                    prob_dict = {}
                    for k, class_name in enumerate(self.model_metadata['classes']):
                        prob_dict[class_name] = batch_probabilities[j][k].item()
                    probabilities_list.append(prob_dict)
        
        if return_probabilities:
            return predictions, probabilities_list
        
        return predictions
    
    def evaluate_model(self, test_texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        predictions, probabilities = self.predict_batch(test_texts, return_probabilities=True)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Get unique labels in the predictions and true labels (these are string class names)
        unique_labels = sorted(list(set(true_labels) | set(predictions)))
        
        # Classification report - only use labels that appear in the data
        class_report = classification_report(
            true_labels, predictions, 
            labels=unique_labels,
            target_names=unique_labels,  # Use the labels directly as they are already class names
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix - only use labels that appear in the data
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.model_metadata['classes']):
            if class_name in class_report:
                per_class_metrics[class_name] = {
                    'precision': class_report[class_name]['precision'],
                    'recall': class_report[class_name]['recall'],
                    'f1': class_report[class_name]['f1-score'],
                    'support': class_report[class_name]['support']
                }
        
        evaluation_results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities,
            'target_achieved': accuracy >= PERFORMANCE_TARGET['accuracy']
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {f1:.4f}")
        logger.info(f"Target achieved: {evaluation_results['target_achieved']}")
        
        return evaluation_results
    
    def match_cv_to_job(self, cv_text: str, job_description: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Match a CV to a job description"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Predict category for CV
        cv_category, cv_probabilities = self.predict_single(cv_text, return_probabilities=True)
        
        # Predict category for job description
        job_category, job_probabilities = self.predict_single(job_description, return_probabilities=True)
        
        # Calculate similarity based on category probabilities
        similarity_score = 0.0
        category_matches = {}
        
        for category in self.model_metadata['classes']:
            cv_prob = cv_probabilities.get(category, 0.0)
            job_prob = job_probabilities.get(category, 0.0)
            
            # Calculate cosine similarity for this category
            category_sim = min(cv_prob, job_prob) / max(max(cv_prob, job_prob), 1e-8)
            category_matches[category] = category_sim
            similarity_score += category_sim
        
        # Normalize similarity score
        similarity_score /= len(self.model_metadata['classes'])
        
        # Determine match
        is_match = similarity_score >= threshold
        
        # Extract skills for additional analysis
        cv_features = self.data_processor.create_features(cv_text)
        job_features = self.data_processor.create_features(job_description)
        
        # Skill matching
        skill_matches = {}
        for skill_type in ['programming', 'frameworks', 'databases', 'cloud', 'tools']:
            if f'{skill_type}_skills' in cv_features and f'{skill_type}_skills' in job_features:
                cv_skills = set(cv_features[f'{skill_type}_skills'])
                job_skills = set(job_features[f'{skill_type}_skills'])
                
                if job_skills:
                    skill_match_ratio = len(cv_skills.intersection(job_skills)) / len(job_skills)
                    skill_matches[skill_type] = {
                        'ratio': skill_match_ratio,
                        'cv_skills': list(cv_skills),
                        'job_skills': list(job_skills),
                        'matched_skills': list(cv_skills.intersection(job_skills))
                    }
        
        match_result = {
            'is_match': is_match,
            'similarity_score': similarity_score,
            'threshold': threshold,
            'cv_category': cv_category,
            'job_category': job_category,
            'cv_probabilities': cv_probabilities,
            'job_probabilities': job_probabilities,
            'category_similarities': category_matches,
            'skill_matches': skill_matches,
            'cv_experience_years': cv_features.get('experience_years', 0),
            'job_requirements_met': similarity_score >= threshold
        }
        
        return match_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            'model_metadata': self.model_metadata,
            'feature_weights': self.model.get_feature_weights() if hasattr(self.model, 'get_feature_weights') else None,
            'device': str(device),
            'is_loaded': self.is_loaded,
            'classes': self.model_metadata['classes'],
            'num_classes': self.model_metadata['num_classes']
        }

class ModelEvaluator:
    """Comprehensive model evaluation utility"""
    
    def __init__(self, predictor: HybridModelPredictor):
        self.predictor = predictor
    
    def cross_validate(self, texts: List[str], labels: List[str], k_folds: int = 5) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import StratifiedKFold
        
        logger.info(f"Performing {k_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=DATASET_CONFIG["random_state"])
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"Fold {fold + 1}/{k_folds}")
            
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Evaluate on validation set
            fold_result = self.predictor.evaluate_model(val_texts, val_labels)
            fold_results.append(fold_result['overall_metrics'])
        
        # Calculate mean and std for each metric
        cv_results = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
            cv_results[f'{metric}_values'] = values
        
        logger.info("Cross-validation complete:")
        for metric in metrics:
            mean_val = cv_results[f'{metric}_mean']
            std_val = cv_results[f'{metric}_std']
            logger.info(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_results
    
    def generate_evaluation_report(self, test_texts: List[str], true_labels: List[str], 
                                 output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Evaluate model
        evaluation_results = self.predictor.evaluate_model(test_texts, true_labels)
        
        # Perform cross-validation
        cv_results = self.cross_validate(test_texts, true_labels)
        
        # Combine results
        full_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': self.predictor.get_model_info(),
            'test_evaluation': evaluation_results,
            'cross_validation': cv_results,
            'performance_target_met': evaluation_results['target_achieved']
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return full_report

def main():
    """Example usage of the predictor"""
    # Initialize predictor
    predictor = HybridModelPredictor()
    
    if not predictor.is_loaded:
        logger.error("Could not load model. Make sure the model has been trained.")
        return
    
    # Example prediction
    sample_cv = """
    Software Engineer with 5 years of experience in Python, Java, and machine learning.
    Worked on web applications using Django and React. Experience with AWS and Docker.
    Master's degree in Computer Science.
    """
    
    sample_job = """
    Looking for a Senior Software Engineer with Python and machine learning experience.
    Must have experience with web frameworks like Django or Flask.
    Cloud experience with AWS preferred.
    """
    
    # Single prediction
    prediction = predictor.predict_single(sample_cv, return_probabilities=True)
    logger.info(f"CV Category: {prediction[0]}")
    logger.info(f"Probabilities: {prediction[1]}")
    
    # CV-Job matching
    match_result = predictor.match_cv_to_job(sample_cv, sample_job)
    logger.info(f"Match Score: {match_result['similarity_score']:.4f}")
    logger.info(f"Is Match: {match_result['is_match']}")

if __name__ == "__main__":
    main()