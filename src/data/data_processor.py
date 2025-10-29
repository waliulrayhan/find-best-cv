"""
Enhanced Data Processor for CV Screening
Handles data loading, preprocessing, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer

from config.config import *

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    logging.warning("spaCy not available or English model not found. Some features will be limited.")
    nlp = None

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

class EnhancedDataProcessor:
    """Enhanced data processor with advanced feature extraction"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.skill_patterns = self._load_skill_patterns()
        self.processed_data = {}
        
    def _load_skill_patterns(self) -> Dict[str, List[str]]:
        """Load skill patterns for different categories"""
        skills = {
            "programming": [
                "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift",
                "kotlin", "scala", "rust", "typescript", "r", "matlab", "sql", "html", "css"
            ],
            "frameworks": [
                "react", "angular", "vue", "django", "flask", "spring", "node.js", "express",
                "laravel", "rails", "asp.net", "tensorflow", "pytorch", "keras", "scikit-learn"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis", "cassandra",
                "elasticsearch", "neo4j", "influxdb", "dynamodb"
            ],
            "cloud": [
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
                "ansible", "vagrant", "heroku", "digitalocean"
            ],
            "tools": [
                "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
                "trello", "asana", "figma", "sketch", "photoshop", "illustrator"
            ],
            "soft_skills": [
                "leadership", "communication", "teamwork", "problem solving", "creativity",
                "critical thinking", "adaptability", "time management", "project management"
            ]
        }
        return skills
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\\b\\d{3}-\\d{3}-\\d{4}\\b|\\b\\d{10}\\b|\\b\\(\\d{3}\\)\\s*\\d{3}-\\d{4}\\b', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^a-zA-Z0-9\\s\\+\\#\\.]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text using pattern matching and NER"""
        text_lower = text.lower()
        extracted_skills = {}
        
        # Pattern-based skill extraction
        for category, skills in self.skill_patterns.items():
            found_skills = []
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
            extracted_skills[category] = found_skills
        
        # NER-based skill extraction using spaCy (if available)
        if nlp:
            doc = nlp(text)
            ner_skills = []
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE"]:
                    ner_skills.append(ent.text.lower())
            extracted_skills["ner_skills"] = ner_skills
        
        return extracted_skills
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience information from text"""
        experience_info = {
            "years": 0,
            "experience_keywords": [],
            "job_titles": []
        }
        
        # Extract years of experience
        year_patterns = [
            r'(\\d+)\\s*(?:\\+)?\\s*years?\\s*(?:of\\s*)?(?:experience|exp)',
            r'(\\d+)\\s*(?:\\+)?\\s*yrs?\\s*(?:of\\s*)?(?:experience|exp)',
            r'(?:experience|exp)\\s*(?:of\\s*)?(\\d+)\\s*(?:\\+)?\\s*years?',
            r'(?:experience|exp)\\s*(?:of\\s*)?(\\d+)\\s*(?:\\+)?\\s*yrs?'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    years = max([int(match) for match in matches])
                    experience_info["years"] = years
                    break
                except ValueError:
                    continue
        
        # Extract experience keywords
        exp_keywords = [
            "senior", "junior", "lead", "principal", "manager", "director",
            "architect", "consultant", "specialist", "expert", "developer"
        ]
        
        for keyword in exp_keywords:
            if keyword in text.lower():
                experience_info["experience_keywords"].append(keyword)
        
        # Extract job titles using NER (if available)
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" or "engineer" in ent.text.lower() or "developer" in ent.text.lower():
                    experience_info["job_titles"].append(ent.text)
        
        return experience_info
    
    def create_features(self, text: str) -> Dict[str, Any]:
        """Create comprehensive features from text"""
        features = {}
        
        # Basic text features
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["avg_word_length"] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Skill features
        skills = self.extract_skills(text)
        for category, skill_list in skills.items():
            features[f"{category}_count"] = len(skill_list)
            features[f"{category}_skills"] = skill_list
        
        # Experience features
        experience = self.extract_experience(text)
        features["experience_years"] = experience["years"]
        features["experience_level_count"] = len(experience["experience_keywords"])
        
        # Education indicators
        education_keywords = ["degree", "bachelor", "master", "phd", "diploma", "certificate", "university", "college"]
        features["education_mentions"] = sum(1 for keyword in education_keywords if keyword in text.lower())
        
        # Professional indicators
        professional_keywords = ["project", "team", "led", "managed", "developed", "implemented", "designed"]
        features["professional_mentions"] = sum(1 for keyword in professional_keywords if keyword in text.lower())
        
        return features
    
    def load_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load the dataset"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(dataset_path, encoding=encoding)
                    logger.info(f"Dataset loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode dataset with any encoding")
            
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess the dataset"""
        logger.info("Starting data preprocessing...")
        
        # Identify text columns (assuming the dataset has resume text and category columns)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it's a text column (not category)
                avg_length = df[col].str.len().mean()
                if avg_length > 100:  # Assume text columns have longer content
                    text_columns.append(col)
        
        logger.info(f"Identified text columns: {text_columns}")
        
        # Clean text columns
        for col in text_columns:
            logger.info(f"Cleaning {col}...")
            df[col] = df[col].apply(self.clean_text)
        
        # Filter out rows with very short text
        if text_columns:
            main_text_col = text_columns[0]  # Assume first text column is main content
            initial_len = len(df)
            df = df[df[main_text_col].str.len() >= DATASET_CONFIG["min_text_length"]]
            logger.info(f"Filtered out {initial_len - len(df)} rows with short text")
        
        # Create features for each text entry
        logger.info("Extracting features...")
        feature_list = []
        
        for idx, row in df.iterrows():
            if text_columns:
                # Combine all text columns
                combined_text = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                features = self.create_features(combined_text)
                features['text'] = combined_text
                features['original_index'] = idx
                
                # Add category if available
                category_cols = [col for col in df.columns if col.lower() in ['category', 'job_role', 'position', 'class']]
                if category_cols:
                    features['category'] = row[category_cols[0]]
                
                feature_list.append(features)
        
        # Create processed dataframe
        processed_df = pd.DataFrame(feature_list)
        
        # Handle missing values
        processed_df = processed_df.fillna(0)
        
        # Create metadata
        metadata = {
            "total_samples": len(processed_df),
            "text_columns": text_columns,
            "feature_columns": [col for col in processed_df.columns if col not in ['text', 'original_index', 'category']],
            "categories": processed_df['category'].unique().tolist() if 'category' in processed_df.columns else [],
            "preprocessing_config": DATASET_CONFIG
        }
        
        logger.info(f"Preprocessing complete. Processed {len(processed_df)} samples")
        
        return processed_df, metadata
    
    def create_training_data(self, processed_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create training, validation, and test splits"""
        logger.info("Creating training data splits...")
        
        # Prepare features and labels
        X_text = processed_df['text'].values
        
        if 'category' in processed_df.columns:
            # Initialize label encoder if not already done
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(processed_df['category'])
            else:
                y = self.label_encoder.transform(processed_df['category'])
        else:
            # Create dummy labels for unsupervised case
            logger.warning("No category column found. Creating dummy labels.")
            y = np.zeros(len(processed_df))
        
        # Create train/val/test splits
        X_train_text, X_temp_text, y_train, y_temp = train_test_split(
            X_text, y, 
            test_size=(DATASET_CONFIG["val_split"] + DATASET_CONFIG["test_split"]),
            random_state=DATASET_CONFIG["random_state"],
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Split remaining data into validation and test
        val_ratio = DATASET_CONFIG["val_split"] / (DATASET_CONFIG["val_split"] + DATASET_CONFIG["test_split"])
        X_val_text, X_test_text, y_val, y_test = train_test_split(
            X_temp_text, y_temp,
            test_size=(1 - val_ratio),
            random_state=DATASET_CONFIG["random_state"],
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        logger.info(f"Training samples: {len(X_train_text)}")
        logger.info(f"Validation samples: {len(X_val_text)}")
        logger.info(f"Test samples: {len(X_test_text)}")
        
        return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test
    
    def prepare_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[Any, Any, Any]:
        """Prepare additional features (TF-IDF, etc.)"""
        logger.info("Preparing traditional ML features...")
        
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=TRADITIONAL_ML_CONFIG["tfidf_max_features"],
                ngram_range=TRADITIONAL_ML_CONFIG["tfidf_ngram_range"],
                stop_words='english',
                lowercase=True
            )
            X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        else:
            X_train_tfidf = self.tfidf_vectorizer.transform(X_train)
        
        X_val_tfidf = self.tfidf_vectorizer.transform(X_val)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def save_processed_data(self, processed_df: pd.DataFrame, metadata: Dict[str, Any]):
        """Save processed data and metadata"""
        logger.info("Saving processed data...")
        
        # Save processed dataframe
        processed_df.to_csv(DATASET_CONFIG["processed_dataset_path"], index=False)
        
        # Save metadata
        with open(DATASET_CONFIG["metadata_path"], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save fitted transformers
        transformers = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoder': self.label_encoder
        }
        
        with open(PROCESSED_DATA_DIR / "transformers.pkl", 'wb') as f:
            pickle.dump(transformers, f)
        
        logger.info("Processed data saved successfully")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load previously processed data"""
        logger.info("Loading processed data...")
        
        processed_df = pd.read_csv(DATASET_CONFIG["processed_dataset_path"])
        
        with open(DATASET_CONFIG["metadata_path"], 'r') as f:
            metadata = json.load(f)
        
        # Load transformers
        with open(PROCESSED_DATA_DIR / "transformers.pkl", 'rb') as f:
            transformers = pickle.load(f)
            self.tfidf_vectorizer = transformers['tfidf_vectorizer']
            self.label_encoder = transformers['label_encoder']
        
        logger.info("Processed data loaded successfully")
        
        return processed_df, metadata