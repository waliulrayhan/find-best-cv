"""
Enhanced Hybrid Model for CV Screening
Combines BERT, CNN, LSTM, and traditional ML approaches for optimal performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from config.config import *

# Configure logging
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

class CVDataset(Dataset):
    """Dataset for CV-Job matching"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }

class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature fusion"""
    
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        u_it = torch.tanh(self.W(x))  # (batch_size, seq_len, attention_dim)
        a_it = self.u(u_it)  # (batch_size, seq_len, 1)
        a_it = F.softmax(a_it, dim=1)  # (batch_size, seq_len, 1)
        
        attended_x = torch.sum(a_it * x, dim=1)  # (batch_size, hidden_dim)
        return self.dropout(attended_x)

class BERTComponent(nn.Module):
    """BERT-based component for semantic understanding"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config["model_name"])
        
        if config["freeze_bert"]:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"] // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size"] // 2, config["hidden_size"] // 4)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different BERT model outputs
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use CLS token representation from last hidden state
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback: mean pooling of last hidden state
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        features = self.classifier(pooled_output)
        
        return features

class CNNComponent(nn.Module):
    """CNN component for local pattern recognition"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, config: Dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_size)
            for num_filters, filter_size in zip(config["num_filters"], config["filter_sizes"])
        ])
        
        self.dropout = nn.Dropout(config["dropout"])
        self.fc = nn.Linear(sum(config["num_filters"]), sum(config["num_filters"]) // 2)
        
    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, sum(num_filters))
        features = self.dropout(concatenated)
        features = F.relu(self.fc(features))
        
        return features

class LSTMComponent(nn.Module):
    """LSTM component for sequential pattern recognition"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, config: Dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            bidirectional=config["bidirectional"],
            batch_first=True
        )
        
        lstm_output_size = config["hidden_size"] * (2 if config["bidirectional"] else 1)
        self.attention = AttentionLayer(lstm_output_size, HYBRID_CONFIG["attention_dim"])
        self.fc = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_size * 2)
        
        # Use attention to focus on important parts
        if HYBRID_CONFIG["use_attention"]:
            attended_features = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.config["bidirectional"]:
                attended_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                attended_features = hidden[-1]
        
        features = self.dropout(attended_features)
        features = F.relu(self.fc(features))
        
        return features

class TraditionalMLComponent(nn.Module):
    """Traditional ML component using TF-IDF and engineered features"""
    
    def __init__(self, tfidf_dim: int, additional_features_dim: int = 0):
        super().__init__()
        self.tfidf_dim = tfidf_dim
        self.additional_features_dim = additional_features_dim
        
        total_dim = tfidf_dim + additional_features_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(total_dim // 2, total_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(total_dim // 4, total_dim // 8)
        )
        
    def forward(self, tfidf_features, additional_features=None):
        # tfidf_features shape: (batch_size, tfidf_dim)
        if additional_features is not None:
            features = torch.cat([tfidf_features, additional_features], dim=1)
        else:
            features = tfidf_features
        
        features = self.fc_layers(features)
        return features

class EnhancedHybridModel(nn.Module):
    """Enhanced hybrid model combining multiple architectures"""
    
    def __init__(self, num_classes: int, vocab_size: int, tfidf_dim: int):
        super().__init__()
        
        # Initialize tokenizer for BERT
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_CONFIG["model_name"])
        
        # Component models
        self.bert_component = BERTComponent(BERT_CONFIG)
        
        embedding_dim = 300  # Standard embedding dimension
        self.cnn_component = CNNComponent(vocab_size, embedding_dim, CNN_CONFIG)
        self.lstm_component = LSTMComponent(vocab_size, embedding_dim, LSTM_CONFIG)
        self.traditional_component = TraditionalMLComponent(tfidf_dim)
        
        # Feature fusion
        bert_out_dim = BERT_CONFIG["hidden_size"] // 4
        cnn_out_dim = sum(CNN_CONFIG["num_filters"]) // 2
        lstm_out_dim = LSTM_CONFIG["hidden_size"] * (2 if LSTM_CONFIG["bidirectional"] else 1) // 2
        traditional_out_dim = tfidf_dim // 8
        
        total_features = bert_out_dim + cnn_out_dim + lstm_out_dim + traditional_out_dim
        
        # Weighted feature combination
        self.feature_weights = nn.Parameter(torch.tensor([
            HYBRID_CONFIG["bert_weight"],
            HYBRID_CONFIG["cnn_weight"],
            HYBRID_CONFIG["lstm_weight"],
            HYBRID_CONFIG["traditional_weight"]
        ]))
        
        # Attention mechanism for feature fusion
        if HYBRID_CONFIG["use_attention"]:
            self.fusion_attention = AttentionLayer(total_features, HYBRID_CONFIG["attention_dim"])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(HYBRID_CONFIG["final_dropout"]),
            nn.Linear(total_features // 2, total_features // 4),
            nn.ReLU(),
            nn.Dropout(HYBRID_CONFIG["final_dropout"]),
            nn.Linear(total_features // 4, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask, tfidf_features):
        # Get features from each component
        bert_features = self.bert_component(input_ids, attention_mask)
        cnn_features = self.cnn_component(input_ids)
        lstm_features = self.lstm_component(input_ids)
        traditional_features = self.traditional_component(tfidf_features)
        
        # Apply learned weights
        weighted_bert = bert_features * self.feature_weights[0]
        weighted_cnn = cnn_features * self.feature_weights[1]
        weighted_lstm = lstm_features * self.feature_weights[2]
        weighted_traditional = traditional_features * self.feature_weights[3]
        
        # Combine features
        combined_features = torch.cat([
            weighted_bert, weighted_cnn, weighted_lstm, weighted_traditional
        ], dim=1)
        
        # Apply attention if configured
        if HYBRID_CONFIG["use_attention"]:
            # Reshape for attention (add sequence dimension)
            combined_features = combined_features.unsqueeze(1)  # (batch_size, 1, features)
            attended_features = self.fusion_attention(combined_features)
        else:
            attended_features = combined_features
        
        # Final classification
        logits = self.classifier(attended_features)
        
        return logits
    
    def get_feature_weights(self):
        """Get the learned feature weights"""
        weights = F.softmax(self.feature_weights, dim=0)
        return {
            'bert_weight': weights[0].item(),
            'cnn_weight': weights[1].item(),
            'lstm_weight': weights[2].item(),
            'traditional_weight': weights[3].item()
        }
    
    def predict_proba(self, input_ids, attention_mask, tfidf_features):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, tfidf_features)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, input_ids, attention_mask, tfidf_features):
        """Get predictions"""
        probabilities = self.predict_proba(input_ids, attention_mask, tfidf_features)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop