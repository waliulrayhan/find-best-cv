import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str:
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing punctuation and special characters
    3. Tokenizing and removing stopwords
    4. Applying stemming
    
    Args:
        text: The input text to preprocess
        
    Returns:
        Preprocessed text as a string of stemmed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

def vectorize_text(text: str, fit_vectorizer: bool = False):
    """
    Convert preprocessed text into TF-IDF vector
    
    Args:
        text: The preprocessed text to vectorize
        fit_vectorizer: Whether to fit the vectorizer on this text
        
    Returns:
        TF-IDF vector as a list of tuples (word, score) sorted by score
    """
    if fit_vectorizer or not hasattr(tfidf_vectorizer, 'vocabulary_'):
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    else:
        tfidf_matrix = tfidf_vectorizer.transform([text])
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create a list of tuples (word, score) and sort by score
    tfidf_list = sorted(
        [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))],
        key=lambda x: x[1],
        reverse=True
    )
    
    return tfidf_list

def compute_similarity(job_description: str, cv_texts: list[str]) -> list[tuple[int, float]]:
    """
    Compute cosine similarity between job description and multiple CVs
    
    Args:
        job_description: The job description text
        cv_texts: List of CV texts to compare against
        
    Returns:
        List of tuples (cv_index, similarity_score) sorted by score in descending order
    """
    # Preprocess all texts
    processed_jd = preprocess_text(job_description)
    processed_cvs = [preprocess_text(cv) for cv in cv_texts]
    
    # Combine all texts for vectorization
    all_texts = [processed_jd] + processed_cvs
    
    # Fit and transform all texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarity between job description (first document)
    # and all CVs (remaining documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Create list of (index, score) tuples and sort by score
    ranked_indices = [(i, float(score)) for i, score in enumerate(similarities)]
    ranked_indices.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_indices

def get_matching_keywords(jd_text: str, cv_text: str, top_n: int = 10) -> list[str]:
    """
    Extract matching keywords between job description and CV
    
    Args:
        jd_text: The job description text
        cv_text: The CV text
        top_n: Number of top keywords to return
        
    Returns:
        List of matching keywords
    """
    # Preprocess texts
    processed_jd = preprocess_text(jd_text)
    processed_cv = preprocess_text(cv_text)
    
    # Split texts into word sets
    jd_words = set(processed_jd.split())
    cv_words = set(processed_cv.split())
    
    # Find matching words
    matching_words = jd_words.intersection(cv_words)
    
    # Filter out very short words
    filtered_words = [word for word in matching_words if len(word) > 2]
    
    # Return top N matching words
    return filtered_words[:top_n]