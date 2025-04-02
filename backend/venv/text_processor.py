import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

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