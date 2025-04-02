import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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

# Test the preprocessing function on a sample CV text
if __name__ == "__main__":
    sample_cv_text = """
    John Doe
    Software Engineer
    
    EXPERIENCE
    Senior Software Engineer at ABC Corp (2018-Present)
    - Developed and maintained web applications using React.js and Node.js
    - Implemented CI/CD pipelines using Jenkins and Docker
    - Collaborated with cross-functional teams to deliver high-quality software
    
    Software Developer at XYZ Inc. (2015-2018)
    - Built RESTful APIs using Python and Flask
    - Worked on database optimization and query performance
    
    EDUCATION
    Master of Computer Science, University of Technology (2013-2015)
    Bachelor of Engineering, State University (2009-2013)
    
    SKILLS
    Programming Languages: Python, JavaScript, Java, C++
    Frameworks: React, Node.js, Flask, Django
    Tools: Git, Docker, Jenkins, AWS
    """
    
    preprocessed_text = preprocess_text(sample_cv_text)
    print("Original Text:")
    print(sample_cv_text)
    print("\nPreprocessed Text:")
    print(preprocessed_text)
    
    # Print some statistics
    original_word_count = len(sample_cv_text.split())
    processed_word_count = len(preprocessed_text.split())
    
    print(f"\nOriginal word count: {original_word_count}")
    print(f"Processed word count: {processed_word_count}")
    print(f"Reduction: {(1 - processed_word_count/original_word_count) * 100:.2f}%") 