# text_processor.py
import re

# A basic list of stop words to filter out common filler text
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    'can', 'will', 'just', 'should', 'now'
}

def clean_and_tokenize(text):
    """Lowercase text, strip punctuation, and remove basic stop words."""
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using a regular expression
    text = re.sub(r'[^\w\s]', '', text)
    # Split the sentence into individual words
    words = text.split()
    # Retain only words that are not in our stop words list
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return filtered_words
