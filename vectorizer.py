# vectorizer.py
import math

class CustomTFIDFVectorizer:
    def __init__(self):
        self.vocabulary = []
        self.idf = {}
    def fit_transform(self, tokenized_docs):
        """Builds a global vocabulary and creates a normalized TF-IDF matrix."""
        # 1. Gather all unique words across all documents to create a vocabulary index
        unique_words = set()
        for doc in tokenized_docs:
            unique_words.update(doc)
        self.vocabulary = sorted(list(unique_words))
        
        num_docs = len(tokenized_docs)
        
        # 2. Calculate Inverse Document Frequency (IDF) for every vocabulary word
        for word in self.vocabulary:
            # Count how many documents contain this specific word
            docs_with_word = sum(1 for doc in tokenized_docs if word in doc)
            # Apply standard smooth IDF formula: ln(Total Docs / Docs with Word) + 1
            self.idf[word] = math.log((num_docs) / (1 + docs_with_word)) + 1

        # 3. Compute raw TF-IDF vector matrix
        matrix = []
        for doc in tokenized_docs:
            vector = []
            doc_length = len(doc)
            for word in self.vocabulary:
                if doc_length == 0:
                    vector.append(0.0)
                    continue
                # Term Frequency (TF) = occurrences of word / total words in doc
                tf = doc.count(word) / doc_length
                # TF-IDF calculation
                tfidf_value = tf * self.idf[word]
                vector.append(tfidf_value)
            
            # 4. Normalize the vector to unit length (L2 Normalization)
            # This balances out disparities caused by varying document lengths
            magnitude = math.sqrt(sum(val ** 2 for val in vector))
            if magnitude > 0:
                vector = [val / magnitude for val in vector]
            else:
                vector = [0.0] * len(vector)
                
            matrix.append(vector)
            
        return matrix
