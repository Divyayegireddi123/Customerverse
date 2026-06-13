# main.py
from text_processor import clean_and_tokenize
from vectorizer import CustomTFIDFVectorizer
from clustering import AgglomerativeClusteringFromScratch

def main():
    # 1. Define a raw test collection of text sentences
    raw_corpus = [
        "The central bank lowered basic interest rates to stimulate financial markets.",
        "Traders on Wall Street bought corporate stocks after earnings climbed.",
        "Astrophysicists discovered a remote planetary galaxy using satellite optics.",
        "Deep space telescopes captured images of expanding stellar nebulas.",
        "Medical researchers isolated structural genetic mutations tied to diseases.",
        "Modern gene sequencing helps laboratory biologists study cellular tissue growth."
    ]
    
    print("Step 1: Raw text collected. Preprocessing text documents...")
    # 2. Preprocess text through cleaning script
    tokenized_corpus = [clean_and_tokenize(doc) for doc in raw_corpus]
    for idx, tokens in enumerate(tokenized_corpus):
        print(f" Doc {idx} tokens: {tokens}")
        
    print("\nStep 2: Converting text tokens into normalized TF-IDF Vectors...")
    # 3. Convert clean word tokens to vectors
    vectorizer = CustomTFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_corpus)
    print(f" Generated matrix shapes: {len(tfidf_matrix)} docs x {len(vectorizer.vocabulary)} unique words.")
    
    print("\nStep 3: Executing hierarchical Agglomerative clustering...")
    # 4. Cluster the vectors using our scratch algorithm
    engine = AgglomerativeClusteringFromScratch()
    merge_history = engine.fit(tfidf_matrix)
    
    # 5. Display the final tree structure
    engine.display_tree(total_documents=len(raw_corpus))

if __name__ == "__main__":
    main()
