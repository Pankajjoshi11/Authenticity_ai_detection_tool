import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK resources
# nltk.download('punkt')

# Function to load the TF-IDF + KMeans model
def load_tfidf_model(model_path="tfidf_summarizer_model_final.pkl"):
    try:
        with open(model_path, "rb") as f:
            tfidf_vectorizer, kmeans_model = pickle.load(f)
        print("✅ TF-IDF Summarizer model loaded successfully!")
        return tfidf_vectorizer, kmeans_model
    except Exception as e:
        print(f"❌ Failed to load TF-IDF model: {str(e)}")
        return None, None

# Function to summarize text using the loaded TF-IDF model
def summarize_with_tfidf_model(text, vectorizer, model):
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    try:
        # Convert sentences into a matrix of TF-IDF features
        X = vectorizer.transform(sentences).toarray()

        # Get the cluster centers from the KMeans model
        cluster_centers = model.cluster_centers_
        num_clusters = model.n_clusters

        # Select the sentence closest to each cluster center
        summary_sentences = []
        for i in range(num_clusters):
            cluster_indices = np.where(model.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
            closest_index = min(
                cluster_indices,
                key=lambda idx: np.linalg.norm(X[idx] - cluster_centers[i])
            )
            summary_sentences.append((closest_index, sentences[closest_index]))

        # Sort the summary sentences by their original order in the text
        summary_sentences.sort()
        return [sent.strip().capitalize() for idx, sent in summary_sentences]
    except Exception as e:
        print(f"❌ TF-IDF summarization failed: {str(e)}")
        return []
