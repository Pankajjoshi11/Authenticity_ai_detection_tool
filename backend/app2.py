from flask import Flask, request, jsonify, render_template
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans

nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

def load_model(filename="./model/text_summarizer/tfidf_summarizer_model_final.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model['vectorizer'], model['kmeans']

def summarize_text(text, vectorizer, kmeans, summary_ratio=0.25):
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    
    X = vectorizer.transform(sentences).toarray()
    num_sentences = max(1, int(len(sentences) * summary_ratio))
    
    if len(kmeans.cluster_centers_) != num_sentences:
        kmeans = KMeans(n_clusters=num_sentences, random_state=0, n_init='auto').fit(X)
    
    summary_sentences = []
    for i in range(num_sentences):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
        closest_index = min(
            cluster_indices,
            key=lambda idx: np.linalg.norm(X[idx] - kmeans.cluster_centers_[i])
        )
        summary_sentences.append((closest_index, sentences[closest_index]))

    summary_sentences.sort()
    return [sent.strip().capitalize() for idx, sent in summary_sentences]

# Load model
vectorizer, kmeans = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        summary = summarize_text(text, vectorizer, kmeans, summary_ratio=0.25)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)