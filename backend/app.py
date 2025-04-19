from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch
import fitz  # PyMuPDF for PDF text extraction
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# ========== LOAD TF-IDF SUMMARIZATION MODEL ==========
SUMMARIZATION_MODEL_PATH = "./model/text_summarizer/tfidf_summarizer_model_final.pkl"

try:
    def load_model(filename=SUMMARIZATION_MODEL_PATH):
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model['vectorizer'], model['kmeans']
    
    vectorizer, kmeans = load_model()
    print("✅ TF-IDF Summarization model loaded successfully!")
except Exception as e:
    print("❌ Error loading TF-IDF summarization model:", str(e))
    vectorizer, kmeans = None, None

# ========== LOAD AI DETECTION MODEL ==========
AI_DETECTION_MODEL_PATH = "./model/ai_detection"

try:
    ai_tokenizer = RobertaTokenizer.from_pretrained(AI_DETECTION_MODEL_PATH)
    ai_model = AutoModelForSequenceClassification.from_pretrained(AI_DETECTION_MODEL_PATH)
    ai_model.eval()
    print("✅ AI Detection model loaded successfully!")
except Exception as e:
    print("❌ Error loading AI detection model:", str(e))
    ai_model = None

# ========== PDF TEXT EXTRACTION ==========
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    except Exception as e:
        print("❌ PDF extraction error:", str(e))
    return text.strip()

# ========== TF-IDF SUMMARIZATION FUNCTION ==========
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

# ========== SUMMARIZATION ENDPOINT ==========
@app.route("/summarize", methods=["POST"])
def summarize_text_endpoint():
    if vectorizer is None or kmeans is None:
        return jsonify({"error": "TF-IDF Summarization model not loaded"}), 500

    # Handle PDF upload
    if "file" in request.files:
        file = request.files["file"]
        text = extract_text_from_pdf(file)
    else:
        data = request.get_json()
        text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        summary = summarize_text(text, vectorizer, kmeans, summary_ratio=0.25)
        # Join sentences into a single string to match original endpoint's response format
        summary_text = " ".join(summary)
        return jsonify({"summary": summary_text})
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

# ========== AI DETECTION ENDPOINT ==========
@app.route("/detect", methods=["POST"])
def detect_ai():
    if not ai_model:
        return jsonify({"error": "AI Detection model not loaded"}), 500

    # Handle PDF upload
    if "file" in request.files:
        file = request.files["file"]
        text = extract_text_from_pdf(file)
    else:
        data = request.get_json()
        text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    sentences = text.split(". ")
    ai_sentences = []
    total_ai_score = 0

    try:
        for sentence in sentences:
            if not sentence.strip():
                continue

            inputs = ai_tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = ai_model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            ai_score = probabilities[0][1].item() * 100

            total_ai_score += ai_score
            if ai_score > 50:
                ai_sentences.append({"sentence": sentence, "ai_probability": f"{ai_score:.2f}%"})

        avg_ai_score = total_ai_score / len(sentences) if sentences else 0
        result = "AI-Generated" if avg_ai_score > 50 else "Human-Written"

        return jsonify({
            "prediction": result,
            "ai_probability": f"{avg_ai_score:.2f}%",
            "ai_detected_sentences": ai_sentences
        })
    except Exception as e:
        return jsonify({"error": f"AI Detection failed: {str(e)}"}), 500

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)