from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import fitz  # PyMuPDF for PDF text extraction
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

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

# ========== LOAD GRAMMAR CHECK MODELS ==========
LOGISTIC_MODEL_PATH = "./model/grammar/grammar_model_lr.pkl"
VECTORIZER_PATH = "./model/grammar/grammar_vectorizer.pkl"

try:
    if not os.path.exists(LOGISTIC_MODEL_PATH):
        raise FileNotFoundError(f"Logistic Regression model file not found at {LOGISTIC_MODEL_PATH}")
    with open(LOGISTIC_MODEL_PATH, "rb") as f:
        grammar_model = pickle.load(f)
    print("✅ Logistic Regression model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Logistic Regression model: {str(e)}")
    print(f"File path checked: {LOGISTIC_MODEL_PATH}")
    print(f"File exists: {os.path.exists(LOGISTIC_MODEL_PATH)}")
    grammar_model = None

try:
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"TfidfVectorizer file not found at {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "rb") as f:
        grammar_vectorizer = pickle.load(f)
    # Verify vectorizer is fitted
    if not hasattr(grammar_vectorizer, 'vocabulary_'):
        raise ValueError("Loaded TfidfVectorizer is not fitted")
    print("✅ TfidfVectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TfidfVectorizer: {str(e)}")
    print(f"File path checked: {VECTORIZER_PATH}")
    print(f"File exists: {os.path.exists(VECTORIZER_PATH)}")
    grammar_vectorizer = None

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

# ========== BASIC RULE-BASED GRAMMAR CHECKS ==========
def is_grammatically_incorrect(sentence):
    # Check for 'dont' instead of 'don't'
    if re.search(r"\b(?<!\w)dont(?!\w)\b", sentence):
        return True
    # Check for subject-verb agreement errors (e.g., "he don't" instead of "he doesn't")
    if re.search(r"\b(?:he|she|it|Rahul|John|Mary)\b\s+(?:dont|don't|doesn't|isn't|are|were)\b", sentence):
        return True
    # Check for incorrect reflexive pronouns like 'myself' when they should be 'I' or 'me'
    if re.search(r"\bmyself\b", sentence) and not re.search(r"Rahul and myself", sentence):
        return True
    # Check for 'goes' instead of 'go'
    if re.search(r"\bgoes\b", sentence) and not re.search(r"can't go", sentence):
        return True
    # Check for incorrect use of plural verbs (e.g., "loves" instead of "love")
    if re.search(r"\bloves\b", sentence) and not re.search(r"\bthey\b", sentence):
        return True
    # Check for subject-verb agreement (plural vs singular)
    if re.search(r"\bwe\b\s+has\b", sentence):
        return True
    # Check for incorrect verb form (e.g., "can't goes" instead of "can't go")
    if re.search(r"\bcan't goes\b", sentence):
        return True
    # Check for "don't" with singular subjects where "doesn't" should be used (e.g., "he don't")
    if re.search(r"\bhe don't\b", sentence):
        return True
    # Check for "She don't" (incorrect subject-verb agreement)
    if re.search(r"\bShe don't\b", sentence):
        return True
    # Check for "Rahul and myself" (incorrect reflexive pronoun usage)
    if re.search(r"Rahul and myself", sentence):
        return True
    return False

# ========== GRAMMAR CHECK FUNCTION ==========
def check_grammar(text, vectorizer, model):
    sentences = sent_tokenize(text)
    incorrect_sentences = []
    ml_errors = []

    for sentence in sentences:
        # Remove leading/trailing spaces and check for grammar
        sentence = sentence.strip()
        if not sentence:
            continue

        # Rule-based grammar check
        if is_grammatically_incorrect(sentence):
            incorrect_sentences.append(sentence)
        else:
            # Machine learning-based grammar check
            try:
                if not hasattr(vectorizer, 'vocabulary_'):
                    raise ValueError("TfidfVectorizer is not fitted")
                sentence_vec = vectorizer.transform([sentence])
                prediction = model.predict(sentence_vec)
                if prediction == 0:  # 0 means grammatically incorrect
                    incorrect_sentences.append(sentence)
            except Exception as e:
                ml_errors.append(f"ML prediction error for sentence '{sentence}': {str(e)}")

    return incorrect_sentences, ml_errors

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

# ========== GRAMMAR CHECK ENDPOINT ==========
@app.route("/check-grammar", methods=["POST"])
def check_grammar_endpoint():
    if grammar_model is None or grammar_vectorizer is None:
        return jsonify({"error": "Grammar check models not loaded"}), 500

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
        incorrect_sentences, ml_errors = check_grammar(text, grammar_vectorizer, grammar_model)
        response = {
            "status": "Incorrect sentences found" if incorrect_sentences else "All sentences are grammatically correct",
            "incorrect_sentences": incorrect_sentences
        }
        if ml_errors:
            response["ml_warnings"] = ml_errors
            response["status"] = "Partial success: Rule-based checks completed, but ML predictions failed"
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Grammar check failed: {str(e)}"}), 500

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)





# add the dataset abd retrain the tfidf jorden model
