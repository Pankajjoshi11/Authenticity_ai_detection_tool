from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import nltk
from werkzeug.utils import secure_filename
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import  fitz  # PyMuPDF for PDF text extraction
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
from flask_pymongo import PyMongo


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://kuwarsarthak711:NqNS3a5sKP6EhtCC@papers.srhpdkr.mongodb.net/papers?retryWrites=true&w=majority"
CORS(app)  # Enable CORS for all routes
try:
    mongo = PyMongo(app)
    print(f"Connected to MongoDB ")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

nltk.download('punkt_tab')
nltk.download('stopwords')

# Load TF-IDF Summarization Model
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

# Load AI Detection Model
AI_DETECTION_MODEL_PATH = "./model/ai_detection"
try:
    ai_tokenizer = RobertaTokenizer.from_pretrained(AI_DETECTION_MODEL_PATH)
    ai_model = AutoModelForSequenceClassification.from_pretrained(AI_DETECTION_MODEL_PATH)
    ai_model.eval()
    print("✅ AI Detection model loaded successfully!")
except Exception as e:
    print("❌ Error loading AI detection model:", str(e))
    ai_model = None

# Load Grammar Check Models
LOGISTIC_MODEL_PATH = "./model/grammar/model/logistic_regression_model.pkl"
VECTORIZER_PATH = "./model/grammar/model/tfidf_vectorizer.pkl"
try:
    if not os.path.exists(LOGISTIC_MODEL_PATH):
        raise FileNotFoundError(f"Logistic Regression model file not found at {LOGISTIC_MODEL_PATH}")
    with open(LOGISTIC_MODEL_PATH, "rb") as f:
        grammar_model = pickle.load(f)
    print("✅ Logistic Regression model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Logistic Regression model: {str(e)}")
    grammar_model = None

try:
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"TfidfVectorizer file not found at {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "rb") as f:
        grammar_vectorizer = pickle.load(f)
    if not hasattr(grammar_vectorizer, 'vocabulary_'):
        raise ValueError("Loaded TfidfVectorizer is not fitted")
    print("✅ TfidfVectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TfidfVectorizer: {str(e)}")
    grammar_vectorizer = None

# PDF Text Extraction
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    except Exception as e:
        print("❌ PDF extraction error:", str(e))
    return text.strip()

# TF-IDF Summarization Function
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

# Grammar Check Functions
def is_grammatically_incorrect(sentence):
    if re.search(r"\b(?<!\w)dont(?!\w)\b", sentence):
        return True
    if re.search(r"\b(?:he|she|it|Rahul|John|Mary)\b\s+(?:dont|don't|doesn't|isn't|are|were)\b", sentence):
        return True
    if re.search(r"\bmyself\b", sentence) and not re.search(r"Rahul and myself", sentence):
        return True
    if re.search(r"\bgoes\b", sentence) and not re.search(r"can't go", sentence):
        return True
    if re.search(r"\bloves\b", sentence) and not re.search(r"\bthey\b", sentence):
        return True
    if re.search(r"\bwe\b\s+has\b", sentence):
        return True
    if re.search(r"\bcan't goes\b", sentence):
        return True
    if re.search(r"\bhe don't\b", sentence):
        return True
    if re.search(r"\bShe don't\b", sentence):
        return True
    if re.search(r"Rahul and myself", sentence):
        return True
    return False

def check_grammar(text, vectorizer, model):
    sentences = sent_tokenize(text)
    incorrect_sentences = []
    ml_errors = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if is_grammatically_incorrect(sentence):
            incorrect_sentences.append(sentence)
        else:
            try:
                if not hasattr(vectorizer, 'vocabulary_'):
                    raise ValueError("TfidfVectorizer is not fitted")
                sentence_vec = vectorizer.transform([sentence])
                prediction = model.predict(sentence_vec)
                if prediction == 0:
                    incorrect_sentences.append(sentence)
            except Exception as e:
                ml_errors.append(f"ML prediction error for sentence '{sentence}': {str(e)}")
    return incorrect_sentences, ml_errors

# Endpoints
@app.route("/summarize", methods=["POST"])
def summarize_text_endpoint():
    if vectorizer is None or kmeans is None:
        return jsonify({"error": "TF-IDF Summarization model not loaded"}), 500
    if "file" in request.files:
        file = request.files["file"]
        text = extract_text_from_pdf(file)
    else:
        try:
            data = request.get_json()
            text = data.get("text", "")
        except Exception as e:
            return jsonify({"error": f"Invalid JSON data: {str(e)}"}), 400
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    try:
        summary = summarize_text(text, vectorizer, kmeans, summary_ratio=0.25)
        summary_text = " ".join(summary)
        return jsonify({"summary": summary_text})
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route("/detect", methods=["POST"])
def detect_ai():
    if not ai_model:
        return jsonify({"error": "AI Detection model not loaded"}), 500
    if "file" in request.files:
        file = request.files["file"]
        text = extract_text_from_pdf(file)
    else:
        try:
            data = request.get_json()
            text = data.get("text", "")
        except Exception as e:
            return jsonify({"error": f"Invalid JSON data: {str(e)}"}), 400
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

@app.route("/check-grammar", methods=["POST"])
def check_grammar_endpoint():
    if grammar_model is None or grammar_vectorizer is None:
        return jsonify({"error": "Grammar check models not loaded"}), 500
    if "file" in request.files:
        file = request.files["file"]
        text = extract_text_from_pdf(file)
    else:
        try:
            data = request.get_json()
            text = data.get("text", "")
        except Exception as e:
            return jsonify({"error": f"Invalid JSON data: {str(e)}"}), 400
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

def query_database(user_text):
    try:
        # Load vectorizer data
        vocab_doc = mongo.db.vocab.find_one()
        if not vocab_doc:
            raise ValueError("No vectorizer data found")
        
        vocab_data = pickle.loads(vocab_doc['vectorizer_data'])
        
        # Reconstruct the vectorizer EXACTLY as it was during storage
        vectorizer = TfidfVectorizer(
            stop_words='english',
            vocabulary=vocab_data['vocabulary'],
            lowercase=True,  # Must match storage parameters
            token_pattern=r'(?u)\b\w\w+\b'  # Default token pattern
        )
        vectorizer.idf_ = vocab_data['idf']
        
        # Debug: Print vocabulary size
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Transform user text - ensure identical preprocessing
        user_vector = vectorizer.transform([user_text])
        print(f"User vector shape: {user_vector.shape}")
        
        # Process all paragraphs
        similarities = []
        para_count = 0
        match_count = 0
        
        for para in mongo.db.paragraphs.find():
            para_count += 1
            try:
                # Reconstruct sparse vector
                vec_data = pickle.loads(para['vector_data'])
                stored_vector = type(user_vector)(
                    (vec_data['data'], vec_data['indices'], vec_data['indptr']),
                    shape=vec_data['shape']
                )
                
                # Calculate similarity
                similarity = cosine_similarity(user_vector, stored_vector)[0][0]
                
                # Debug output
                if similarity > 0.5:  # Temporary lower threshold for debugging
                    print(f"Match found: {similarity:.2f} - {para['url']}")
                    print(f"Stored text: {para['paragraph'][:50]}...")
                    print(f"User text: {user_text[:50]}...")
                    match_count += 1
                
                if similarity > 0.1:  # Your actual threshold
                    similarities.append({
                        "url": para['url'],
                        "paragraph": para['paragraph'],
                        "similarity": float(similarity)
                    })
            except Exception as e:
                print(f"Error processing paragraph: {str(e)}")
                continue
        
        print(f"Processed {para_count} paragraphs, found {match_count} potential matches")
        
        # Get best match per URL
        best_matches = {}
        for item in similarities:
            url = item['url']
            if url not in best_matches or item['similarity'] > best_matches[url]['similarity']:
                best_matches[url] = item
        
        return sorted(best_matches.values(), key=lambda x: x['similarity'], reverse=True)
    
    except Exception as e:
        print(f"Query error: {str(e)}")
        return []

    
@app.route('/query', methods=['POST'])
def query_paper():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                user_text = f.read()
            os.remove(filepath)
        else:
            data = request.get_json()
            user_text = data.get('text', '')
        
        if not user_text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        results = query_database(user_text)
        
        # Simplified response with only plagiarism status and top URLs
        if not results:
            return jsonify({
                "plagiarism_found": False,
                "message": "No plagiarism detected"
            })
        
        # Get top 3 most similar sources
        top_results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:3]
        
        return jsonify({
            "plagiarism_found": True,
            "message": "Potential plagiarism detected",
            "sources": [
                {
                    "url": r['url'],
                    "similarity_score": round(r['similarity'] * 100, 1)
                } for r in top_results
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)