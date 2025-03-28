from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# ========== LOAD SUMMARIZATION MODEL ==========
SUMMARIZATION_MODEL_PATH = "./model/ai_model"

try:
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_PATH)
    summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_PATH)
    summarizer = pipeline("summarization", model=summarization_model, tokenizer=summarization_tokenizer)
    print("✅ Summarization model loaded successfully!")
except Exception as e:
    print("❌ Error loading summarization model:", str(e))
    summarizer = None

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

# ========== SUMMARIZATION ENDPOINT ==========
@app.route("/summarize", methods=["POST"])
def summarize_text():
    """Summarizes input text."""
    if not summarizer:
        return jsonify({"error": "Summarization model not loaded"}), 500

    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

# ========== AI DETECTION ENDPOINT ==========
@app.route("/detect", methods=["POST"])
def detect_ai():
    """Detects whether a given text is AI-generated or human-written."""
    if not ai_model:
        return jsonify({"error": "AI Detection model not loaded"}), 500

    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    inputs = ai_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = ai_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    result = "AI-Generated" if prediction == 1 else "Human-Written"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
