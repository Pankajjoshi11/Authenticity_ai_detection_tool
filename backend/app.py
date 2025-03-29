from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    if not ai_model:
        return jsonify({"error": "AI Detection model not loaded"}), 500

    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    sentences = text.split(". ")
    ai_sentences = []

    total_ai_score = 0
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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
