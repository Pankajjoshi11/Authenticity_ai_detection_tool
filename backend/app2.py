# import pickle
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Assuming these are the trained models
# model = LogisticRegression(max_iter=1000)  # Your trained model
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Your trained vectorizer

# # Save the Logistic Regression model
# with open("./model/grammar/grammar_model_lr.pkl", "wb") as f:
#     pickle.dump(model, f)

# # Save the TfidfVectorizer
# with open("./model/grammar/grammar_vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# LOGISTIC_MODEL_PATH = "./model/grammar/grammar_model_lr.pkl"
# VECTORIZER_PATH = "./model/grammar/grammar_vectorizer.pkl"





import pickle

try:
    with open("./model/grammar/grammar_model_lr.pkl", "rb") as f:
        model = pickle.load(f)
    print("Logistic Regression model loaded successfully!")
except Exception as e:
    print("Error loading Logistic Regression model:", str(e))

try:
    with open("./model/grammar/grammar_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("TfidfVectorizer loaded successfully!")
except Exception as e:
    print("Error loading TfidfVectorizer:", str(e))