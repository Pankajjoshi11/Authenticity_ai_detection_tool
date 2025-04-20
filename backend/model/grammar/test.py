import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the dataset (replace with your actual dataset path)
try:
    df = pd.read_csv("grammar_dataset.csv")
except FileNotFoundError:
    print("Error: grammar_dataset.csv not found. Please provide the dataset.")
    exit(1)

# Verify dataset columns
if 'text' not in df.columns or 'label' not in df.columns:
    print("Error: Dataset must contain 'text' and 'label' columns.")
    exit(1)

# Split the dataset into train and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_vec = vectorizer.transform(X_test)  # Transform test data

# Verify vectorizer is fitted
if not hasattr(vectorizer, 'vocabulary_'):
    print("Error: TfidfVectorizer failed to fit.")
    exit(1)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate the model's accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Create model directory if it doesn't exist
os.makedirs("./model", exist_ok=True)

# Save the Logistic Regression model
with open("./model/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Logistic Regression model saved to ./model/logistic_regression_model.pkl")

# Save the TfidfVectorizer
with open("./model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("TfidfVectorizer saved to ./model/tfidf_vectorizer.pkl")