import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Load dataset
data = pd.read_csv("health_news.csv")

# 2. Split features and labels
X = data["text"]
y = data["label"]

# 3. Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# 4. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Save model + vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training complete! Model and vectorizer saved.")
