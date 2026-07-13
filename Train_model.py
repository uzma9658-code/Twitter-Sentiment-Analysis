import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Excel file
data = pd.read_excel("twitter_sentiment.csv.xlsx")

# Use correct column names
X = data["text"]
y = data["sentiment"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# Save files
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")
