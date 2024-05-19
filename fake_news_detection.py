import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import joblib
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier

# Load the CSV files
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Explore the structure and contents
print("Fake News Dataset:")
print(fake_df.head())
print(fake_df.info())

print("\nTrue News Dataset:")
print(true_df.head())
print(true_df.info())

# Check for missing values
print("\nMissing Values in Fake News Dataset:")
print(fake_df.isnull().sum())

print("\nMissing Values in True News Dataset:")
print(true_df.isnull().sum())

# Check for duplicates
print("\nDuplicates in Fake News Dataset:")
print(fake_df.duplicated().sum())

print("\nDuplicates in True News Dataset:")
print(true_df.duplicated().sum())

def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

fake_df['text'] = fake_df['text'].apply(clean_text)
true_df['text'] = true_df['text'].apply(clean_text)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the fake news text
fake_tfidf = tfidf_vectorizer.fit_transform(fake_df['text'])

# Transform the true news text (using the same vectorizer)
true_tfidf = tfidf_vectorizer.transform(true_df['text'])

# Concatenate the TF-IDF features and labels
X = scipy.sparse.vstack([fake_tfidf, true_tfidf])
y = np.concatenate([np.zeros(fake_tfidf.shape[0]), np.ones(true_tfidf.shape[0])])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the model
model.fit(X_train, y_train)  # Train the model

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'fake_news_detection_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
