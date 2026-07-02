import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix

print("=" * 50)
print("Tweet Sentiment Classification")
print("=" * 50)

# Load dataset
df = pd.read_csv("Tweets.csv")

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove rows with missing tweet text
df = df.dropna(subset=["text"])

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows if there are any
df = df.drop_duplicates()

# Keep only the required columns
df = df[["text", "sentiment"]]

print("\nDataset after cleaning:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# Text Preprocessing

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol
    text = re.sub(r"#", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply preprocessing
df["text"] = df["text"].apply(clean_text)

print("\nCleaned tweets:")
print(df.head())


# Feature Extraction

X = df["text"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(X)

print("\nTF-IDF matrix shape:")
print(X.shape)


# Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Logistic Regression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\n===== Logistic Regression =====")
print(f"Accuracy: {lr_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))


# Multinomial Naive Bayes

nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_predictions = nb.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print("\n===== Multinomial Naive Bayes =====")
print(f"Accuracy: {nb_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))


# Model Comparison

print("\n===== Model Comparison =====")

print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

if lr_accuracy > nb_accuracy:
    print("\nLogistic Regression performed better on this dataset.")
elif nb_accuracy > lr_accuracy:
    print("\nMultinomial Naive Bayes performed better on this dataset.")
else:
    print("\nBoth models achieved the same accuracy.")

