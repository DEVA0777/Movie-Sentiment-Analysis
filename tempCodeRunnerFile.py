import pandas as pd

df = pd.read_csv("dataset/IMDB Dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nSentiment count:")
print(df["sentiment"].value_counts())

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

X = df["review"]
y = df["sentiment"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")
print("Training shape:", X_train_tfidf.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


review = input("Enter your movie review: ")

review_tfidf = vectorizer.transform([review])

prediction = model.predict(review_tfidf)[0]

if prediction == 1:
    print("😊 Positive Review")
else:
    print("😞 Negative Review")


import pickle

# Save trained model
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")





