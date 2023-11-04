from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/spam/spam.csv")
vectorizer = CountVectorizer()
model = MultinomialNB()

X = vectorizer.fit_transform(df["Message"])
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Model Accuracy:", score)


