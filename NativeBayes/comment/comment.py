from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

#METİN TABANLI VERİ SETLERİ İÇİN 
df = pd.read_csv("datasets/spam/spam.csv")
vectorizer = CountVectorizer()
model = MultinomialNB()

X = vectorizer.fit_transform(df["Message"]) #METİNİ SAYISAL VERİLERE DÖNÜSTÜYORUZ
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

def detect_spam(message):
    new_data = pd.DataFrame({"Message": [message]})
    new_data = vectorizer.transform(new_data["Message"])

    prediction = model.predict(new_data)
    return prediction[0]

while True:
    ins = input(": ")
    message = detect_spam(ins)
    print("Spam Detection Result:", message)
