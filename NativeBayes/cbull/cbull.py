import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
model = MultinomialNB()

df = pd.read_csv("datasets/cbull/cyberbullying_tweets.csv")

x = vectorizer.fit_transform(df["tweet_text"])
y = df["cyberbullying_type"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

def create_new_df(text: str):
    data = vectorizer.transform([text])
    prediction = model.predict(data)
    return prediction[0]

msg = input("Enter a message: ")
result = create_new_df(msg)

print("Result:", result)
