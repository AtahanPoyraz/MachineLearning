from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/bank/bank.csv")

text_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
df['text'] = df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

res = model.score(X_test, y_test)
print("Model Accuracy:", res)
