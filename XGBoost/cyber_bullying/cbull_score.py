from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

vectorizer = CountVectorizer()
model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

df = pd.read_csv("datasets/cbull/cyberbullying_tweets.csv")
df["cyberbullying_type"] = df["cyberbullying_type"].map({"not_cyberbullying": 0, "gender": 1, "religion":2, "other_cyberbullying":3, "age":4, "ethnicity":5})

x = vectorizer.fit_transform(df["tweet_text"])
y = df["cyberbullying_type"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print(score)