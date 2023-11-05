import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

classes = {0:"Clear", 1:"Spam"}

scaler = StandardScaler(with_mean=False)
vectorizer = CountVectorizer()

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

df = pd.read_csv("datasets/spam/spam.csv")
df["Category"] = df["Category"].map({"spam": 1, "ham": 0})

x = vectorizer.fit_transform(df["Message"])
y = df["Category"]

x_train, x_test, y_train, y_test = train_test_split(x, y ,random_state=42, test_size=0.2)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

def create_new_df(text: str):
    data = vectorizer.transform([text])

    prediction = model.predict(data)

    return prediction[0]


text = input(": ")
result = create_new_df(text)
print(classes[int(result)])