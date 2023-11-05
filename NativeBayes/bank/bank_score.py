import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("datasets/bank/bank.csv")
df = pd.get_dummies(df, columns=['job', 'marital', 'education', "default", "housing", "loan", "contact", "month", "poutcome"])
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

model = MultinomialNB()

x = df.drop("deposit", axis=1)
y = df["deposit"]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=42, test_size=0.2)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(score)
