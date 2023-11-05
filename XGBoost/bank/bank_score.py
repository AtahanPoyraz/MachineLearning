import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/bank/bank.csv")
df = pd.get_dummies(df, columns=['job', 'marital', 'education', "default", "housing", "loan", "contact", "month", "poutcome"])
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

x = df.drop("deposit", axis=1)
y = df["deposit"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(score)