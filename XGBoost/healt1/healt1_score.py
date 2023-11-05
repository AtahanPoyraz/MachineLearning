import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
df = pd.read_csv("datasets/healt1/metabolic.csv")
df = pd.get_dummies(df, columns=['Sex', 'Marital', 'Race'])

x = df.drop("MetabolicSyndrome", axis=1)
y = df["MetabolicSyndrome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model.fit(x_train, y_train)

res = model.score(x_test, y_test)


print(res)
