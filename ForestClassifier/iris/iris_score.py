from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(n_estimators=100, random_state=42)
df = pd.read_csv("datasets/iris/iris.csv")

x = df.drop("variety", axis=1)
y = df["variety"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print(score)