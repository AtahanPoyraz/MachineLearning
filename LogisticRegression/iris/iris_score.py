from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

model = LogisticRegression(C=1.0, max_iter=1000, penalty='l2')
df = pd.read_csv("datasets/iris/iris.csv")

x = df.drop("variety", axis=1)
y = df["variety"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=42)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)