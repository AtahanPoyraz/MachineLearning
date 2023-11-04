import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
data_frame = pd.read_csv("datasets/kalp/heart.csv")

x = data_frame.drop("target", axis=1)
y = data_frame["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)

model.fit(x_train, y_train)
scr = model.score(x_test, y_test)

print(scr)