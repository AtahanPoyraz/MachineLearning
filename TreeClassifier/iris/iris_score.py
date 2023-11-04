import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_frame = pd.read_csv("datasets/iris/iris.csv")
model = DecisionTreeClassifier()

x = data_frame.drop("variety", axis=1)
y = data_frame["variety"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)
