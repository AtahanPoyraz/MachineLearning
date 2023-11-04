from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib as mp

#ORMAN YAPILARI FAZLA ESLESME DURUMUNU AZALTIR DAHA TUTARLI SONUC VERİR

model = RandomForestClassifier(n_estimators=255, random_state=42)
df = pd.read_csv("datasets/kalp/heart.csv")

x = df.drop("target", axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.75)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)

prediction = model.predict(x_test)
print(x_test)
print(prediction)