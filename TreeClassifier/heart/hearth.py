from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

model = DecisionTreeClassifier()
data_frame = pd.read_csv("datasets/kalp/heart.csv")

x = data_frame.drop("target", axis=1)
y = data_frame["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)

model.fit(x_train, y_train)

def check_healt(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    new_data = pd.DataFrame({"age":[age], "sex":[sex], "cp":[cp], "trestbps":[trestbps], "chol":[chol], "fbs":[fbs], "restecg":[restecg], 
                             "thalach":[thalach], "exang":[exang], "oldpeak":[oldpeak], "slope":[slope], "ca":[ca], "thal":[thal]})
    
    prediction = model.predict(new_data)
    return prediction[0]


new_data =  check_healt(18,0,1,120,204,0,1,133,1,0.6,2,0,2)
new_data2 = check_healt(75,1,0,120,248,0,1,122,1,3.2,1,3,3)

print(new_data, new_data2)