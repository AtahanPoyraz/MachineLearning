from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Veri kümesini yükleyin
data_frame = pd.read_csv("datasets/iris/iris.csv")

# Veriyi özellikler (x) ve etiketler (y) olarak ayırın
x = data_frame.drop('variety', axis=1)
y = data_frame['variety']

# Eğitim ve test veri kümelerini oluşturun
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluşturun ve eğitin
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Tahmin fonksiyonunu tanımlayın
def predict_flower_species(sepal_length, sepal_width, petal_length, petal_width):
    # Yeni veriyi oluşturun
    new_data = pd.DataFrame({'sepal.length': [sepal_length], 'sepal.width': [sepal_width], 'petal.length': [petal_length], 'petal.width': [petal_width]})

    # Modeli kullanarak tahmin yap
    prediction = model.predict(new_data)
    return prediction[0]

# Örnek tahmin yapma
sepal_length = float(input("sepal_length: "))
sepal_width = float(input("sepal_width: "))
petal_length = float(input("petal_length: "))
petal_width = float(input("petal_width: "))

predicted_species = predict_flower_species(sepal_length, sepal_width, petal_length, petal_width)
print("Tahmin edilen tür: ", predicted_species)
