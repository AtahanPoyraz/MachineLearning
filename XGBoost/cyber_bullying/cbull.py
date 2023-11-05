from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import pandas as pd

# Veri çerçevesini yükle
df = pd.read_csv("datasets/cbull/cyberbullying_tweets.csv")
df["cyberbullying_type"] = df["cyberbullying_type"].map({"not_cyberbullying": 0, "gender": 1, "religion": 2, "other_cyberbullying": 3, "age": 4, "ethnicity": 5})

# CountVectorizer kullanarak metin verilerini vektörlere dönüştür
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df["tweet_text"])
y_train = df["cyberbullying_type"]

# XGBoost modelini oluştur ve eğit
model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model.fit(X_train, y_train)

# Yeni veriyi tahmin etmek için fonksiyonu oluştur
def create_new_frame(text: str):
    new_data = vectorizer.transform([text])
    prediction = model.predict(new_data)
    return prediction[0]

# Kullanıcıdan yorum al ve tahmin yap
while True:
    try:
        text = input("Yorum: ")
        new_ex = create_new_frame(text)
        print(new_ex)
    except Exception as e:
        print(str(e))
        continue
