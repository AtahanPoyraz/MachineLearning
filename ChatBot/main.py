import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time

def prediction(girdi, model):
    try:
        tahmin = model.predict([girdi])
        cevap = tahmin[0]
        if cevap:
            print("\x1b[1;32m[+]\x1b[1;0m FRIDAY")
            for harf in cevap:
                print(harf, end="", flush=True)
                time.sleep(0.06)
            print("")
        else:
            print("\n\x1b[1;31m[-]\x1b[1;0m Model cevap veremedi.\n")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("\n\x1b[1;31m[-]\x1b[1;0m Hata: {}".format(str(e)))

df = pd.read_csv("datasets/dataset.csv")

with open("stopword/stopword.txt", "r", encoding="utf-8") as file:
    turkish_stop_words = file.read().split("\n")

X = df["soru"]
y = df["cevap"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(stop_words=turkish_stop_words), RandomForestClassifier())
model.fit(X_train, y_train)

while True:
    try:
        kullanici_girisi = input(f"\x1b[1;34m[x]\x1b[1;0m {os.getlogin().capitalize()}\n>> ")

        if kullanici_girisi == "OGRENIM MODU":
            print("Cıkış için 'çıkış' yazınız")
            while True:
                soru = input("SORU: ")
                if soru.lower() == "çıkış":
                    break
                cevap = input("CEVAP: ")
                if soru == "" and cevap == "":
                    print("Gecersiz bilgi")
                data = f"\n{soru},{cevap.capitalize()}"
                with open("datasets/dataset.csv", "a", encoding="utf-8") as file:
                    file.write(data)
                print("Bilgi eklendi..")

        else:
            prediction(girdi=kullanici_girisi, model=model)
    except KeyboardInterrupt:
        break
