import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time

class FRIDAY:
    def __init__(self, dataset_path: str, stopwords_path : str):
        self.current_answer = None
        self.df = pd.read_csv(dataset_path)
        with open(stopwords_path, "r", encoding="utf-8") as file:
            self.stop_words = file.read().split("\n")

        self.X = self.df["soru"] 
        self.y = self.df["cevap"] 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = make_pipeline(TfidfVectorizer(stop_words=self.stop_words), RandomForestClassifier())
        self.model.fit(self.X_train, self.y_train)

    def prediction(self, girdi):
        try:
            tahmin = self.model.predict([girdi])
            cevap = tahmin[0]

            if cevap != self.current_answer:
                print("\x1b[1;32m[+]\x1b[1;0m FRIDAY")
                
                for harf in cevap:
                    print(harf, end="", flush=True)
                    time.sleep(0.06)
                
                print("")
                self.current_answer = cevap
            else:
                print("\x1b[1;31m[-]\x1b[1;0m FRIDAY")
                print("Uzgunum Bu soru için gerekli bilgiye sahip degilim.")
        
        except KeyboardInterrupt:
            print("")
            pass
        
        except Exception as e:
            print("\n\x1b[1;31m[-]\x1b[1;0m Hata: {}".format(str(e)))

    def learn_mode(self):
        print("Çıkış için 'çıkış' yazınız")
        while True:
            soru = input("SORU: ")
            if soru.lower() == "çıkış":
                break
            cevap = input("CEVAP: ")
            if soru == "" and cevap == "":
                print("Geçersiz bilgi")
            data = f"\n{soru},{cevap.capitalize()}"
            with open("datasets/dataset.csv", "a", encoding="utf-8") as file:
                file.write(data)
            print("Bilgi eklendi..")

    def run(self):
        while True:
            try:
                kullanici_girisi = input(f"\x1b[1;34m[x]\x1b[1;0m {os.getlogin().capitalize()}\n>> ")
                if kullanici_girisi == "OGRENIM MODU":
                    self.learn_mode()
                else:
                    self.prediction(girdi=kullanici_girisi)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    friday = FRIDAY(dataset_path="datasets/dataset.csv", stopwords_path="stopword/stopword.txt")
    friday.run()
