from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Veri setini oku
df = pd.read_csv("datasets/mental/mental.csv")

# Sütun isimlerini güncelle
df.columns = ["Timestamp", "Gender", "Age", "Course", "Year", "CGPA", "Marital", "Depression", "Anxiety", "Panic_attack", "Treatment"]

# CountVectorizer ve MultinomialNB modelini oluştur
vectorizer = CountVectorizer()
model = MultinomialNB()

# Sınıflandırma için kullanılacak sütunları bir araya getir
text_columns = ["Gender", "Age", "Course", "Year", "CGPA", "Marital"]
df['text'] = df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Metin verilerini sayısallaştır
X = vectorizer.fit_transform(df['text'])
y = df["Depression"]  # Depresyon sütunu hedef değişkenimiz

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model.fit(X_train, y_train)

# Modelin performansını değerlendir
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
