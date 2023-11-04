from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/bank/bank.csv")

text_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
df['text'] = df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1) #YENI SATIR OLUŞTURUYORUZ

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

res = model.score(X_test, y_test)
print("Model Accuracy:", res)

def check_deposit(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome):
    new_data = pd.DataFrame({'age': [age],
                             'job': [job],
                             'marital': [marital],
                             'education': [education],
                             'default': [default],
                             'balance': [balance],
                             'housing': [housing],
                             'loan': [loan],
                             'contact': [contact],
                             'day': [day],
                             'month': [month],
                             'duration': [duration],
                             'campaign': [campaign],
                             'pdays': [pdays],
                             'previous': [previous],
                             'poutcome': [poutcome]})

    new_data["text"] = df[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    new_data = vectorizer.transform(new_data["text"])

    prediction = model.predict(new_data)

    return prediction[0]

age = 35
job = "admin."
marital = "married"
education = "secondary"
default = "no"
balance = 15000
housing = "yes"
loan = "no"
contact = "unknown"
day = 10
month = "may"
duration = 800
campaign = 2
pdays = -1
previous = 0
poutcome = "unknown"

#result = check_deposit(age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)
result = check_deposit(39,"self-employed","married","tertiary","no",2630,"no","no","unknown",12,"jun",651,5,-1,0,"unknown")
print(result)