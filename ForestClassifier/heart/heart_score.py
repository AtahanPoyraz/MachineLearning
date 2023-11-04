from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib as mp

#ORMAN YAPILARI FAZLA ESLESME DURUMUNU AZALTIR DAHA TUTARLI SONUC VERİR

model = RandomForestClassifier(n_estimators=100, random_state=42)
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


#GORSELLIK
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, prediction)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Tahmin Edilen Etiketler")
plt.ylabel("Gerçek Etiketler")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
roc_auc = roc_auc_score(y_test, prediction)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
