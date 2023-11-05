import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/healt1/metabolic.csv")
df = pd.get_dummies(df, columns=['Sex', 'Marital', 'Race'])

y = df["MetabolicSyndrome"]

x = df.drop("MetabolicSyndrome", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model.fit(x_train, y_train)

def create_frame(seqn, Age, Income, WaistCirc, BMI, Albuminuria, UrAlbCr, UricAcid, BloodGlucose, HDL,
                Triglycerides, Sex_Female, Sex_Male, Marital_Divorced, Marital_Married,
                Marital_Separated, Marital_Single, Marital_Widowed, Race_Asian, Race_Black, Race_Hispanic,
                Race_MexAmerican, Race_Other, Race_White):
    new_df = pd.DataFrame({
        'seqn': [seqn],
        'Age': [Age],
        'Income': [Income],
        'WaistCirc': [WaistCirc],
        'BMI': [BMI],
        'Albuminuria': [Albuminuria],
        'UrAlbCr': [UrAlbCr],
        'UricAcid': [UricAcid],
        'BloodGlucose': [BloodGlucose],
        'HDL': [HDL],
        'Triglycerides': [Triglycerides],
        'Sex_Female': [Sex_Female],
        'Sex_Male': [Sex_Male],
        'Marital_Divorced': [Marital_Divorced],
        'Marital_Married': [Marital_Married],
        'Marital_Separated': [Marital_Separated],
        'Marital_Single': [Marital_Single],
        'Marital_Widowed': [Marital_Widowed],
        'Race_Asian': [Race_Asian],
        'Race_Black': [Race_Black],
        'Race_Hispanic': [Race_Hispanic],
        'Race_MexAmerican': [Race_MexAmerican],
        'Race_Other': [Race_Other],
        'Race_White': [Race_White]
    })

    prediction = model.predict(new_df)

    return prediction[0]

new_example = create_frame(2, 35, 60000, 85.5, 24.5, 0, 3.5, 5.9, 95, 55, 110, False, True, False, True, False, False, False, False, False, True, False, False, False)
new_example2 = create_frame(65823,27,300,104,29.5,0,3.41,5.5,96,43,178,1,True, False, False, True, False, False, False, False, False, True, False, False)
status = str(bool(new_example2))
print(f"Metabolik Sendrom Durumu: {status}")
