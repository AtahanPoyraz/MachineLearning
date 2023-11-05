import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/bank/bank.csv")
df = pd.get_dummies(df, columns=['job', 'marital', 'education', "default", "housing", "loan", "contact", "month", "poutcome"])
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

x = df.drop("deposit", axis=1)
y = df["deposit"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

def create_bank_frame(age, balance, day, duration, campaign, pdays, previous,
                      job_admin, job_blue_collar, job_entrepreneur, job_housemaid, job_management,
                      job_retired, job_self_employed, job_services, job_student, job_technician, job_unemployed, job_unknown,
                      marital_divorced, marital_married, marital_single,
                      education_primary, education_secondary, education_tertiary, education_unknown,
                      default_no, default_yes, housing_no, housing_yes, loan_no, loan_yes,
                      contact_cellular, contact_telephone, contact_unknown,
                      month_apr, month_aug, month_dec, month_feb, month_jan, month_jul, month_jun, month_mar, month_may, month_nov, month_oct, month_sep,
                      poutcome_failure, poutcome_other, poutcome_success, poutcome_unknown):
    data = {
        'age': [age],
        'balance': [balance],
        'day': [day],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'job_admin.': [job_admin],
        'job_blue-collar': [job_blue_collar],
        'job_entrepreneur': [job_entrepreneur],
        'job_housemaid': [job_housemaid],
        'job_management': [job_management],
        'job_retired': [job_retired],
        'job_self-employed': [job_self_employed],
        'job_services': [job_services],
        'job_student': [job_student],
        'job_technician': [job_technician],
        'job_unemployed': [job_unemployed],
        'job_unknown': [job_unknown],
        'marital_divorced': [marital_divorced],
        'marital_married': [marital_married],
        'marital_single': [marital_single],
        'education_primary': [education_primary],
        'education_secondary': [education_secondary],
        'education_tertiary': [education_tertiary],
        'education_unknown': [education_unknown],
        'default_no': [default_no],
        'default_yes': [default_yes],
        'housing_no': [housing_no],
        'housing_yes': [housing_yes],
        'loan_no': [loan_no],
        'loan_yes': [loan_yes],
        'contact_cellular': [contact_cellular],
        'contact_telephone': [contact_telephone],
        'contact_unknown': [contact_unknown],
        'month_apr': [month_apr],
        'month_aug': [month_aug],
        'month_dec': [month_dec],
        'month_feb': [month_feb],
        'month_jan': [month_jan],
        'month_jul': [month_jul],
        'month_jun': [month_jun],
        'month_mar': [month_mar],
        'month_may': [month_may],
        'month_nov': [month_nov],
        'month_oct': [month_oct],
        'month_sep': [month_sep],
        'poutcome_failure': [poutcome_failure],
        'poutcome_other': [poutcome_other],
        'poutcome_success': [poutcome_success],
        'poutcome_unknown': [poutcome_unknown]
    }

    new_df = pd.DataFrame(data)

    prediction = model.predict(new_df)
    return prediction[0]

sample_data = create_bank_frame(age=30, balance=1000, day=10, duration=300, campaign=2, pdays=15, previous=3,
                               job_admin=False, job_blue_collar=True, job_entrepreneur=False, job_housemaid=False,
                               job_management=True, job_retired=False, job_self_employed=False, job_services=False,
                               job_student=False, job_technician=False, job_unemployed=False, job_unknown=False,
                               marital_divorced=True, marital_married=False, marital_single=False,
                               education_primary=False, education_secondary=True, education_tertiary=False,
                               education_unknown=False, default_no=True, default_yes=False, housing_no=True,
                               housing_yes=False, loan_no=True, loan_yes=False, contact_cellular=True,
                               contact_telephone=False, contact_unknown=False, month_apr=False, month_aug=True,
                               month_dec=False, month_feb=False, month_jan=False, month_jul=False, month_jun=True,
                               month_mar=False, month_may=False, month_nov=False, month_oct=False, month_sep=False,
                               poutcome_failure=False, poutcome_other=False, poutcome_success=False, poutcome_unknown=True)


print(f"Deposit Durumunuz: {str(bool(sample_data))}")