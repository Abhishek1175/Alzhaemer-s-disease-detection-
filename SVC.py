import pandas as pd
from sklearn.model_selection import train_test_split
from cuml.svm import SVC

from sklearn.metrics import accuracy_score

df=pd.read_csv("alzheimers_disease_data.csv")
print("Data loaded sucessfully\n")

print(df.head())

x = df.drop(columns = ["Diagnosis" ,"PatientID", "DoctorInCharge"], axis = 1)

y=df["Diagnosis"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,random_state=42
)

model=SVC(kernel='liner')
model.fit(x_train,y_train)
y_pre=model.predict(x_test)
acc=accuracy_score(y_pre,y_test)
print("Accuracy score:",acc)
person=[83,0,0,1,30.95364704240386,0,4.692218861789743,0.44269086420938963,1.2892518488393956,7.869067535006348,0,0,0,0,0,0,107,104,186.55983336914747,63.42324157105127,88.90087636957928,114.9232292815966,28.661262174905925,8.153338978912668,1,0,2.1116385814865812,1,0,0,0,0,0]
prob=model.predict_proba([person])
result=model.predict([person])
if result == 1:
    print("The patient is detected with Alzheimer's Disease.")
else:
    print("The patient is not detected with Alzheimer's Disease.")