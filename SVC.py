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


    C = [0.01, 0.1, 1,10,100]
for i in range(len(C)):
  model_i = SVC(
      C = C[i],
      kernel = "linear"
  )
  model_i.fit(x_train,y_train)
  ypred_i = model_i.predict(x_test)
  train_acc = model_i.score(x_train,y_train)
  score_i = accuracy_score(y_test,ypred_i)
  print(f"training accuracy is given by{train_acc} \n testing accuracy is given by {score_i}")

  from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

kernels = ['rbf', 'sigmoid']

best_score = 0
best_model = None

for kernel in kernels:

    svc = SVC(kernel=kernel)

    grid = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(x_train, y_train)

    print(f"\nKernel: {kernel}")
    print("Best Parameters:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid

print("\nOverall Best Model")
print("Best Kernel:", best_model.best_estimator_.kernel)
print("Best Parameters:", best_model.best_params_)
print("Best CV Accuracy:", best_model.best_score_)

# Test Accuracy
y_pred = best_model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, y_pred))