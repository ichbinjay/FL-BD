from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
ds = pd.read_csv(r"C:\Users\ADMIN\FLPrrojectS\employee_burnout_analysis.csv")

X = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, confusion_matrix

acc = accuracy_score(y_test, y_pred) * 100
loss = classifier.loss_
f1sr = f1_score(y_test, y_pred, average="macro") * 100
recall = recall_score(y_test, y_pred, average="macro") * 100
confusion_mat = confusion_matrix(y_test, y_pred)
print("acc {:.3f}".format(acc), end="   ")
print("loss {:.3f}".format(loss), end="   ")
print("f1sr {:.3f}".format(f1sr), end="   ")
print("recall {:.3f}".format(recall), end="   ")

import pickle

# Specify the filename for the saved model
filename = 'final_saved_model.pkl'
# Open the file in binary write mode and use pickle to serialize the model
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)
print(f"Model has been saved to {filename}")

