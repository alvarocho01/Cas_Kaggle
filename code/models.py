import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    dataset['gender'] = dataset['gender'].astype('category')
    dataset['gender'] = dataset['gender'].cat.codes
    dataset['class'] = dataset['class'].astype('category')
    dataset['class'] = dataset['class'].cat.codes
    return dataset

dataset = load_dataset("../dataset/bodyPerformance.csv")

titles = dataset.columns.values
values = dataset.values

x = values[:,:-1]
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
y = values[:,-1]
title_x = titles
title_x = np.delete(title_x, 11, 0)
title_y = titles[11]




particions = [0.5, 0.8, 0.7]
models = [svm.SVC(probability=True), Perceptron(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "Perceptron", "KNN", "Decision Tree", "Random Forest", "Logistic Regression"]
for i,model in enumerate(models):
    print("---- ", nom_models[i], " ----")
    print("Parametres per defecte: " + str(model.get_params()))
    print("")
    for part in particions:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=part)
        model.fit(x_train, y_train)
        print("Score del model amb ", part, ": ", model.score(x_test, y_test))
    print("")
    for k in range(2,7):
        scores = cross_val_score(model, x, y, cv=k)
        print("Score promig amb k-fold = ", k, " : ", scores.mean())
    print("")