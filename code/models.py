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

n_clases = dataset["class"].nunique()


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
    

param_svm = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
param_perceptron = {'penalty': ['l2','l1','elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                    'fit_intercept': [True, False], 'shuffle': [True, False]}
param_knn = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
param_decisiontree = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
param_randomforest = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
param_logireg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

param_grid = [param_perceptron, param_knn, param_decisiontree, param_randomforest, param_logireg]

models = [Perceptron(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "Perceptron", "KNN", "Decision Tree", "Random Forest", "Logistic Regression"]

for i,model in enumerate(models):
    '''Busqueda exhaustiva de los mejores parametros'''
    print("BUSQUEDA EXHAUSTIVA DE PARAMETROS")
    grid = GridSearchCV(model, param_grid[i], verbose=3, n_jobs=-1)
    grid.fit(x,y)
    print("Els millors parametres: ",grid.best_params_)
    print("El millor score: ", grid.best_score_)
    print("")

models = [svm.SVC(probability=True), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
nom_models = ["Support Vector Machines", "KNN", "Decision Tree", "Random Forest", "Logistic Regression"]

for i,model in enumerate(models):
    #Generar corbes ROC i PR
    x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=0.8)
    model.fit(x_t,y_t)
    probs = model.predict_proba(x_v)
    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for j in range(n_clases):
        precision[j], recall[j], _ = precision_recall_curve(y_v == j, probs[:, j])
        average_precision[j] = average_precision_score(y_v == j, probs[:, j])

        plt.plot(recall[j], precision[j],
        label='Precision-recall curve of class {0} (area = {1:0.2f})'
                               ''.format(j, average_precision[j]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(nom_models[i])
        plt.legend(loc="upper right")
    plt.savefig("../Grafiques/corbes_PR/corba-pr" + str(nom_models[i]) + ".png")
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for j in range(n_clases):
        fpr[j], tpr[j], _ = roc_curve(y_v == j, probs[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    rnd_fpr, rnd_tpr, _ = roc_curve(y_v>0, np.zeros(y_v.size))
    plt.plot(rnd_fpr, rnd_tpr, linestyle='--', label='Sense capacitat predictiva')
    for j in range(n_clases):
        plt.plot(fpr[j], tpr[j], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(j, roc_auc[j]))
    plt.title(nom_models[i])
    plt.legend()
    plt.savefig("../Grafiques/corbes_ROC/curva-roc" + str(nom_models[i]) + ".png")










