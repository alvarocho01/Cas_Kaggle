import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    dataset['gender'] = dataset['gender'].astype('category')
    dataset['gender'] = dataset['gender'].cat.codes
    dataset['class'] = dataset['class'].astype('category')
    dataset['class'] = dataset['class'].cat.codes
    return dataset

dataset = load_dataset("../dataset/bodyPerformance.csv")

#Printar les dades de cada classe que hi ha per veure si estan desbalancejades
n_clases = dataset["class"].nunique()
values = dataset.values
cont = 0
for element in range(n_clases):
    array = np.where(values[:,-1]==element)
    print(np.asarray(array).size)
    cont = cont + np.asarray(array).size
print(cont)

#Printar quantes dades hi ha de cada genere
n_genere = dataset["gender"].nunique()
values = dataset.values
cont = 0
for element in range(n_genere):
    array = np.where(values[:,1]==element)
    print(np.asarray(array).size)
    cont = cont + np.asarray(array).size
print(cont)

#Veure si hi ha valors duplicats i valors nuls
print(dataset[dataset.duplicated()])
print(dataset.isnull().values.any())

titles = dataset.columns.values
values = dataset.values

x = values[:,:-1]
y = values[:,-1]
title_x = titles
title_x = np.delete(title_x, 11, 0)
title_y = titles[11]

#Calcul de les desviacions de cada atribut
desviaciones = np.std(values,axis=0)
with open("../results/deviations.txt",'w') as d:
    for i, des in enumerate(desviaciones):
        d.write(titles[i] + " : " + str(des) + "\n")
        d.write("----------------------------\n")

plt.figure()


#Generar grafiques de dispersió i histogrames per tots els atributs d'entrada
for i in range(x.shape[1]):
    sns.scatterplot(data=dataset, x=x[:,i], y=x[:,3], hue="class")
    plt.xlabel(title_x[i])
    plt.ylabel(title_x[3])
    plt.savefig("../Grafiques/punts/" + str(i) + " " + title_x[i] + ".png")
    plt.clf()
      
    sns.histplot(data=x[:,i],kde=True, line_kws={'linewidth': 2}, color='g', alpha=0.3)
    plt.savefig("../Grafiques/histogrames/" + str(i) + " "+ title_x[i] + ".png")
    plt.clf()
    sns.violinplot(x=y[:], y=x[:,i],data=dataset, inner=None)
    plt.xlabel(title_y)
    plt.ylabel(title_x[i])
    plt.savefig("../Grafiques/violin_plot/" + str(i) + " "+ title_x[i] + ".png")
    plt.clf()


#Test de shapiro per veure la normalitat de les dades
resultats = []
for i in range(x.shape[1]):
    resultats.append(scipy.stats.shapiro(x[:,i]))
resultats = np.array(resultats)
    
with open("../results/results_shapiro.txt",'w') as f:
    f.write(" - TEST DE SHAPIRO - \n")
    f.write("---------------------\n")
    for k, res in enumerate(resultats):
        f.write(title_x[k] + " : Estadistico: " + str(res[0]) + "   |   P-Valor: " + str(res[1]) + "\n")
        if res[1] < 0.05:
            f.write("Se puede rechazar la hipotesis de que los datos se distribuyen de forma normal\n")
        else:
            f.write("No se puede rechazar la hipotesis de que los datos se distribuyen de forma normal\n")
        f.write("-----------------------------------------------------------------------------\n")
 

#Generar taula de correlacio
correlacio = dataset.corr()
plt.title("Correlacio")
plt.figure(figsize=(8,8))
ax = sns.heatmap(correlacio, annot=True, fmt=".2f")
plt.savefig("../Grafiques/correlacio/correlacio.png")
plt.clf()







