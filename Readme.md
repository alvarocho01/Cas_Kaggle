#Pràctica Kaggle APC UAB 2021-22
###Nom: Álvaro Caravaca Hernández
###Dataset: Dades de Rendiment Corporal
###URL: https://www.kaggle.com/kukuroo3/body-performance-data

##Resum
El dataset conté 13393 instàncies amb 12 atributs cadascuna. 2 d'aquests atributs són categòrics i la resta són numèrics. 
Els atributs representen característiques físiques d'una persona, com per exemple el pes o el percentatge de grassa corporal, i característiques esportives d'una persona, com per exemple el nombre d'abdominals que pot fer en un determinat temps.

##Objectius del dataset
L'objectiu principal serà classificar la salut d'una persona depenent dels atributs d'entrada. Hi ha quatre tipus de "salut" en una persona: A, B, C i D, sent A una salut molt bona i D una salut molt dolenta.

##Preprocessat
El rang de valors que prenen tots els atributs és molt diferent, la qual cosa pot fer que els diferents models que s'apliquin no puguin classificar bé.
Per tal d'ajustar tots els atributs al mateix rang de dades, hauré de normalitzar les dades.

##Model
He provat cinc models diferents i he buscat els millors hiperparàmetres de cadascun. Aquests han estat els resultats:
|Model|Hiperparàmetres|Mètrica|Temps|
|--|--|--|--|
| SVM | ?????? | ??% | xms |
| kNN | 'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'distance'| 63.6% | xms |
| Decision Tree | 'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 5 | 68.3% | xms |
| Random Forest | 'bootstrap': True, 'max_depth': 110, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 200 | 74.0% | xms |
| Logistic Regression | 'C': 0.1 | 61.7% | xms |

##Conclusions
Es pot dir que el nivell de salut d'una persona esta directament relacionat amb el pes, el percentatge de grassa corporal, els centímetres que es pot inclinar cap endevant estant assegut i el nombre d'abdominals que és capaç de fer.
Hi ha una clara tendència a tenir pitjor salut si els resultats de les proves físiques son pitjors. En canvi, hi ha una tendència a tenir millor salut si el pes i el percentatge de grassa corporal és menor.
A partir de les dades proporcionades, es pot veure com les persones de gènere femení tenen una millor salut que les persones de gènere masculí.
Les dades estan representades en intervals molt diversos, tenint la necessitat de normalitzar-les per poder tractar-les correctament
El millor model que classifica aquestes dades és el Random Forest, amb un score màxim de 74%.
La cerca dels millors hiperparàmetres no ha millorat gaire el rendiment dels models. En concret, el Random Forest ha millorat solament un 1.6% respecte els valors per defecte.
