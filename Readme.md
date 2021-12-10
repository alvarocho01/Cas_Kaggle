# Pràctica Kaggle APC UAB 2021-22
### Nom: Álvaro Caravaca Hernández
### Dataset: Dades de Rendiment Corporal
### URL: https://www.kaggle.com/kukuroo3/body-performance-data

## Resum
El dataset conté 13393 instàncies amb 12 atributs cadascuna. 2 d'aquests atributs són categòrics i la resta són numèrics. 
Els atributs representen característiques físiques d'una persona, com per exemple el pes o el percentatge de grassa corporal, i característiques esportives d'una persona, com per exemple el nombre d'abdominals que pot fer en un determinat temps.

## Objectius del dataset
L'objectiu principal serà classificar la salut d'una persona depenent dels atributs d'entrada. Hi ha quatre tipus de "salut" en una persona: A, B, C i D, sent A la millor salut i D la pitjor.

## Enteniment de les dades
Els atributs que més correlació tenen amb la sortida (atribut class) són el pes, el percentatge de grassa corporal, seure i inclinar-se cap endavant (sit and bend forward), recompte d'abdominals (sit-ups counts) i salt de longitud (broad jump). Es pot veure que els dos primers atributs tenen correlació positiva, la qual cosa significa que quan augmenta el valor de l'atribut també augmenta la classe (si augmenta el percentatge de grassa corporal és més probable que la classe sigui 3, per tant, pitjor salut). En canvi, els altres tres atributs, que són els que representen un esforç físic tenen correlació negativa. Això significa que quan disminueix el valor de l'atribut augmenta la classe (o a l'inrevés). Té sentit, ja que quantes menys abdominals puguis fer, més sentit té que la classe sigui major (més propera a 3, que és el pitjor).

També he volgut determinar la tendència de la salut depenent del gènere i el resultat ha estat que de les dades que es disposen, les persones de gènere masculí tendeixen a tenir pitjor salut i les persones de gènere femení tendeixen a tenir millor salut. 

## Preprocessat
El rang de valors que prenen tots els atributs és molt diferent, la qual cosa pot fer que els diferents models que s'apliquin no puguin classificar bé.
Per tal d'ajustar tots els atributs al mateix rang de dades, hauré de normalitzar les dades.

## Model
He provat cinc models diferents amb els seus paràmetres per defecte. He aplicat el k-fold amb k=5 i aquests són els seus rendiments:
|Model|Mètrica|
|--|--|
| SVM | 0.6951 |
| kNN | 0.5950 |
| Decision Tree | 0.6427 |
| Random Forest | 0.7358 |
| Logistic Regression | 0.6165 |

Es pot veure que el Random Forest és el model amb millors resultats.

També he buscat els millors hiperparàmetres de cadascun. Aquests han estat els resultats:

|Model|Hiperparàmetres|Mètrica|
|--|--|--|
| SVM | 'kernel': 'rbf' | 0.6951 |
| kNN | 'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'distance'| 0.6363 |
| Decision Tree | 'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 5 | 0.6834 |
| Random Forest | 'bootstrap': True, 'max_depth': 110, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 200 | 0.7406 |
| Logistic Regression | 'C': 0.1 | 0.6173 |

El rendiment dels models no ha incrementat gaire respecte els paràmetres per defecte. El millor resultat segueix sent el del Random Forest, amb un rendiment del 74%.

## Conclusions
1. Es pot dir que el nivell de salut d'una persona està directament relacionat amb el pes, el percentatge de grassa corporal, els centímetres que es pot inclinar cap endevant estant assegut i el nombre d'abdominals que és capaç de fer.
2. Hi ha una clara tendència a tenir pitjor salut si els resultats de les proves físiques son pitjors. En canvi, hi ha una tendència a tenir millor salut si el pes i el percentatge de grassa corporal és menor.
3. A partir de les dades proporcionades, es pot veure com les persones de gènere femení tenen una millor salut que les persones de gènere masculí.
4. Les dades estan representades en intervals molt diversos, tenint la necessitat de normalitzar-les per poder tractar-les correctament.
5. El millor model que classifica aquestes dades és el Random Forest, amb un score màxim de 74%. El SVM també funciona bé, amb un rendiment de pràcticament un 70%.
6. La cerca dels millors hiperparàmetres no ha millorat gaire el rendiment dels models. En concret, el Random Forest ha millorat solament un 1.6% respecte els valors per defecte.
