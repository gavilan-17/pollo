import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree




Glucose = [150, 140, 140, 120, 80, 70, 90, 70]
Age = [50, 45, 60, 70, 75, 40, 80, 80]
Outcome = [1, 0, 1, 1, 0, 0, 0, 0]




df = pd.DataFrame({'Glucose': Glucose, 'Age': Age, 'Outcome': Outcome})


X=df[['Glucose','Age']].values
y=df['Outcome'].values


ax = sns.scatterplot(data=df, x='Glucose', y='Age', hue='Outcome', color='black', palette='viridis')
ax.axvline(x=120, ymin=0, ymax=1, color='red', linestyle='dashed')
ax.axhline(y=33, xmin=0, xmax=1, color='black', linestyle='dashed')
plt.legend(['enfermos', 'sanos'])
plt.show()


I = (df['Glucose'] <= 120).sum()
D = (df['Glucose'] > 120).sum()
print('Glucose con valor <=120, rama izquierda:', I)
print('Glucose con valor >120, rama derecha:', D)


print('Probabilidad de tener Glucosa<=120:', I / (I + D))
print('Probabilidad de tener Glucosa>120:', D / (I + D))


PI = round(I / (I + D), 2)
PD = round(D / (I + D), 2)


#Rama Izquierda
sanos_I = (df['Glucose'] <= 120) & (df['Outcome'] == 0)
enfermos_I = (df['Glucose'] <= 120) & (df['Outcome'] == 1)
print('Pacientes con Glucosa <=120 clasificados como sanos:', sanos_I.sum())
print('Pacientes con Glucosa <=120 clasificados como diabéticos:', enfermos_I.sum())


C1_I = sanos_I.sum()
C2_I = enfermos_I.sum()
print('Probabilidad de ser clasificado como sano en esta rama:', round(C1_I / I, 2))
print('Probabilidad de ser clasificado como enfermo en esta rama:', round(C2_I / I, 2))


#Rama Derecha
sanos_D = (df['Glucose'] > 120) & (df['Outcome'] == 0)
enfermos_D = (df['Glucose'] > 120) & (df['Outcome'] == 1)
print('Pacientes con Glucosa >120 clasificados como sanos:', sanos_D.sum())
print('Pacientes con Glucosa >120 clasificados como diabéticos:', enfermos_D.sum())


C1_D = sanos_D.sum()
C2_D = enfermos_D.sum()
print('Probabilidad de ser clasificado como sano en esta rama:', round(C1_D / D, 2))
print('Probabilidad de ser clasificado como enfermo en esta rama:', round(C2_D / D, 2))




def Gini(C1, C2):
    GiniX = 2 * (C1 / (C1 + C2)) * (C2 / (C1 + C2))
    return GiniX




def Gini_pond(GI, GD, PI, PD):
    Gini_p = PI * GI + PD * GD
    return Gini_p




GiniI = Gini(C1_I, C2_I)
GiniD = Gini(C1_D, C2_D)
Gini_ini = Gini(PI, PD)




print('Gini de la rama izquierda es:', round(GiniI, 2))
print('Gini de la rama derecha es:', round(GiniD, 2))
print('Gini de la rama derecha es:', round(Gini_ini, 2))


Coste = Gini_pond(GiniI, GiniD, PI, PD)
print('El Gini de la partición o función de coste es:', round(Coste, 2))
print('La Ganancia de información es:', round((Gini_ini - Coste), 2))


tree= DecisionTreeClassifier(criterion='entropy', random_state=0,).fit(X,y)


print(tree.predict([[90, 50]]))
plt.figure()
plt.figure(figsize=(8,12))
plot_tree(tree, max_depth=2, filled=True)
plt.show()
