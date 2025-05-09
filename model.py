from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


# Cargar conjunto de datos de ejemplo (Iris)
df=pd.read_csv('atletas.csv')
X=df[['Edad','Peso','Volumen_O2_max']]
y=df['Atleta']


# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Definir y entrenar el modelo de Árbol de Decisión
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Guardar el modelo entrenado
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
