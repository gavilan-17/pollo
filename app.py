import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# Configuración de la página
st.set_page_config(
    page_title='Clasificador de Atletas',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Configuración global de matplotlib
matplotlib.use('Agg')
plt.style.use('ggplot')

# Crear directorio para modelos si no existe
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Cargar datos
@st.cache_data
def cargar_datos(archivo='atletas.csv'):
    """
    Carga y preprocesa los datos de los atletas.
    Maneja posibles errores en la carga de datos.
    """
    try:
        if not os.path.exists(archivo):
            st.error(f"El archivo {archivo} no existe.")
            return pd.DataFrame()
        
        df = pd.read_csv(archivo)
        
        # Verificar que el DataFrame tiene las columnas esperadas
        columnas_requeridas = ['Atleta', 'Edad', 'Peso', 'Volumen_O2_max', 'Umbral_lactato', '%Fibras_rapidas', '%Fibras_lentas']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            st.error(f"Faltan columnas en el archivo CSV: {', '.join(columnas_faltantes)}")
            return pd.DataFrame()
        
        # Convertir etiquetas a valores numéricos
        if 'Atleta' in df.columns:
            # Verificar si ya es numérico
            if not pd.api.types.is_numeric_dtype(df['Atleta']):
                # Mapear solo si contiene strings
                df['Atleta'] = df['Atleta'].map({'Fondista': 1, 'Velocista': 0})
        
        # Manejar valores nulos
        if df.isna().any().any():
            st.warning(f"Se han detectado {df.isna().sum().sum()} valores nulos en el dataset. Se eliminarán las filas afectadas.")
            filas_antes = len(df)
            df.dropna(inplace=True)
            st.warning(f"Se eliminaron {filas_antes - len(df)} filas con valores nulos.")
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

# Sidebar con controles
def add_sidebar():
    """
    Configura los controles en la barra lateral para el usuario.
    Incluye opciones avanzadas y básicas.
    """
    st.sidebar.title('Panel de Control')
    
    # Parámetros del modelo
    st.sidebar.header('Configuración del Modelo')
    
    modelo_seleccionado = st.sidebar.selectbox(
        "Selecciona un modelo", 
        ["Árbol de Decisión", "Regresión Logística", "Random Forest"]
    )
    
    # Parámetros específicos según el modelo seleccionado
    parametros = {}
    
    if modelo_seleccionado == "Árbol de Decisión":
        parametros['max_depth'] = st.sidebar.slider('Profundidad máxima', 1, 20, 3)
        parametros['criterion'] = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])
        parametros['min_samples_split'] = st.sidebar.slider('Muestras mínimas para dividir', 2, 20, 2)
    
    elif modelo_seleccionado == "Random Forest":
        parametros['n_estimators'] = st.sidebar.slider('Número de árboles', 10, 200, 100)
        parametros['max_depth'] = st.sidebar.slider('Profundidad máxima', 1, 20, 5)
        parametros['criterion'] = st.sidebar.selectbox('Criterio de división', ['gini', 'entropy'])
    
    elif modelo_seleccionado == "Regresión Logística":
        parametros['C'] = st.sidebar.slider('Inverso de regularización', 0.01, 10.0, 1.0)
        parametros['solver'] = st.sidebar.selectbox('Algoritmo', ['lbfgs', 'liblinear', 'newton-cg', 'sag'])

    # Opciones de entrenamiento
    test_size = st.sidebar.slider('Tamaño del conjunto de prueba (%)', 10, 40, 20) / 100
    standardize = st.sidebar.checkbox('Estandarizar datos', True)
    random_state = st.sidebar.number_input('Semilla aleatoria', 0, 999, 42)
    
    # Datos del atleta para predicción
    st.sidebar.header('Datos del Atleta')
    with st.sidebar.form("datos_atleta"):
        st.subheader("Ingrese los datos del atleta")
        edad = st.number_input("Edad", min_value=15, max_value=60, value=25)
        peso = st.number_input("Peso (kg)", min_value=40.0, max_value=120.0, value=70.0, step=0.1)
        vo2 = st.number_input("Volumen O2 máx (ml/kg/min)", min_value=20.0, max_value=90.0, value=50.0, step=0.1)
        lactato = st.number_input("Umbral de lactato (mmol/L)", min_value=2.0, max_value=6.0, value=4.0, step=0.1)
        fibras_rapidas = st.slider("% Fibras rápidas", 0, 100, 50)
        fibras_lentas = 100 - fibras_rapidas
        st.text(f"% Fibras lentas: {fibras_lentas}")
        
        submitted = st.form_submit_button("Realizar predicción")
    
    # Guardar/cargar modelo
    st.sidebar.header('Gestión de modelos')
    guardar_modelo = st.sidebar.button('Guardar modelo actual')
    cargar_modelo = st.sidebar.button('Cargar modelo guardado')
    
    return {
        'modelo_seleccionado': modelo_seleccionado,
        'parametros': parametros,
        'test_size': test_size,
        'standardize': standardize,
        'random_state': random_state,
        'atleta': {
            'edad': edad,
            'peso': peso,
            'vo2': vo2,
            'lactato': lactato,
            'fibras_rapidas': fibras_rapidas,
            'fibras_lentas': fibras_lentas
        },
        'submitted': submitted,
        'guardar_modelo': guardar_modelo,
        'cargar_modelo': cargar_modelo
    }

def crear_modelo(tipo_modelo, parametros):
    """
    Crea una instancia del modelo seleccionado con los parámetros especificados.
    """
    if tipo_modelo == "Árbol de Decisión":
        return DecisionTreeClassifier(
            max_depth=parametros.get('max_depth', 3),
            criterion=parametros.get('criterion', 'gini'),
            min_samples_split=parametros.get('min_samples_split', 2),
            random_state=parametros.get('random_state', 42)
        )
    elif tipo_modelo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=parametros.get('n_estimators', 100),
            max_depth=parametros.get('max_depth', 5),
            criterion=parametros.get('criterion', 'gini'),
            random_state=parametros.get('random_state', 42)
        )
    elif tipo_modelo == "Regresión Logística":
        return LogisticRegression(
            C=parametros.get('C', 1.0),
            solver=parametros.get('solver', 'lbfgs'),
            max_iter=1000,
            random_state=parametros.get('random_state', 42)
        )
    else:
        st.error(f"Modelo {tipo_modelo} no soportado")
        return None

def guardar_modelo_actual(model, scaler=None, tipo_modelo='model'):
    """
    Guarda el modelo entrenado y el scaler (si existe) para uso futuro.
    """
    try:
        # Guardar el modelo
        modelo_path = os.path.join(MODEL_DIR, f"{tipo_modelo}.joblib")
        joblib.dump(model, modelo_path)
        
        # Guardar el scaler si existe
        if scaler is not None:
            scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            
        st.success(f"¡Modelo {tipo_modelo} guardado con éxito!")
        return True
    except Exception as e:
        st.error(f"Error al guardar el modelo: {str(e)}")
        return False

def cargar_modelo_guardado(tipo_modelo='model'):
    """
    Carga un modelo previamente guardado y su scaler asociado si existe.
    """
    try:
        modelo_path = os.path.join(MODEL_DIR, f"{tipo_modelo}.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        
        if not os.path.exists(modelo_path):
            st.warning(f"No se encontró un modelo guardado ({tipo_modelo}.joblib)")
            return None, None
        
        model = joblib.load(modelo_path)
        
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
        st.success(f"¡Modelo {tipo_modelo} cargado con éxito!")
        return model, scaler
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

# Entrenamiento de modelos
def entrenar_modelo(df, config):
    """
    Preprocesa los datos, entrena el modelo especificado y retorna los resultados.
    """
    if df.empty:
        st.error("No hay datos para entrenar el modelo.")
        return None
    
    # Separar características y variable objetivo
    X = df.drop('Atleta', axis=1)
    y = df['Atleta']
    
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # Estandarización si es necesario
    scaler = None
    if config['standardize']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Convertimos a DataFrame para mantener los nombres de columnas
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # Crear y entrenar el modelo seleccionado
    parametros = config['parametros'].copy()
    parametros['random_state'] = config['random_state']
    
    model = crear_modelo(config['modelo_seleccionado'], parametros)
    
    if model is None:
        return None
    
    # Entrenamiento y predicción
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Probabilidades para curva ROC (si el modelo lo soporta)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred  # Fallback
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    resultados = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model': model,
        'scaler': scaler,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cv_scores': cv_scores
    }
    
    return resultados

# Predicción para un nuevo atleta
def predecir_atleta(config, model, scaler=None):
    """
    Realiza una predicción para un nuevo atleta con los datos proporcionados.
    """
    atleta = config['atleta']
    
    # Crear DataFrame con los datos del nuevo atleta
    nuevo = pd.DataFrame([
        [atleta['edad'], atleta['peso'], atleta['vo2'], atleta['lactato'], 
         atleta['fibras_rapidas'], atleta['fibras_lentas']]
    ], columns=['Edad', 'Peso', 'Volumen_O2_max', 'Umbral_lactato', '%Fibras_rapidas', '%Fibras_lentas'])
    
    # Aplicar el mismo preprocesamiento si es necesario
    if scaler is not None:
        nuevo_scaled = scaler.transform(nuevo)
        nuevo = pd.DataFrame(nuevo_scaled, columns=nuevo.columns)
    
    # Realizar predicción
    prediccion = model.predict(nuevo)[0]
    
    # Obtener probabilidades si es posible
    probabilidades = None
    if hasattr(model, "predict_proba"):
        probabilidades = model.predict_proba(nuevo)[0]
    
    tipo = "Fondista" if prediccion == 1 else "Velocista"
    
    return tipo, prediccion, probabilidades

# Detección de outliers
def detectar_outliers(X):
    """
    Detecta outliers en el conjunto de datos utilizando Isolation Forest.
    """
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = isolation_forest.fit_predict(X)
    return outliers == -1  # True para outliers

# Mostrar gráficos
def mostrar_graficos(resultados, config):
    """
    Muestra diversos gráficos relacionados con el modelo y los datos.
    """
    st.subheader("Visualización del Modelo y los Datos")
    
    # Distribución de clases
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        clase_counts = resultados['y_test'].value_counts()
        sns.barplot(x=clase_counts.index.map({1: 'Fondista', 0: 'Velocista'}), y=clase_counts.values, palette='viridis', ax=ax)
        ax.set_title('Distribución de Clases en Conjunto de Prueba')
        ax.set_xlabel('Tipo de Atleta')
        ax.set_ylabel('Cantidad')
        st.pyplot(fig)
        
    with col2:
        # Matriz de Confusión
        fig, ax = plt.subplots(figsize=(8, 5))
        conf_matrix = confusion_matrix(resultados['y_test'], resultados['y_pred'])
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=['Velocista', 'Fondista'],
                    yticklabels=['Velocista', 'Fondista'], ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)
    
    # Curva ROC
    fig, ax = plt.subplots(figsize=(8, 5))
    if all(isinstance(x, (int, float)) for x in resultados['y_prob']):
        fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax.set_title('Curva ROC')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    
    # Visualización del árbol si es un modelo de árbol
    if config['modelo_seleccionado'] in ["Árbol de Decisión"]:
        st.subheader("Visualización del Árbol de Decisión")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(resultados['model'], filled=True, feature_names=resultados['X_test'].columns,
                  class_names=["Velocista", "Fondista"], rounded=True, ax=ax, fontsize=10)
        st.pyplot(fig, use_container_width=True)
    
    # Importancia de características para árboles y random forest
    if config['modelo_seleccionado'] in ["Árbol de Decisión", "Random Forest"]:
        st.subheader("Importancia de Características")
        importances = resultados['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        features = resultados['X_test'].columns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis', ax=ax)
        ax.set_title('Importancia de Características')
        ax.set_xlabel('Importancia')
        st.pyplot(fig)
    
    # Para regresión logística, mostrar coeficientes
    if config['modelo_seleccionado'] == "Regresión Logística":
        st.subheader("Coeficientes de Regresión Logística")
        coef = resultados['model'].coef_[0]
        features = resultados['X_test'].columns
        
        coef_df = pd.DataFrame({'Característica': features, 'Coeficiente': coef})
        coef_df = coef_df.sort_values('Coeficiente', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coeficiente', y='Característica', data=coef_df, palette='viridis', ax=ax)
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.set_title('Coeficientes de Regresión Logística')
        st.pyplot(fig)
    
    # Gráfico de dispersión para variables seleccionadas
    st.subheader("Gráfico de Dispersión")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        x_var = st.selectbox('Variable X:', resultados['X_test'].columns, index=2)  # VO2 por defecto
        y_var = st.selectbox('Variable Y:', resultados['X_test'].columns, index=3)  # Lactato por defecto
    
    with col2:
        X_plot = resultados['X_test'].copy()
        X_plot['Tipo'] = resultados['y_test'].map({1: 'Fondista', 0: 'Velocista'})
        
        fig = px.scatter(X_plot, x=x_var, y=y_var, color='Tipo', 
                         color_discrete_map={'Fondista': 'blue', 'Velocista': 'red'},
                         title=f'Dispersión de {x_var} vs {y_var} por tipo de atleta')
        
        # Añadir línea de decisión si las variables son importantes en el modelo
        st.plotly_chart(fig, use_container_width=True)

# Mostrar métricas
def mostrar_metricas(resultados):
    """
    Muestra las métricas de rendimiento del modelo.
    """
    st.subheader('Métricas del Modelo')
    
    # Calcular métricas básicas
    accuracy = accuracy_score(resultados['y_test'], resultados['y_pred'])
    precision = precision_score(resultados['y_test'], resultados['y_pred'], average='weighted')
    recall = recall_score(resultados['y_test'], resultados['y_pred'], average='weighted')
    f1 = f1_score(resultados['y_test'], resultados['y_pred'], average='weighted')
    
    # Mostrar métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Exactitud", f"{accuracy:.3f}")
    
    with col2:
        st.metric("Precisión", f"{precision:.3f}")
    
    with col3:
        st.metric("Sensibilidad", f"{recall:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Mostrar resultados de validación cruzada
    cv_mean = resultados['cv_scores'].mean()
    cv_std = resultados['cv_scores'].std()
    
    st.subheader("Validación Cruzada (5-fold)")
    st.write(f"Precisión media: **{cv_mean:.3f}** ± **{cv_std:.3f}**")
    
    # Mostrar informe de clasificación completo
    st.subheader("Informe de Clasificación Detallado")
    report = classification_report(resultados['y_test'], resultados['y_pred'], 
                                  target_names=['Velocista', 'Fondista'],
                                  output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))

# Mostrar outliers
def mostrar_outliers(resultados):
    """
    Detecta y visualiza outliers en los datos.
    """
    st.subheader("Detección de Outliers")
    
    # Detectar outliers
    X_test = resultados['X_test'].copy()
    outliers_mask = detectar_outliers(X_test)
    outliers_count = np.sum(outliers_mask)
    
    st.write(f"Se detectaron **{outliers_count}** outliers de un total de **{len(X_test)}** muestras.")
    
    # Visualizar outliers en pares de características
    if outliers_count > 0:
        st.subheader("Visualización de Outliers")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            x_var = st.selectbox('Eje X:', X_test.columns, index=2, key='out_x')
            y_var = st.selectbox('Eje Y:', X_test.columns, index=3, key='out_y')
        
        with col2:
            X_plot = X_test.copy()
            X_plot['outlier'] = outliers_mask
            X_plot['outlier'] = X_plot['outlier'].map({True: 'Outlier', False: 'Normal'})
            
            fig = px.scatter(X_plot, x=x_var, y=y_var, color='outlier',
                             color_discrete_map={'Normal': 'blue', 'Outlier': 'red'},
                             title=f'Outliers en {x_var} vs {y_var}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla con los outliers detectados
        if st.checkbox("Mostrar datos de outliers"):
            st.write("Valores de los outliers detectados:")
            outlier_data = X_test[outliers_mask]
            outlier_data['Tipo'] = resultados['y_test'][outliers_mask].map({1: 'Fondista', 0: 'Velocista'})
            st.dataframe(outlier_data)

# Comparación de modelos
def mostrar_comparacion():
    """
    Compara el rendimiento de diferentes modelos.
    """
    st.subheader("Comparación de Modelos")
    
    df = cargar_datos()
    if df.empty:
        st.error("No hay datos para entrenar los modelos.")
        return
    
    # Separar características y variable objetivo
    X = df.drop('Atleta', axis=1)
    y = df['Atleta']
    
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a comparar con sus parámetros por defecto
    modelos = {
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
        'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Entrenar y evaluar cada modelo
    resultados = {}
    
    for nombre, modelo in modelos.items():
        # Entrenar modelo
        modelo.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test_scaled)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Validación cruzada
        cv_scores = cross_val_score(modelo, scaler.transform(X), y, cv=5)
        
        resultados[nombre] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Crear DataFrame para visualización
    resultados_df = pd.DataFrame(resultados).transpose()
    
    # Mostrar tabla de resultados
    st.write("Comparación de métricas entre modelos:")
    st.dataframe(resultados_df.style.format("{:.3f}"))
    
    # Visualizar en gráficos de barras
    metricas_plot = resultados_df.reset_index()
    metricas_plot = pd.melt(metricas_plot, id_vars=['index'], value_vars=['accuracy', 'precision', 'recall', 'f1', 'cv_mean'])
    metricas_plot.columns = ['Modelo', 'Métrica', 'Valor']
    
    fig = px.bar(metricas_plot, x='Modelo', y='Valor', color='Métrica', barmode='group',
                title='Comparación de Métricas por Modelo')
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de radar para comparación visual intuitiva
    categorias = ['Exactitud', 'Precisión', 'Sensibilidad', 'F1', 'Val. Cruzada']
    
    fig = go.Figure()
    
    for modelo in resultados.keys():
        valores = [
            resultados[modelo]['accuracy'],
            resultados[modelo]['precision'],
            resultados[modelo]['recall'],
            resultados[modelo]['f1'],
            resultados[modelo]['cv_mean']
        ]
        # Repetir el primer valor para cerrar el polígono
        valores.append(valores[0])
        categorias_cerradas = categorias + [categorias[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias_cerradas,
            fill='toself',
            name=modelo
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Diagrama de Radar - Comparación de Modelos"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Página de ayuda y documentación    
def mostrar_ayuda():
    """
    Muestra información de ayuda sobre la aplicación y su uso.
    """
    st.header("Ayuda y Documentación")
    
    st.markdown("""
    ## Sobre la aplicación
    
    Esta aplicación permite clasificar atletas como **Velocistas** o **Fondistas** basándose en sus características físicas y fisiológicas. Utiliza algoritmos de aprendizaje automático para realizar la predicción.
    
    ## Cómo usar la aplicación
    
    1. **Cargar datos**: La aplicación utiliza un archivo CSV llamado 'atletas.csv' que debe contener las siguientes columnas:
       - Atleta: Categoría del atleta ('Fondista' o 'Velocista')
       - Edad: Edad del atleta en años
       - Peso: Peso en kilogramos
       - Volumen_O2_max: Consumo máximo de oxígeno (ml/kg/min)
       - Umbral_lactato: Umbral de lactato en mmol/L
       - %Fibras_rapidas: Porcentaje de fibras musculares rápidas
       - %Fibras_lentas: Porcentaje de fibras musculares lentas
    
    2. **Configurar el modelo**: En el panel lateral, seleccione el tipo de modelo y ajuste sus parámetros según sus necesidades.
    
    3. **Ingresar datos del atleta**: Complete el formulario con los datos del atleta que desea clasificar.
    
    4. **Explorar resultados**: Navegue por las diferentes pestañas para ver métricas, gráficos y análisis detallados.
    
    5. **Guardar/Cargar modelo**: Puede guardar el modelo entrenado para usarlo posteriormente.
    
    ## Descripción de los modelos
    
    - **Árbol de Decisión**: Modelo sencillo que divide los datos en regiones basadas en umbrales de características.
    - **Regresión Logística**: Modelo estadístico que estima la probabilidad de pertenencia a cada clase.
    - **Random Forest**: Conjunto de árboles de decisión que mejora la precisión y reduce el sobreajuste.
    
    ## Interpretación de resultados
    
    - **Exactitud (Accuracy)**: Proporción de predicciones correctas.
    - **Precisión**: Capacidad del modelo para no etiquetar como positiva una muestra negativa.
    - **Sensibilidad (Recall)**: Capacidad del modelo para encontrar todas las muestras positivas.
    - **F1-Score**: Media armónica de precisión y sensibilidad.
    
    ## Detección de outliers
    
    Los outliers son observaciones atípicas que pueden afectar al rendimiento del modelo. La aplicación utiliza el algoritmo Isolation Forest para detectarlos.
    """)
    
    # Añadir información de contacto
    st.markdown("""
    ## Contacto y soporte
    
    Para cualquier duda o sugerencia, contacte con el equipo de desarrollo.
    """)

# Página principal
def main():
    """
    Función principal que estructura la aplicación.
    """
    # Título y descripción
    st.title('🏃‍♂️ Clasificador de Atletas')
    st.markdown("""
    Esta aplicación permite clasificar atletas como **Velocistas** o **Fondistas** basándose en sus características físicas y fisiológicas.
    Utiliza algoritmos de machine learning para realizar predicciones precisas.
    """)
    
    # Cargar datos
    df = cargar_datos()
    
    # Si no hay datos, mostrar error y detener ejecución
    if df.empty:
        st.error("No se pueden cargar los datos. Por favor, verifica que el archivo 'atletas.csv' existe y tiene el formato correcto.")
        st.stop()
    
    # Mostrar estadísticas básicas del dataset
    with st.expander("Ver estadísticas del dataset"):
        st.write(f"**Número de registros**: {len(df)}")
        st.write(f"**Distribución de clases**:")
        clase_counts = df['Atleta'].value_counts()
        clase_labels = {1: 'Fondista', 0: 'Velocista'}
        distribución = pd.Series({clase_labels[k]: v for k, v in clase_counts.items()})
        st.bar_chart(distribución)
        
        st.write("**Estadísticas descriptivas**:")
        st.dataframe(df.describe())
        
        # Mostrar correlaciones
        st.write("**Matriz de correlación**:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    
    # Configuración desde sidebar
    config = add_sidebar()
    
    # Pestañas de navegación
    tabs = st.tabs([
        "💡 Predicción", 
        "📊 Gráficos", 
        "📈 Métricas", 
        "🔍 Outliers", 
        "🔄 Comparación", 
        "❓ Ayuda"
    ])
    
    # Variable para almacenar resultados del entrenamiento
    if 'resultados' not in st.session_state:
        st.session_state.resultados = None
        st.session_state.modelo_actual = None
        st.session_state.scaler = None
    
    # Gestión de modelos (guardar/cargar)
    if config['guardar_modelo'] and st.session_state.resultados:
        guardar_modelo_actual(
            st.session_state.resultados['model'], 
            st.session_state.resultados['scaler'], 
            config['modelo_seleccionado']
        )
    
    if config['cargar_modelo']:
        loaded_model, loaded_scaler = cargar_modelo_guardado(config['modelo_seleccionado'])
        if loaded_model:
            st.session_state.modelo_actual = loaded_model
            st.session_state.scaler = loaded_scaler
    
    # Entrenar modelo si no hay uno cargado o si se cambian parámetros
    if st.session_state.modelo_actual is None or st.checkbox("Reentrenar modelo", value=False):
        with st.spinner('Entrenando modelo...'):
            resultados = entrenar_modelo(df, config)
            if resultados:
                st.session_state.resultados = resultados
                st.session_state.modelo_actual = resultados['model']
                st.session_state.scaler = resultados['scaler']
    
    # Tab 1: Predicción
    with tabs[0]:
        st.header("Predicción de Tipo de Atleta")
        
        if st.session_state.modelo_actual is None:
            st.warning("No hay modelo entrenado o cargado. Por favor, entrene un modelo primero.")
        else:
            if config['submitted']:
                tipo, _, probabilidades = predecir_atleta(config, st.session_state.modelo_actual, st.session_state.scaler)
                
                # Mostrar resultado con estilo
                st.subheader("Resultado de la predicción")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if tipo == "Velocista":
                        st.image("https://img.icons8.com/color/96/000000/running.png", width=100)
                    else:
                        st.image("https://img.icons8.com/color/96/000000/marathon.png", width=100)
                
                with col2:
                    st.markdown(f"### El atleta es un **{tipo}**")
                    
                    if probabilidades is not None:
                        prob_velocista = probabilidades[0] * 100
                        prob_fondista = probabilidades[1] * 100
                        
                        st.write(f"Probabilidad de ser Velocista: **{prob_velocista:.2f}%**")
                        st.write(f"Probabilidad de ser Fondista: **{prob_fondista:.2f}%**")
                        
                        # Gráfico de confianza
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh([''], [prob_velocista], color='red', label='Velocista')
                        ax.barh([''], [prob_fondista], left=[prob_velocista], color='blue', label='Fondista')
                        ax.set_xlim(0, 100)
                        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
                        st.pyplot(fig)
                
                # Interpretación del resultado
                st.subheader("Interpretación")
                
                if tipo == "Velocista":
                    st.write("""
                    Los **velocistas** suelen tener:
                    - Mayor proporción de fibras musculares rápidas
                    - Menor volumen de O2 máximo
                    - Mayor umbral de lactato
                    - Mejor rendimiento en esfuerzos explosivos de corta duración
                    """)
                else:
                    st.write("""
                    Los **fondistas** suelen tener:
                    - Mayor proporción de fibras musculares lentas
                    - Mayor volumen de O2 máximo
                    - Menor umbral de lactato
                    - Mejor rendimiento en esfuerzos prolongados de resistencia
                    """)
    
    # Tab 2: Gráficos
    with tabs[1]:
        if st.session_state.resultados:
            mostrar_graficos(st.session_state.resultados, config)
        else:
            st.warning("Entrene un modelo primero para visualizar los gráficos.")
    
    # Tab 3: Métricas
    with tabs[2]:
        if st.session_state.resultados:
            mostrar_metricas(st.session_state.resultados)
        else:
            st.warning("Entrene un modelo primero para ver las métricas.")
    
    # Tab 4: Outliers
    with tabs[3]:
        if st.session_state.resultados:
            mostrar_outliers(st.session_state.resultados)
        else:
            st.warning("Entrene un modelo primero para detectar outliers.")
    
    # Tab 5: Comparación de modelos
    with tabs[4]:
        mostrar_comparacion()
    
    # Tab 6: Ayuda
    with tabs[5]:
        mostrar_ayuda()
    
    # Pie de página
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Desarrollado para análisis deportivo | Versión 2.0</p>
    </div>
    """, unsafe_allow_html=True)

# Ejecutar la aplicación cuando se ejecute el script
if __name__ == '__main__':
    main()