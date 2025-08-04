# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:21:40 2025
@author: zegar
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(page_title="Análisis Estadístico", layout="centered")
st.title("📊 Análisis de Satisfacción de Vida")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

try:
    dataset = cargar_datos()
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset_estadistica.csv'.")
    st.stop()

st.write("Vista previa de los datos:")
st.dataframe(dataset.head())

# Estadística descriptiva
st.subheader("📊 Estadística Descriptiva de Variables Numéricas")
ds_num = dataset.select_dtypes(include=['float64', 'int64'])
st.dataframe(ds_num.describe())

# Imputación de valores nulos
numeric_cols = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']
categorical_cols = ['Nivel_Educativo', 'Genero']

numeric_imputer = SimpleImputer(strategy="mean")
dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

categorical_imputer = SimpleImputer(strategy="most_frequent")
dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

# Codificación de la variable objetivo
bins = [0, 4, 7, 10]
labels = ['Baja', 'Media', 'Alta']
dataset["Satisfaccion_Clase"] = pd.cut(dataset["Satisfaccion_Vida"], bins=bins, labels=labels, include_lowest=True)

# Distribución de Género
st.subheader("📌 Distribución de Género")
gender_counts = dataset['Genero'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
ax1.axis("equal")
st.pyplot(fig1)

# Nivel Educativo
st.subheader("🎓 Nivel Educativo")
nivel_educativo_percent = dataset['Nivel_Educativo'].value_counts(normalize=True) * 100
fig2, ax2 = plt.subplots()
sns.barplot(x=nivel_educativo_percent.index, y=nivel_educativo_percent.values, palette='viridis', ax=ax2)
ax2.set_ylabel("Porcentaje (%)")
ax2.set_title("Distribución de Nivel Educativo")
st.pyplot(fig2)

# Histogramas y Boxplots
st.subheader("📉 Distribuciones y Boxplots")
for col in ds_num.columns:
    st.write(f"**{col}**")
    fig3, ax3 = plt.subplots()
    sns.histplot(dataset[col], kde=True, ax=ax3, color='skyblue')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.boxplot(x=dataset[col], ax=ax4, color='lightgreen')
    st.pyplot(fig4)

# Matriz de correlación
st.subheader("📈 Matriz de correlación")
fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
sns.heatmap(ds_num.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Pairplot
st.subheader("🔍 Relaciones entre variables numéricas")
fig_pair = sns.pairplot(dataset[numeric_cols + ['Satisfaccion_Vida']])
fig_pair.fig.suptitle("Relaciones entre variables numéricas", y=1.02)
st.pyplot(fig_pair.fig)

# Selección de modelo
st.sidebar.header("1️⃣ Selecciona el modelo de predicción")
modelo_seleccionado = st.sidebar.selectbox("Modelo", ["Regresión Lineal", "KNN"])

# Variables y escalado
x = dataset[numeric_cols]
y_regresion = dataset['Satisfaccion_Vida']
y_clasificacion = dataset['Satisfaccion_Clase']

x_train, x_test, y_train_reg, y_test_reg = train_test_split(x, y_regresion, test_size=0.2, random_state=0)
_, _, y_train_clf, y_test_clf = train_test_split(x, y_clasificacion, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Modelo: Regresión
if modelo_seleccionado == "Regresión Lineal":
    st.subheader("📈 Regresión Lineal Múltiple")
    modelo_rl = LinearRegression()
    modelo_rl.fit(x_train_scaled, y_train_reg)
    y_pred = modelo_rl.predict(x_test_scaled)

    st.write("**R² Score:**", round(r2_score(y_test_reg, y_pred), 4))
    st.write("**MSE:**", round(mean_squared_error(y_test_reg, y_pred), 2))

    fig5, ax5 = plt.subplots()
    ax5.scatter(y_test_reg, y_pred, alpha=0.5)
    ax5.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
    ax5.set_xlabel("Real")
    ax5.set_ylabel("Predicho")
    ax5.set_title("Regresión Lineal - Real vs Predicho")
    st.pyplot(fig5)

# Modelo: KNN Clasificación
elif modelo_seleccionado == "KNN":
    st.subheader("🔍 Clasificación con KNN")

    # Inputs del usuario
    edad = st.slider("Edad", 18, 80, 30)
    ingreso = st.slider("Ingreso mensual", 500, 10000, 2500)
    horas_estudio = st.slider("Horas de estudio semanales", 0, 50, 10)

    nuevo_dato = [[edad, ingreso, horas_estudio]]
    nuevo_dato_scaled = scaler.transform(nuevo_dato)

    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(x_train_scaled, y_train_clf)
    prediccion_knn = modelo_knn.predict(nuevo_dato_scaled)

    st.success(f"✅ Predicción de clase de Satisfacción de Vida: **{prediccion_knn[0]}**")

    # Visualización 2D de datos originales y nuevo punto
    st.subheader("🌐 Visualización de los datos en el espacio original")

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))

    class_color_map = {'Baja': 0, 'Media': 1, 'Alta': 2}
    colores = y_clasificacion.map(class_color_map)

    scatter = ax_scatter.scatter(
        x['Edad'],
        x['Ingreso_Mensual'],
        c=colores,
        cmap='viridis',
        edgecolor='k',
        alpha=0.7
    )

    ax_scatter.scatter(
        nuevo_dato[0][0],
        nuevo_dato[0][1],
        color='red',
        marker='X',
        s=120,
        label=f"Nuevo Dato ({nuevo_dato[0][0]}, {nuevo_dato[0][1]})"
    )

    ax_scatter.set_xlabel("Edad")
    ax_scatter.set_ylabel("Ingreso Mensual")
    ax_scatter.set_title("Distribución de datos y nuevo punto")

    legend_labels = list(class_color_map.keys())
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                 markerfacecolor=plt.cm.viridis(val / 2), markersize=10, markeredgecolor='k')
                      for label, val in class_color_map.items()]
    legend_handles.append(plt.Line2D([0], [0], marker='X', color='red', label='Nuevo Dato', markersize=10))

    ax_scatter.legend(handles=legend_handles, title="Clases de Satisfacción")
    st.pyplot(fig_scatter)

    # Vecinos más cercanos
    distancias, indices = modelo_knn.kneighbors(nuevo_dato_scaled)
    vecinos = pd.DataFrame(x_train.iloc[indices[0]])
    vecinos["Clase_Satisfacción"] = y_train_clf.iloc[indices[0]].values
    st.write("👥 Vecinos más cercanos:")
    st.dataframe(vecinos)

    # Evaluación
    y_pred_knn = modelo_knn.predict(x_test_scaled)
    st.subheader("📌 Evaluación del Modelo KNN")
    cm = confusion_matrix(y_test_clf, y_pred_knn)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Matriz de Confusión")
    ax_cm.set_xlabel("Predicho")
    ax_cm.set_ylabel("Real")
    st.pyplot(fig_cm)

    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test_clf, y_pred_knn))
