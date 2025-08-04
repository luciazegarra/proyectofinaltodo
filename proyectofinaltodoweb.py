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

# 1. Configuración inicial
st.set_page_config(page_title="Análisis Estadístico", layout="centered")
st.title("📊 Análisis de Satisfacción de Vida")

# 2. Carga de datos
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

# 3. Estadística descriptiva
st.subheader("📊 Estadística Descriptiva de Variables Numéricas")
ds_num = dataset.select_dtypes(include=['float64', 'int64'])
st.dataframe(ds_num.describe())

# 4. Imputación de valores nulos
numeric_cols = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']
categorical_cols = ['Nivel_Educativo', 'Genero']

numeric_imputer = SimpleImputer(strategy="mean")
dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

categorical_imputer = SimpleImputer(strategy="most_frequent")
dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

# 5. Conversión de variable objetivo a clases categóricas
# Definir umbrales para clases (ajustar si es necesario)
bins = [0, 4, 7, 10]
labels = ['Baja', 'Media', 'Alta']
dataset["Satisfaccion_Clase"] = pd.cut(dataset["Satisfaccion_Vida"], bins=bins, labels=labels, include_lowest=True)

# 6. Distribución de Género
st.subheader("📌 Distribución de Género")
gender_counts = dataset['Genero'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
ax1.axis("equal")
st.pyplot(fig1)

# 7. Nivel Educativo
st.subheader("🎓 Nivel Educativo")
nivel_educativo_percent = dataset['Nivel_Educativo'].value_counts(normalize=True) * 100
fig2, ax2 = plt.subplots()
sns.barplot(x=nivel_educativo_percent.index, y=nivel_educativo_percent.values, palette='viridis', ax=ax2)
ax2.set_ylabel("Porcentaje (%)")
ax2.set_title("Distribución de Nivel Educativo")
st.pyplot(fig2)

# 8. Histogramas y Boxplots
st.subheader("📉 Distribuciones y Boxplots")
for col in ds_num.columns:
    st.write(f"**{col}**")
    fig3, ax3 = plt.subplots()
    sns.histplot(dataset[col], kde=True, ax=ax3, color='skyblue')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.boxplot(x=dataset[col], ax=ax4, color='lightgreen')
    st.pyplot(fig4)

# 9. Matriz de correlación
st.subheader("📈 Matriz de correlación")
fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
sns.heatmap(ds_num.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# 10. Pairplot - Relaciones entre variables
st.subheader("🔍 Relaciones entre variables numéricas")
fig_pair = sns.pairplot(dataset[numeric_cols + ['Satisfaccion_Vida']])
fig_pair.fig.suptitle("Relaciones entre variables numéricas", y=1.02)
st.pyplot(fig_pair.fig)

# 11. Sidebar: Selección de modelo
st.sidebar.header("1️⃣ Selecciona el modelo de predicción")
modelo_seleccionado = st.sidebar.selectbox("Modelo", ["Regresión Lineal", "KNN"])

# 12. Variables predictoras y objetivo
x = dataset[numeric_cols]
y_regresion = dataset['Satisfaccion_Vida']
y_clasificacion = dataset['Satisfaccion_Clase']

# 13. División de datos y escalado
x_train, x_test, y_train_reg, y_test_reg = train_test_split(x, y_regresion, test_size=0.2, random_state=0)
_, _, y_train_clf, y_test_clf = train_test_split(x, y_clasificacion, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 14. REGRESIÓN LINEAL
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

# 15. KNN CLASIFICACIÓN
elif modelo_seleccionado == "KNN":
    st.subheader("🔍 Clasificación con KNN")

    edad = st.slider("Edad", 18, 80, 30)
    ingreso = st.slider("Ingreso mensual", 500, 10000, 2500)
    horas_estudio = st.slider("Horas de estudio semanales", 0, 50, 10)

    nuevo_dato = [[edad, ingreso, horas_estudio]]
    nuevo_dato_scaled = scaler.transform(nuevo_dato)

    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(x_train_scaled, y_train_clf)
    prediccion_knn = modelo_knn.predict(nuevo_dato_scaled)

    st.success(f"✅ Predicción de clase de Satisfacción de Vida: **{prediccion_knn[0]}**")

    # Vecinos más cercanos
    distancias, indices = modelo_knn.kneighbors(nuevo_dato_scaled)
    vecinos = pd.DataFrame(x_train.iloc[indices[0]])
    vecinos["Clase_Satisfacción"] = y_train_clf.iloc[indices[0]].values
    st.write("👥 Vecinos más cercanos:")
    st.dataframe(vecinos)

    # Evaluación del modelo
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
