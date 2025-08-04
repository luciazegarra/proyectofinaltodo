# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:21:40 2025

@author: zegar
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Configuración inicial
st.set_page_config(page_title="Análisis Estadístico", layout="centered")
st.title("📊 Análisis de Satisfacción de Vida")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

dataset = cargar_datos()
st.write("Vista previa de los datos:")
st.dataframe(dataset.head())

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

# Análisis de distribución de Género
st.subheader("📌 Distribución de Género")
gender_counts = dataset['Genero'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
ax1.axis("equal")
st.pyplot(fig1)

# Análisis de Nivel Educativo
st.subheader("🎓 Nivel Educativo")
nivel_educativo_percent = dataset['Nivel_Educativo'].value_counts(normalize=True) * 100
fig2, ax2 = plt.subplots()
sns.barplot(x=nivel_educativo_percent.index, y=nivel_educativo_percent.values, palette='viridis', ax=ax2)
ax2.set_ylabel("Porcentaje (%)")
ax2.set_title("Distribución de Nivel Educativo")
st.pyplot(fig2)

# Histogramas y Boxplots
st.subheader("📉 Distribuciones y Boxplots")
ds_num = dataset.select_dtypes(include=['float64', 'int64'])
for col in ds_num.columns:
    st.write(f"**{col}**")
    fig3, ax3 = plt.subplots()
    sns.histplot(dataset[col], kde=True, ax=ax3, color='skyblue')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.boxplot(x=dataset[col], ax=ax4, color='lightgreen')
    st.pyplot(fig4)

# Modelado: Regresión Lineal Múltiple
st.subheader("📈 Regresión Lineal Múltiple")
x = dataset[numeric_cols]
y = dataset['Satisfaccion_Vida']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

modelo_rl = LinearRegression()
modelo_rl.fit(x_train_scaled, y_train)
y_pred = modelo_rl.predict(x_test_scaled)

st.write("**R² Score:**", round(r2_score(y_test, y_pred), 4))
st.write("**MSE:**", round(mean_squared_error(y_test, y_pred), 2))

# Gráfico de predicción vs realidad
fig5, ax5 = plt.subplots()
ax5.scatter(y_test, y_pred, alpha=0.5)
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax5.set_xlabel("Real")
ax5.set_ylabel("Predicho")
ax5.set_title("Regresión Lineal - Real vs Predicho")
st.pyplot(fig5)

# Clasificación KNN
st.subheader("🔍 Clasificación con KNN")

edad = st.slider("Edad", 18, 80, 30)
ingreso = st.slider("Ingreso mensual", 500, 10000, 2500)
horas_estudio = st.slider("Horas de estudio semanales", 0, 50, 10)

nuevo_dato = [[edad, ingreso, horas_estudio]]
nuevo_dato_scaled = scaler.transform(nuevo_dato)

modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(x_train_scaled, y_train)
prediccion_knn = modelo_knn.predict(nuevo_dato_scaled)

st.success(f"✅ Predicción de Satisfacción de Vida para el nuevo dato: **{prediccion_knn[0]}**")
