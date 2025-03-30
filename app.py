import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# === CONFIGURACIÓN GENERAL ===
archivo_excel = "EXCELINCIDENCIAS.xlsx"
hoja = "Sheet1"
archivo_salida = "datos_sinteticos.xlsx"
carpeta_graficas = "graficas"
cantidad_sintetica = 10000

# === CARGA Y PREPROCESAMIENTO ===
print("Cargando datos...")
df = pd.read_excel(archivo_excel, sheet_name=hoja)

# Convertir fechas a string (recomendado para evitar errores)
fechas = ["INICIO INCIDENCIA", "HORA DE LLEGADA", "CIERRE DE INCIDENCIA"]
for col in fechas:
    df[col] = df[col].astype(str)

# === CREAR METADATOS AUTOMÁTICAMENTE ===
print("Detectando metadatos...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# === ENTRENAMIENTO DEL MODELO ===
print("Entrenando CTGANSynthesizer...")
synthesizer = CTGANSynthesizer(metadata=metadata, epochs=400)
synthesizer.fit(df)
print("Entrenamiento completado.")

# === GENERACIÓN DE DATOS SINTÉTICOS ===
print(f"Generando {cantidad_sintetica} filas sintéticas...")
synthetic_data = synthesizer.sample(num_rows=cantidad_sintetica)

# === GUARDAR RESULTADO ===
synthetic_data.to_excel(archivo_salida, index=False)
print(f"Datos sintéticos guardados en: {archivo_salida}")

# === CREACIÓN DE CARPETA PARA GRÁFICAS ===
os.makedirs(carpeta_graficas, exist_ok=True)

# === COMPARACIÓN ESTADÍSTICA ===
print("\n=== Estadísticas comparativas ===")
columnas_numericas = df.select_dtypes(include=["number"]).columns
resumen = pd.DataFrame(columns=["Variable", "Conjunto", "Media", "Mediana", "Desviación estándar"])

for col in columnas_numericas:
    resumen = pd.concat([
        resumen,
        pd.DataFrame([
            {"Variable": col, "Conjunto": "Real", "Media": df[col].mean(), "Mediana": df[col].median(), "Desviación estándar": df[col].std()},
            {"Variable": col, "Conjunto": "Sintético", "Media": synthetic_data[col].mean(), "Mediana": synthetic_data[col].median(), "Desviación estándar": synthetic_data[col].std()}
        ])
    ], ignore_index=True)

print(resumen.to_string(index=False))

# === GRÁFICAS NUMÉRICAS ===
print("\nGenerando gráficas de distribución...")
for col in columnas_numericas:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[col], label='Real', fill=True)
    sns.kdeplot(synthetic_data[col], label='Sintético', fill=True)
    plt.title(f'Distribución de {col}')
    plt.legend()
    plt.savefig(f"{carpeta_graficas}/{col}_distribucion.png")
    plt.close()

# === GRÁFICAS CATEGÓRICAS ===
columnas_categoricas = df.select_dtypes(include=["object"]).columns
for col in columnas_categoricas:
    plt.figure(figsize=(12, 5))
    real_counts = df[col].value_counts(normalize=True)
    sint_counts = synthetic_data[col].value_counts(normalize=True)
    comparar = pd.DataFrame({'Real': real_counts, 'Sintético': sint_counts}).fillna(0)
    comparar.plot(kind="bar", width=0.8)
    plt.title(f'Frecuencia de {col}')
    plt.ylabel('Proporción')
    plt.tight_layout()
    plt.savefig(f"{carpeta_graficas}/{col}_frecuencia.png")
    plt.close()

print(f"\n✔ Todo listo. Gráficas guardadas en la carpeta '{carpeta_graficas}' y datos sintéticos en '{archivo_salida}'.")
