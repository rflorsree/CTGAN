import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


archivo_excel = "EXCELINCIDENCIAS.xlsx"
hoja = "Sheet1"
archivo_salida = "datos_sinteticos.xlsx"
carpeta_graficas = "graficas"
cantidad_sintetica = 10000



print("Cargando datos.")
df = pd.read_excel(archivo_excel, sheet_name=hoja)

#fechas a string  
fechas = ["INICIO INCIDENCIA", "HORA DE LLEGADA", "CIERRE DE INCIDENCIA"]
for col in fechas:
    df[col] = df[col].astype(str)


print("Detectando columnas")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# Entrenamiento
print("Entrenando CTGANS")
synthesizer = CTGANSynthesizer(metadata=metadata, epochs=400)
synthesizer.fit(df)
print("Entrenamiento completado.")

# Generacion de datos
print(f"Generando {cantidad_sintetica} ")
synthetic_data = synthesizer.sample(num_rows=cantidad_sintetica)

# ALmacena resultados en la carpeta
synthetic_data.to_excel(archivo_salida, index=False)
print(f"Datos sintéticos guardados en: {archivo_salida}")

# carpeta de imagenes
os.makedirs(carpeta_graficas, exist_ok=True)

# graficas de comparacion
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

# graficas de distribucion
print("\nGenerando gráficas de distribución...")
for col in columnas_numericas:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[col], label='Real', fill=True)
    sns.kdeplot(synthetic_data[col], label='Sintético', fill=True)
    plt.title(f'Distribución de {col}')
    plt.legend()
    plt.savefig(f"{carpeta_graficas}/{col}_distribucion.png")
    plt.close()

# graficas categoricas
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
