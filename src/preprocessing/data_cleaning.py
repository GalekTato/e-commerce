import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



try:
    print("Inicio de Cargar Datos")
    df = pd.read_csv('../../data/raw/diversified_ecommerce_dataset.csv', encoding='latin1')
    df.columns = df.columns.str.strip() #quitar vacios

    #creamos cuartiles para la casificascion
    q33 = df['Popularity Index'].quantile(0.33)
    q66 = df['Popularity Index'].quantile(0.66)

    def generar_clases(valor):
        if valor <= q33:
            return 0  # Baja Popularidad / Rotación lenta
        elif valor <= q66:
            return 1  # Popularidad Moderada
        else:
            return 2  # Alta Popularidad / Producto Estrella

    df['Target_Clase'] = df['Popularity Index'].apply(generar_clases)

    
    columnas_categoricas = ['Category', 'Customer Age Group', 'Customer Gender', 'Shipping Method', 'Seasonality']
    df = pd.get_dummies(df, columns=columnas_categoricas, dtype=int)
    df = df.fillna(0)
    
    columnas_data = ['Price', 'Stock Level', 'Shipping Cost', 'Popularity Index']
    
    print("Aplicando Estandarización (Media=0, Desviación=1) a los descriptores...")
    scaler = StandardScaler() 
    df[columnas_data] = scaler.fit_transform(df[columnas_data])

    scaler = StandardScaler() 
    df[columnas_data] = scaler.fit_transform(df[columnas_data])

    #como estos manejas valores entre 0 y 1 veremos su comportamiento   
    columnas_data.append('Discount')
    columnas_data.append('Tax Rate')
    columnas_data.append('Return Rate')

    #momento medio vivecoding y dolor d emi cabeza 
    print("Generando visualización de datos estandarizados...")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 9))
    axes = axes.flatten()

    for i, col in enumerate(columnas_data):
        df[col].plot(
            kind='hist', 
            bins=40, 
            ax=axes[i], 
            color='lightgreen', 
            edgecolor='black',
            title=f'{col} (Z-Score)'
        )
        axes[i].set_xlabel('Valor Z')
        axes[i].set_ylabel('Frecuencia')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1.5) # Línea en la media
        
    plt.suptitle('Histogramas de Descriptores Numéricos con Media Centrada (Z-Score)', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

    #Borramos valores inutiles no numericos 

    data_inutl = ['Product ID','Product Name','Customer Location']
    df = df.drop(columns=[col for col in data_inutl if col in df.columns], errors='ignore')

    X = df
    # y toma únicamente la clasificación objetivo que creamos
    y = df['Target_Clase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


    print("Data guardada en data/processed/dataset_listo_para_rna.csv")
    X_train.to_csv('../../data/processed/X_train.csv', index=False)
    X_test.to_csv('../../data/processed/X_test.csv', index=False)
    y_train.to_csv('../../data/processed/y_train.csv', index=False)
    y_test.to_csv('../../data/processed/y_test.csv', index=False)
    df.to_csv('../../data/processed/dataset_listo_para_rna.csv', index=False)


except Exception as e:
    print(f"Error: {e}")