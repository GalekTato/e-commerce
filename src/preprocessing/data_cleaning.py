import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
=============================================================================
Dataset name: NASA Kepler Exoplanet Search Results
Origin: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results
(also available via NASA Exoplanet Archive)

Why it was created: To catalog and classify candidate exoplanet transits 
detected by the Kepler space telescope between 2009–2018.

What we are solving: Multiclass classification — determine whether a Kepler 
Object of Interest is a confirmed exoplanet, a false positive, or an 
unresolved candidate.

Features used: Photometric transit parameters (period, duration, depth, 
impact parameter, SNR), planetary parameters (radius, equilibrium temperature, 
insolation flux), and stellar parameters (effective temperature, surface 
gravity, radius, magnitude).

Class distribution: CONFIRMED ~24%, CANDIDATE ~24%, FALSE POSITIVE ~52%
=============================================================================
"""

def load_and_preprocess():
    print("Iniciando carga y preprocesamiento de NASA Kepler Exoplanet Search Results...")
    
    data_paths = [
        '../../data/raw/cumulative.csv',
        'data/raw/cumulative.csv',
        'cumulative.csv'
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
            
    if df is None:
        raise FileNotFoundError("No se encontró el archivo cumulative.csv en las rutas esperadas.")

    cols_to_drop = [
        'rowid', 'kepid', 'kepoi_name', 'kepler_name', 
        'koi_pdisposition', 'koi_tce_delivname', 
        'koi_teq_err1', 'koi_teq_err2'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    target_map = {
        'CONFIRMED': 0,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 2
    }
    
    if 'koi_disposition' not in df.columns:
        raise ValueError("La columna objetivo 'koi_disposition' no se encuentra en el dataset.")
        
    y = df['koi_disposition'].map(target_map)
    df = df.drop(columns=['koi_disposition'])
    
    threshold = 0.6 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    
    nan_counts_before = df.isna().sum()
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    if df.isna().any().any():
        raise ValueError("Aún existen valores NaN en el dataset después de la imputación.")
        
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # ---------------------------------------------------------
    # GENERACIÓN DEL HISTOGRAMA CORRECTO (FIX 2)
    # ---------------------------------------------------------
    print("Generando visualización de datos estandarizados del Dataset Kepler...")
    features_to_plot = [
        'koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad', 
        'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 'koi_slogg', 
        'koi_srad', 'koi_kepmag', 'ra', 'dec', 'koi_score'
    ]
    # Filtrar solo los que existen (deberían existir todos) y rellenar hasta 16
    features_to_plot = [f for f in features_to_plot if f in X_train_scaled.columns][:16]
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        X_train_scaled[col].plot(
            kind='hist', bins=40, ax=axes[i], color='lightgreen', edgecolor='black'
        )
        axes[i].set_title(f'{col}', fontsize=10)
        axes[i].set_xlabel('Z-Score')
        axes[i].set_ylabel('Frecuencia')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1.5)
        
    # Ocultar subplots vacíos
    for j in range(len(features_to_plot), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Histogramas de Descriptores Numéricos del Dataset Kepler (Z-Score)', fontsize=16, weight='bold')
    plt.tight_layout()
    
    plots_dir = '../../plots/model'
    if not os.path.exists('../../data/processed'):  # ajustando la ruta para cuando se corre local
        plots_dir = 'plots/model'
        
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'descriptores_histogram.png'), dpi=150)
    plt.close()
    print(f"Gráfica guardada en {plots_dir}/descriptores_histogram.png")
    # ---------------------------------------------------------

    # Guardar CSVs
    proc_dir = '../../data/processed'
    if not os.path.exists('../../data/raw'):
        proc_dir = 'data/processed'
    os.makedirs(proc_dir, exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join(proc_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(proc_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(proc_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(proc_dir, 'y_test.csv'), index=False)
    
    print("Datos preprocesados guardados correctamente.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess()