# Ajuste de Hiperparámetros de una RNA mediante GA+GWO

Este proyecto implementa una red neuronal artificial (Perceptrón Multicapa - `MLPClassifier`) optimizada mediante una metaheurística híbrida que combina un **Algoritmo Genético (GA)** y el **Grey Wolf Optimizer (GWO)**. 

Se utiliza el dataset [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) para predecir si un objeto de interés espacial es un exoplaneta confirmado, un candidato o un falso positivo.

## Características Técnicas

*   **Preprocesamiento Robusto:** Eliminación de ruido, imputación por mediana, y *target encoding* para predecir la disposición de exoplanetas de la NASA sin *target leakage*.
*   **Paralelismo por CPU (Multiprocessing):** El cálculo de aptitud computacionalmente costoso ($K$-Fold Cross Validation) está distribuido entre todos los núcleos disponibles de la CPU (`multiprocessing.Pool`), reduciendo el tiempo de convergencia de horas a escasos minutos.
*   **Optimizador Híbrido Cooperativo:** El Algoritmo Genético provee exploración global y diversidad, mientras que la técnica matemática GWO ejerce un refinamiento exhaustivo local en las regiones de élite.
*   **Visualización Científica Automática:** Un generador en Python mapea la evolución, convergencia y topología final de la red.

## Estructura del Directorio

```text
TIA Tercer Parcial/
├── data/
│   ├── raw/                 # Colocar aquí cumulative.csv
│   └── processed/           # Matrices procesadas generadas automáticamente
├── plots/                   # Exportación de 14 gráficos de desempeño
├── reporte/                 # Código fuente y PDF de la documentación LaTeX
├── results/                 # Historiales y bitácoras CSV/JSON
└── src/
    ├── algorithms/          # Lógica metaheurística matemática (GA, GWO)
    ├── optimization/        # Ejecución principal y Pipeline
    └── preprocessing/       # Limpieza y preparación de datos
```

## Instalación y Ejecución

1.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Limpiar datos e imputar:**
    ```bash
    python src/preprocessing/data_cleaning.py
    ```

3.  **Ejecutar Optimización de Hiperparámetros:**
    *(Nota: Este script utilizará el 100% de la CPU para paralelizar la evaluación).*
    ```bash
    python src/optimization/ga_gwo_mlp.py
    ```

4.  **Generar Reportes Analíticos (14 diagramas PNG):**
    ```bash
    python src/optimization/plot_generator.py
    ```

## Algoritmos Implementados

- **MLP (Perceptrón Multicapa)**: Clasificador base.
- **Algoritmo Genético (GA)**: Utilizado para la evolución de la población de hiperparámetros.
- **Grey Wolf Optimizer (GWO)**: Utilizado para refinar las mejores soluciones encontradas por el GA.
- **Implementaciones Adicionales**:
  - Optimización por Colonias de Hormigas (ACO) para el problema del viajero.
  - GA básico para evolución de cadenas (`genetic_algorithm.py`).
  - GWO básico para funciones matemáticas (`gwo_math.py`).

## Autores
[Gael Askary Razo Montañez](https://github.com/GalekTato) |
[Alejandro Hernández Hernández](https://github.com/coloronthewalls) |
[Isaac Aguilar Durán](https://github.com/isaacwide) |
[José Ángel Cambranis Sánchez](https://github.com/Jose-Angel-Sanchez)
