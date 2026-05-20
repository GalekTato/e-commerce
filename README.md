# Kepler-RNA

Proyecto de ajuste de hiperparámetros para una red neuronal artificial de clasificación multiclase sobre el dataset NASA Kepler Exoplanet Search Results. La solución usa un Perceptrón Multicapa (`MLPClassifier` de `scikit-learn`) optimizado con una metaheurística híbrida basada en **Algoritmo Genético (GA)** + **Grey Wolf Optimizer (GWO)**.

El objetivo es clasificar cada objeto de interés de Kepler como:

* `CONFIRMED`: exoplaneta confirmado.
* `CANDIDATE`: candidato todavía no resuelto.
* `FALSE POSITIVE`: falso positivo astronómico.

## Problema Que Resuelve

El espacio de hiperparámetros de una RNA es no convexo, mixto y propenso a mínimos locales. En lugar de usar una búsqueda exhaustiva o manual, el proyecto propone una estrategia híbrida:

* El **GA** explora el espacio global de soluciones y conserva diversidad.
* El **GWO** refina localmente las mejores soluciones del GA.
* El **MLPClassifier** evalúa cada configuración con validación cruzada estratificada.

Esta combinación permite encontrar arquitecturas y parámetros que generalizan bien sin requerir una búsqueda manual costosa.

## Dataset

Dataset usado: [NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

### Origen

El conjunto proviene de observaciones fotométricas de la misión Kepler de la NASA. Su propósito original es identificar objetos de interés astronómico y distinguir entre planetas confirmados, candidatos y falsos positivos.

### Qué contiene

El dataset combina variables físicas y astronómicas como:

* período orbital (`koi_period`)
* duración del tránsito (`koi_duration`)
* profundidad del tránsito (`koi_depth`)
* radio planetario estimado (`koi_prad`)
* temperatura de equilibrio (`koi_teq`)
* insolación (`koi_insol`)
* parámetros estelares (`koi_steff`, `koi_slogg`, `koi_srad`)

### Preprocesamiento

El flujo de preprocesamiento realiza lo siguiente:

* elimina columnas identificadoras o no útiles para el aprendizaje,
* descarta variables con demasiados valores faltantes,
* imputa faltantes numéricos con la mediana,
* divide train/test de forma aleatoria y estratificada,
* estandariza con `StandardScaler` dentro de una `Pipeline`.

El split principal es 80/20 y se cuida que la proporción de clases quede balanceada en ambos subconjuntos.

## Propuesta Metodológica

La propuesta del proyecto se basa en tres piezas:

1. `GA` como explorador global.
2. `GWO` como refinador local sobre la élite.
3. `MLPClassifier` como modelo evaluado en cada individuo.

La codificación de cada individuo representa hiperparámetros del MLP:

* número de capas ocultas y neuronas,
* función de activación,
* tasa de aprendizaje inicial,
* regularización L2 (`alpha`),
* número máximo de iteraciones.

La evaluación usa validación cruzada estratificada de 5 folds y el fitness es `f1_weighted`.

## Resultados Resumidos

En la ejecución actual del proyecto, el modelo final alcanzó aproximadamente:

* `accuracy` de entrenamiento: 0.8998
* `accuracy` de prueba: 0.8981
* `F1 weighted` de prueba: 0.8985

La mejor configuración encontrada fue una red compacta de una capa oculta con 8 neuronas y activación `tanh`.

## Estructura Del Proyecto

```text
Kepler-RNA/
├── data/
│   ├── raw/                # cumulative.csv original
│   └── processed/          # X_train.csv, X_test.csv, y_train.csv, y_test.csv
├── plots/
│   ├── ga/                 # Gráficas del GA
│   ├── gwo/                # Gráficas del GWO
│   └── model/              # Gráficas del preprocesamiento y evaluación final
├── reporte/                # Documento LaTeX del reporte
├── results/
│   ├── history/            # Historiales de evolución CSV
│   └── reports/            # JSON y reportes finales
└── src/
    ├── algorithms/         # Implementaciones auxiliares (GA, ACO, GWO matemático)
    ├── optimization/       # Entrenamiento híbrido y generación de gráficas
    └── preprocessing/      # Limpieza y partición de datos
```

## Requisitos

* Python 3.14 o compatible con los paquetes del proyecto.
* `pip`.
* Dependencias instaladas desde `requirements.txt`.

## Entorno Virtual

El proyecto funciona con un entorno virtual local. Puedes usar `.venv` del repositorio o crear uno nuevo.

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Comandos De Ejecución

Ejecuta siempre desde la raíz del proyecto `Kepler-RNA/`.

### 1. Colocar el dataset

Ubica `cumulative.csv` en:

```text
data/raw/cumulative.csv
```

### 2. Preprocesar datos

```bash
python src/preprocessing/data_cleaning.py
```

Este paso genera:

* `data/processed/X_train.csv`
* `data/processed/X_test.csv`
* `data/processed/y_train.csv`
* `data/processed/y_test.csv`
* `plots/model/descriptores_histogram.png`

### 3. Ejecutar la optimización híbrida

```bash
python src/optimization/ga_gwo_mlp.py
```

Este script:

* toma una muestra estratificada del `train` para la búsqueda,
* evalúa individuos con `GA + GWO`,
* guarda historiales en `results/history/`,
* guarda métricas finales en `results/reports/final_metrics.json`,
* entrena el MLP final sobre el entrenamiento completo,
* evalúa la solución final sobre el `test`.

### 4. Generar gráficas y análisis

```bash
python src/optimization/plot_generator.py
```

Este script genera las gráficas en `plots/ga`, `plots/gwo` y `plots/model`.

## Gráficas Generadas

El generador produce 15 gráficas principales, incluyendo:

* evolución del fitness del GA,
* boxplot de la población por generación,
* tamaño de la población por generación,
* diversidad genética,
* heatmap de hiperparámetros,
* convergencia del GWO,
* dispersión posicional de la manada,
* tabla estadística final del GWO,
* distribución de clases train/test,
* matriz de confusión,
* reporte de clasificación,
* validación cruzada,
* arquitectura final del MLP,
* diagrama del perceptrón,
* línea de tiempo híbrida GA + GWO.

## Archivos Importantes

* `src/preprocessing/data_cleaning.py`: limpieza, imputación, partición estratificada y guardado de CSV procesados.
* `src/optimization/ga_gwo_mlp.py`: búsqueda híbrida de hiperparámetros y entrenamiento final.
* `src/optimization/plot_generator.py`: generación de todas las gráficas y resúmenes.
* `results/history/*.csv`: bitácoras de evolución y población.
* `results/reports/final_metrics.json`: métricas finales del mejor modelo.

## Criterios Cubiertos Del Trabajo

### Punto 2

Se define y justifica una propuesta híbrida GA + GWO para ajustar los hiperparámetros de una RNA.

### Punto 3

Se implementa en Python con `scikit-learn` y `MLPClassifier`, usando:

* split aleatorio y estratificado,
* control de desbalance entre train y test,
* `Pipeline` con `StandardScaler`,
* validación cruzada estratificada,
* evaluación final sobre test.

### Punto 4

Se documenta estadísticamente la evolución con:

* curvas de fitness por generación,
* boxplots de la población,
* diversidad genética,
* tamaño de la población por generación,
* convergencia del GWO,
* métricas finales y matriz de confusión.

## Notas De Ejecución

* El script de optimización usa todos los núcleos de CPU disponibles para acelerar la evaluación.
* Si ejecutas en WSL o Linux, asegúrate de que el dataset esté en `data/raw/cumulative.csv`.
* Si ya existen los CSV procesados y resultados, puedes volver a ejecutar solo `plot_generator.py` para regenerar las figuras.

## Autores

* Gael Askary Razo Montañez
* Alejandro Hernández Hernández
* Isaac Aguilar Durán
* José Ángel Cambranis Sánchez

## Github's

* [Gael Askary Razo Montañez](https://github.com/GalekTato)
* [Alejandro Hernández Hernández](https://github.com/coloronthewalls)
* [Isaac Aguilar Durán](https://github.com/isaacwide)
* [José Ángel Cambranis Sánchez](https://github.com/Jose-Angel-Sanchez)

