# E-Commerce Popularity Prediction & Optimization

Este proyecto implementa una solución de aprendizaje automático para predecir la popularidad de productos en un conjunto de datos de e-commerce, optimizando los hiperparámetros de una Red Neuronal (MLP) mediante algoritmos metaheurísticos (Algoritmo Genético y Optimización por Enjambre de Lobos Grises).

## Estructura del Proyecto

La organización del repositorio es la siguiente:

```text
.
├── data/
│   ├── raw/                # Datos originales sin procesar
│   └── processed/          # Datos limpios y listos para entrenamiento
├── src/
│   ├── preprocessing/      # Scripts de limpieza y transformación de datos
│   ├── optimization/       # Optimización de hiperparámetros (GA-GWO MLP)
│   └── algorithms/         # Implementaciones de referencia de metaheurísticas (ACO, GA, GWO)
├── results/
│   ├── history/            # Historial de métricas por generación
│   └── reports/            # Reportes finales de desempeño
├── .gitignore              # Archivos ignorados por Git
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Descripción del proyecto
```

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/GalekTato/e-commerce.git
   cd e-commerce
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Preprocesamiento**: Para limpiar los datos y generar las clases de popularidad:
   ```bash
   cd src/preprocessing
   python data_cleaning.py
   ```

2. **Optimización**: Para ejecutar la optimización de la red neuronal:
   ```bash
   cd src/optimization
   python ga_gwo_mlp.py
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
[Gael Askary Razo Montañez](https://github.com/GalekTato)
[Alejandro Hernández Hernández](https://github.com/coloronthewalls)
[Isaac Aguilar Durán](https://github.com/isaacwide)
[José Ángel Cambranis Sánchez](https://github.com/Jose-Angel-Sanchez)
