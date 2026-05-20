import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import time
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / 'src' / 'preprocessing'
DATA_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_HISTORY_DIR = PROJECT_ROOT / 'results' / 'history'
RESULTS_REPORTS_DIR = PROJECT_ROOT / 'results' / 'reports'

# --- CONFIGURACIÓN DE HIPERPARÁMETROS DEL OPTIMIZADOR ---
N_JOBS = cpu_count()
N: int = 12              # Tamaño de la población del Genético (acelerado)
G: int = 12              # Número máximo de generaciones (acelerado)
K: int = 3               # Frecuencia de refinamiento GWO (cada K generaciones)
WOLF_COUNT: int = 4      # Cantidad de lobos élite para GWO
TOURNAMENT_SIZE: int = 3 # Tamaño del torneo para selección de GA
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.15
ELITISM_COUNT: int = 2
PATIENCE: int = 3        # Paciencia para early stopping
MIN_DELTA: float = 1e-4
SEED: int = 42
MAX_GWO_ITER: int = 25   # FIX 1: Aumentado a 25 para ver convergencia

# --- ESPACIO DE BÚSQUEDA DEL MLP ---
HIDDEN_LAYERS: List[Tuple[int, ...]] = [
    (8,), (16,), (32,), (64,),
    (16, 8), (16, 16), (32, 16), (32, 32), (64, 32)
]
ACTIVATIONS: List[str] = ['relu', 'tanh', 'logistic']
MAX_ITERS: List[int] = [500, 1000, 1500]

def normalize_params(raw_values: List[Any]) -> List[float]:
    hl_idx: int = HIDDEN_LAYERS.index(raw_values[0])
    act_idx: int = ACTIVATIONS.index(raw_values[1])
    lr: float = float(raw_values[2])
    alpha: float = float(raw_values[3])
    iter_idx: int = MAX_ITERS.index(raw_values[4])

    n_hl: float = hl_idx / max(1, len(HIDDEN_LAYERS) - 1)
    n_act: float = act_idx / max(1, len(ACTIVATIONS) - 1)
    n_lr: float = (np.log10(lr) - (-4)) / (-1 - (-4))
    n_alpha: float = (np.log10(alpha) - (-3)) / (3 - (-3))
    n_iter: float = iter_idx / max(1, len(MAX_ITERS) - 1)

    return [n_hl, n_act, n_lr, n_alpha, n_iter]

def denormalize_params(norm_values: List[float]) -> List[Any]:
    v: List[float] = [max(0.0, min(1.0, val)) for val in norm_values]
    
    hl_idx: int = int(round(v[0] * (len(HIDDEN_LAYERS) - 1)))
    act_idx: int = int(round(v[1] * (len(ACTIVATIONS) - 1)))
    lr: float = 10 ** (v[2] * 3 - 4)
    alpha: float = 10 ** (v[3] * 6 - 3)
    iter_idx: int = int(round(v[4] * (len(MAX_ITERS) - 1)))

    return [
        HIDDEN_LAYERS[hl_idx],
        ACTIVATIONS[act_idx],
        lr,
        alpha,
        MAX_ITERS[iter_idx]
    ]

def encode_individual() -> List[float]:
    return np.random.uniform(0, 1, 5).tolist()

def decode_individual(ind: List[float]) -> Dict[str, Any]:
    denorm: List[Any] = denormalize_params(ind)
    return {
        'hidden_layer_sizes': denorm[0],
        'activation': denorm[1],
        'learning_rate_init': denorm[2],
        'alpha': denorm[3],
        'max_iter': denorm[4]
    }

def evaluate_fitness_worker(individual: List[float], X_search: pd.DataFrame, y_search: pd.Series) -> Tuple[float, Dict[str, Any]]:
    params: Dict[str, Any] = decode_individual(individual)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**params, early_stopping=True, random_state=SEED))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # n_jobs=1 en CV, porque paralelizamos la población a nivel superior
    scores: np.ndarray = cross_val_score(pipe, X_search, y_search, cv=cv, scoring='f1_weighted', n_jobs=1)
    f1 = float(np.mean(scores))
    
    log_entry = {
        'hidden_layer_sizes': str(params['hidden_layer_sizes']),
        'activation': params['activation'],
        'learning_rate_init': params['learning_rate_init'],
        'alpha': params['alpha'],
        'max_iter': params['max_iter'],
        'fitness': f1
    }
    return f1, log_entry

def evaluar_poblacion(poblacion: List[List[float]], X_search: pd.DataFrame, y_search: pd.Series) -> List[Tuple[float, Dict[str, Any]]]:
    # Multiprocessing refactor as requested (Part 1)
    args = [(ind, X_search, y_search) for ind in poblacion]
    with Pool(processes=N_JOBS) as pool:
        resultados = pool.starmap(evaluate_fitness_worker, args)
    return resultados

def tournament_selection(population: List[List[float]], fitnesses: List[float], k: int) -> List[float]:
    indices: np.ndarray = np.random.choice(len(population), k, replace=False)
    best_idx: int = int(indices[np.argmax([fitnesses[i] for i in indices])])
    return population[best_idx].copy()

def uniform_crossover(p1: List[float], p2: List[float], rate: float) -> Tuple[List[float], List[float]]:
    c1: List[float] = p1.copy()
    c2: List[float] = p2.copy()
    if np.random.rand() < rate:
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate(individual: List[float], rate: float) -> List[float]:
    mutated: List[float] = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < rate:
            if i in [0, 1, 4]:
                mutated[i] = float(np.random.uniform(0, 1))
            else:
                mutated[i] += float(np.random.normal(0, 0.1))
                mutated[i] = max(0.0, min(1.0, mutated[i]))
    return mutated

def get_spread(wolves: List[List[float]]) -> float:
    # Calcula la distancia promedio entre los lobos para medir divergencia/convergencia
    if len(wolves) <= 1: return 0.0
    dists = []
    for i in range(len(wolves)):
        for j in range(i+1, len(wolves)):
            dists.append(np.linalg.norm(np.array(wolves[i]) - np.array(wolves[j])))
    return float(np.mean(dists))

def gwo_iterative_refine(
    wolves: List[List[float]], 
    X_search: pd.DataFrame, 
    y_search: pd.Series, 
    g: int,
    max_gwo_iter: int = MAX_GWO_ITER
) -> Tuple[List[List[float]], List[Dict[str, float]], List[Dict[str, Any]]]:
    
    w_count = len(wolves)
    dim = len(wolves[0])

    # FIX 1 (Option B): Add artificial noise/perturbation to initial wolf positions 
    # so they start spread around the GA best solution, showing visible convergence
    for i in range(1, w_count): # Mantenemos el primero (alfa original) intacto
        wolves[i] = [min(1.0, max(0.0, val + np.random.normal(0, 0.15))) for val in wolves[i]]

    # Evaluación paralela inicial de los lobos usando multiprocessing
    t_start = time.perf_counter()
    results = evaluar_poblacion(wolves, X_search, y_search)
    fitnesses = [r[0] for r in results]
    eval_log = [r[1] for r in results]
    print(f"    GWO Evaluación inicial - T. Elapsed: {time.perf_counter() - t_start:.2f}s")
        
    gwo_history: List[Dict[str, float]] = []
    
    for t in range(max_gwo_iter):
        t_iter_start = time.perf_counter()
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        alpha_pos = wolves[sorted_indices[0]].copy()
        alpha_fit = fitnesses[sorted_indices[0]]
        
        beta_pos = wolves[sorted_indices[1]].copy() if w_count > 1 else alpha_pos
        beta_fit = fitnesses[sorted_indices[1]] if w_count > 1 else alpha_fit
        
        delta_pos = wolves[sorted_indices[2]].copy() if w_count > 2 else beta_pos
        delta_fit = fitnesses[sorted_indices[2]] if w_count > 2 else beta_fit
        
        current_spread = get_spread(wolves)
        
        gwo_history.append({
            'generation': float(g),
            'gwo_iteration': float(t + 1),
            'alpha_fitness': alpha_fit,
            'beta_fitness': beta_fit,
            'delta_fitness': delta_fit,
            'spread': current_spread
        })
        
        a = 2.0 * (1.0 - (t / max_gwo_iter))
        
        new_wolves: List[List[float]] = []
        for i in range(w_count):
            if i == sorted_indices[0]: # Siempre conservar estricto al Alfa
                new_wolves.append(wolves[i].copy())
                continue

            pos = wolves[i]
            X_new = []
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - pos[j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - pos[j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - pos[j])
                X3 = delta_pos[j] - A3 * D_delta

                new_val = (X1 + X2 + X3) / 3.0
                new_val = max(0.0, min(1.0, new_val))
                X_new.append(new_val)
            new_wolves.append(X_new)

        wolves = new_wolves
        
        # Evaluación paralela iterativa de los lobos usando multiprocessing
        results = evaluar_poblacion(wolves, X_search, y_search)
        fitnesses = [r[0] for r in results]
        eval_log.extend([r[1] for r in results])
        print(f"    GWO Iter {t+1}/{max_gwo_iter} - Mejor local: {np.max(fitnesses):.4f} - T. Elapsed: {time.perf_counter() - t_iter_start:.2f}s")
        
    return wolves, gwo_history, eval_log

def run_ga_gwo(X_search: pd.DataFrame, y_search: pd.Series, seed: int):
    np.random.seed(seed)
    print(f"Usando {N_JOBS} núcleos de CPU para evaluación paralela", flush=True)
    
    population: List[List[float]] = [encode_individual() for _ in range(N)]
    
    ga_history: List[Dict[str, float]] = []
    ga_population_history: List[Dict[str, float]] = []
    all_gwo_histories: List[Dict[str, float]] = []
    global_eval_log: List[Dict[str, Any]] = []

    best_fitness_overall: float = -1.0
    best_individual: List[float] = []
    epochs_without_improvement: int = 0
    best_gen: int = 1
    early_stopped: bool = False

    for g in range(G):
        print(f"--- Generación Genética {g+1}/{G} ---", flush=True)
        t_gen_start = time.perf_counter()
        
        # Evaluación paralela masiva por CPU de la población
        results = evaluar_poblacion(population, X_search, y_search)
        fitnesses = [r[0] for r in results]
        global_eval_log.extend([r[1] for r in results])
        
        # Guardar historial detallado de la población para boxplot
        for i, fit in enumerate(fitnesses):
            ga_population_history.append({'generation': float(g+1), 'individual': i, 'fitness': fit})

        gen_best_idx: int = int(np.argmax(fitnesses))
        gen_best_fit: float = float(fitnesses[gen_best_idx])
        worst_fit: float = float(np.min(fitnesses))
        mean_fit: float = float(np.mean(fitnesses))
        std_fit: float = float(np.std(fitnesses))

        ga_history.append({
            'generation': float(g + 1),
            'best_fitness': gen_best_fit,
            'mean_fitness': mean_fit,
            'worst_fitness': worst_fit,
            'std_fitness': std_fit
        })

        if gen_best_fit - best_fitness_overall > MIN_DELTA:
            best_fitness_overall = gen_best_fit
            best_individual = population[gen_best_idx].copy()
            epochs_without_improvement = 0
            best_gen = g + 1
            print(f"  (*) ¡Mejora! Nuevo Mejor Fitness (CV): {best_fitness_overall:.5f}")
        else:
            epochs_without_improvement += 1

        print(f"  Tiempo de generación: {time.perf_counter() - t_gen_start:.2f}s")

        if epochs_without_improvement >= PATIENCE:
            print(f"  (!) Early stopping activado por falta de mejora en {PATIENCE} generaciones.", flush=True)     
            early_stopped = True
            break

        sorted_indices = np.argsort(fitnesses)[::-1]

        # --- ACOPLAMIENTO HÍBRIDO GWO ---
        if (g + 1) % K == 0:
            print(f"  --> Aplicando refinamiento GWO local sobre los top {WOLF_COUNT} lobos...", flush=True)        
            top_w_indices = sorted_indices[:WOLF_COUNT].tolist()
            wolves = [population[i].copy() for i in top_w_indices]

            refined_wolves, gwo_history, eval_log = gwo_iterative_refine(wolves, X_search, y_search, g + 1, max_gwo_iter=MAX_GWO_ITER)   
            all_gwo_histories.extend(gwo_history)
            global_eval_log.extend(eval_log)

            for i, idx in enumerate(top_w_indices):
                population[idx] = refined_wolves[i]

            # Evaluación paralela después de la fase de refinamiento GWO
            t_post_gwo = time.perf_counter()
            results = evaluar_poblacion(population, X_search, y_search)
            fitnesses = [r[0] for r in results]
            global_eval_log.extend([r[1] for r in results])
            sorted_indices = np.argsort(fitnesses)[::-1]
            print(f"  --> GWO re-evaluación finalizada en {time.perf_counter() - t_post_gwo:.2f}s")

        # --- OPERADORES GENÉTICOS ---
        new_population: List[List[float]] = [population[i].copy() for i in sorted_indices[:ELITISM_COUNT]]

        while len(new_population) < N:
            p1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            p2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            c1, c2 = uniform_crossover(p1, p2, CROSSOVER_RATE)
            new_population.append(mutate(c1, MUTATION_RATE))
            if len(new_population) < N:
                new_population.append(mutate(c2, MUTATION_RATE))

        population = new_population

    return (
        decode_individual(best_individual), 
        pd.DataFrame(ga_history), 
        pd.DataFrame(all_gwo_histories), 
        best_gen, 
        early_stopped,
        pd.DataFrame(ga_population_history),
        pd.DataFrame(global_eval_log)
    )

def main() -> None:
    print("Cargando y procesando NASA Kepler Exoplanet Search Results...", flush=True)
    sys.path.append(str(PREPROCESSING_DIR))
    try:
        from data_cleaning import load_and_preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess()
    except ImportError:
        print("Error: No se pudo importar load_and_preprocess de data_cleaning.py")
        return
    except Exception as e:
        print(f"Error al cargar/preprocesar: {e}")
        # Si falla cargará de CSV local
        train_x_path = DATA_PROCESSED_DIR / 'X_train.csv'
        if not train_x_path.exists():
            train_x_path = PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'
        X_train = pd.read_csv(train_x_path)
        X_test = pd.read_csv(train_x_path.with_name('X_test.csv'))
        y_train = pd.read_csv(train_x_path.with_name('y_train.csv')).squeeze()
        y_test = pd.read_csv(train_x_path.with_name('y_test.csv')).squeeze()

    print(f"Creando subconjunto estratificado del 25% para optimización de hiperparámetros...", flush=True)
    _, X_search, _, y_search = train_test_split(
        X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    print(f"Búsqueda con: {X_search.shape} | Entrenamiento final con: {X_train.shape}")
    print("Iniciando Optimización Híbrida GA-GWO (Acelerada por Multiprocessing)...", flush=True)

    best_params, ga_history_df, gwo_history_df, best_gen, early_stopped, ga_pop_history, evaluated_df = run_ga_gwo(X_search, y_search, SEED)      

    # Directorio de resultados
    base_res_dir = PROJECT_ROOT / 'results'
    RESULTS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ga_history_df.to_csv(RESULTS_HISTORY_DIR / 'ga_gwo_history.csv', index=False)
    ga_pop_history.to_csv(RESULTS_HISTORY_DIR / 'ga_population_history.csv', index=False)
    evaluated_df.to_csv(RESULTS_HISTORY_DIR / 'all_evaluated_individuals.csv', index=False)

    if len(gwo_history_df) > 0:
        gwo_history_df.to_csv(RESULTS_HISTORY_DIR / 'gwo_wolves_history.csv', index=False)
        
    # --- ENTRENAMIENTO FINAL SOBRE DATASET COMPLETO ---
    print(f"\nEntrenando clasificador MLP final con los mejores parámetros sobre el dataset completo ({len(X_train)} muestras)...", flush=True)
    final_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(**best_params, early_stopping=True, random_state=SEED))
    ])
    final_pipe.fit(X_train, y_train)
    
    # Calcular y guardar Cross Validation en el dataset test (Para plot de k-folds)
    cv_scores = cross_val_score(final_pipe, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=N_JOBS)
    
    train_acc = float(final_pipe.score(X_train, y_train))
    test_acc = float(final_pipe.score(X_test, y_test))
    y_pred = final_pipe.predict(X_test)
    test_f1 = float(f1_score(y_test, y_pred, average='weighted'))

    cm = confusion_matrix(y_test, y_pred)
    class_report_str = classification_report(y_test, y_pred, target_names=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
    class_report_dict = classification_report(y_test, y_pred, target_names=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], output_dict=True)

    test_metrics = {
        'best_params': best_params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'cv_scores': cv_scores.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report_dict': class_report_dict,
        'best_gen': best_gen,
        'early_stopped': early_stopped
    }

    import json
    with open(RESULTS_REPORTS_DIR / 'final_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    print("\n========================================")
    print("GA-GWO HYPERPARAMETER OPTIMIZATION SUMMARY (MULTIPROCESSING)")
    print("========================================")
    print(f"Best generation    : {best_gen}")
    print(f"Best hyperparameters found:")
    print(f"  hidden_layer_sizes : {best_params['hidden_layer_sizes']}")
    print(f"  activation         : {best_params['activation']}")
    print(f"  learning_rate_init : {best_params['learning_rate_init']:.6f}")
    print(f"  alpha              : {best_params['alpha']:.6f}")
    print(f"  max_iter           : {best_params['max_iter']}")
    print("----------------------------------------")
    print("Final MLP on test set:")
    print(f"  Accuracy           : {test_acc:.4f}")
    print(f"  F1-weighted        : {test_f1:.4f}")
    print("========================================")

if __name__ == '__main__':
    main()
