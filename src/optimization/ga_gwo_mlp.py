import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report

N: int = 20
G: int = 30
K: int = 3
WOLF_COUNT: int = 5
TOURNAMENT_SIZE: int = 3
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.15
ELITISM_COUNT: int = 2
PATIENCE: int = 7
MIN_DELTA: float = 1e-4
SEED: int = 42

HIDDEN_LAYERS: List[Tuple[int, ...]] = [
    (32,), (64,), (128,), (256,), (512,), 
    (64, 64), (128, 64), (128, 128), (256, 128), (256, 256), 
    (128, 64, 32), (256, 128, 64)
]
ACTIVATIONS: List[str] = ['relu', 'tanh', 'logistic']
MAX_ITERS: List[int] = [200, 500, 1000]

def normalize_params(raw_values: List[Any]) -> List[float]:
    hl_idx: int = HIDDEN_LAYERS.index(raw_values[0])
    act_idx: int = ACTIVATIONS.index(raw_values[1])
    lr: float = float(raw_values[2])
    alpha: float = float(raw_values[3])
    iter_idx: int = MAX_ITERS.index(raw_values[4])
    
    n_hl: float = hl_idx / max(1, len(HIDDEN_LAYERS) - 1)
    n_act: float = act_idx / max(1, len(ACTIVATIONS) - 1)
    n_lr: float = (np.log10(lr) - (-4)) / (-1 - (-4))
    n_alpha: float = (np.log10(alpha) - (-5)) / (-1 - (-5))
    n_iter: float = iter_idx / max(1, len(MAX_ITERS) - 1)
    
    return [n_hl, n_act, n_lr, n_alpha, n_iter]

def denormalize_params(norm_values: List[float]) -> List[Any]:
    v: List[float] = [max(0.0, min(1.0, val)) for val in norm_values]
    
    hl_idx: int = int(round(v[0] * (len(HIDDEN_LAYERS) - 1)))
    act_idx: int = int(round(v[1] * (len(ACTIVATIONS) - 1)))
    lr: float = 10 ** (v[2] * 3 - 4)
    alpha: float = 10 ** (v[3] * 4 - 5)
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

def evaluate_fitness(individual: List[float], X: pd.DataFrame, y: pd.Series) -> float:
    params: Dict[str, Any] = decode_individual(individual)
    mlp: MLPClassifier = MLPClassifier(**params, random_state=SEED)
    scores: np.ndarray = cross_val_score(mlp, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
    return float(np.mean(scores))

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

def gwo_refine(wolves: List[List[float]], fitnesses: List[float], a: float) -> List[List[float]]:
    sorted_indices: np.ndarray = np.argsort(fitnesses)[::-1]
    alpha_pos: List[float] = wolves[sorted_indices[0]]
    beta_pos: List[float] = wolves[sorted_indices[1]] if len(wolves) > 1 else alpha_pos
    delta_pos: List[float] = wolves[sorted_indices[2]] if len(wolves) > 2 else beta_pos
    
    new_wolves: List[List[float]] = []
    for i in range(len(wolves)):
        if i < 3:
            new_wolves.append(wolves[sorted_indices[i]].copy())
            continue
            
        pos: List[float] = wolves[i]
        X_new: List[float] = []
        for j in range(len(pos)):
            r1: float = float(np.random.rand())
            r2: float = float(np.random.rand())
            A1: float = 2 * a * r1 - a
            C1: float = 2 * r2
            D_alpha: float = abs(C1 * alpha_pos[j] - pos[j])
            X1: float = alpha_pos[j] - A1 * D_alpha
            
            r1 = float(np.random.rand())
            r2 = float(np.random.rand())
            A2: float = 2 * a * r1 - a
            C2: float = 2 * r2
            D_beta: float = abs(C2 * beta_pos[j] - pos[j])
            X2: float = beta_pos[j] - A2 * D_beta
            
            r1 = float(np.random.rand())
            r2 = float(np.random.rand())
            A3: float = 2 * a * r1 - a
            C3: float = 2 * r2
            D_delta: float = abs(C3 * delta_pos[j] - pos[j])
            X3: float = delta_pos[j] - A3 * D_delta
            
            new_val: float = (X1 + X2 + X3) / 3.0
            new_val = max(0.0, min(1.0, new_val))
            X_new.append(new_val)
        new_wolves.append(X_new)
    return new_wolves

def run_ga_gwo(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame, int, bool]:
    np.random.seed(seed)
    population: List[List[float]] = [encode_individual() for _ in range(N)]
    history: List[Dict[str, float]] = []
    
    best_fitness_overall: float = -1.0
    best_individual: List[float] = []
    epochs_without_improvement: int = 0
    best_gen: int = 1
    early_stopped: bool = False
    
    for g in range(G):
        print(f"--- Generation {g+1}/{G} ---", flush=True)
        fitnesses: List[float] = [evaluate_fitness(ind, X_train, y_train) for ind in population]
        
        gen_best_idx: int = int(np.argmax(fitnesses))
        gen_best_fit: float = float(fitnesses[gen_best_idx])
        mean_fit: float = float(np.mean(fitnesses))
        std_fit: float = float(np.std(fitnesses))
        
        history.append({
            'generation': float(g + 1),
            'best_fitness': gen_best_fit,
            'mean_fitness': mean_fit,
            'std_fitness': std_fit
        })
        
        if gen_best_fit - best_fitness_overall > MIN_DELTA:
            best_fitness_overall = gen_best_fit
            best_individual = population[gen_best_idx].copy()
            epochs_without_improvement = 0
            best_gen = g + 1
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= PATIENCE:
            early_stopped = True
            break
            
        sorted_indices: np.ndarray = np.argsort(fitnesses)[::-1]
        
        if (g + 1) % K == 0:
            top_w_indices: List[int] = sorted_indices[:WOLF_COUNT].tolist()
            wolves: List[List[float]] = [population[i] for i in top_w_indices]
            wolf_fitnesses: List[float] = [fitnesses[i] for i in top_w_indices]
            a: float = 2.0 - (2.0 * g / G)
            refined_wolves: List[List[float]] = gwo_refine(wolves, wolf_fitnesses, a)
            for i, idx in enumerate(top_w_indices):
                population[idx] = refined_wolves[i]
            
            fitnesses = [evaluate_fitness(ind, X_train, y_train) for ind in population]
            sorted_indices = np.argsort(fitnesses)[::-1]
            
        new_population: List[List[float]] = [population[i].copy() for i in sorted_indices[:ELITISM_COUNT]]
        
        while len(new_population) < N:
            p1: List[float] = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            p2: List[float] = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            c1, c2 = uniform_crossover(p1, p2, CROSSOVER_RATE)
            new_population.append(mutate(c1, MUTATION_RATE))
            if len(new_population) < N:
                new_population.append(mutate(c2, MUTATION_RATE))
                
        population = new_population
        
    return decode_individual(best_individual), pd.DataFrame(history), best_gen, early_stopped

def main() -> None:
    dataset_path: str = '../../data/processed/dataset_listo_para_rna.csv'
    
    if not os.path.exists(dataset_path):
        # Intentar ruta alternativa si se ejecuta desde la raíz
        dataset_path = 'data/processed/dataset_listo_para_rna.csv'
        if not os.path.exists(dataset_path):
            print(f"No se encontró el dataset en: {dataset_path}")
            return

    df: pd.DataFrame = pd.read_csv(dataset_path)
    if 'Supplier ID' in df.columns:
        df = df.drop(columns=['Supplier ID'])
    X: pd.DataFrame = df.drop(columns=['Target_Clase'])
    y: pd.Series = df['Target_Clase']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    
    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()
    
    train_dist_str = ", ".join([f"{k}: {v}" for k, v in train_dist.items()])
    test_dist_str = ", ".join([f"{k}: {v}" for k, v in test_dist.items()])
    
    best_params, history_df, best_gen, early_stopped = run_ga_gwo(X_train, y_train, SEED)
    history_df.to_csv('../../results/history/ga_gwo_history.csv', index=False)
    
    final_mlp: MLPClassifier = MLPClassifier(**best_params, random_state=SEED)
    final_mlp.fit(X_train, y_train)
    
    train_acc: float = float(final_mlp.score(X_train, y_train))
    test_acc: float = float(final_mlp.score(X_test, y_test))
    y_pred: np.ndarray = final_mlp.predict(X_test)
    test_f1: float = float(f1_score(y_test, y_pred, average='weighted'))
    
    class_report_str = classification_report(y_test, y_pred)
    
    # Calculate per-class f1 for summary
    f1_per_class = f1_score(y_test, y_pred, average=None).tolist()
    per_class_f1_str = ", ".join([f"{c}: {score:.4f}" for c, score in zip(sorted(list(set(y_test))), f1_per_class)])
    
    best_fitness_cv = history_df['best_fitness'].max()
    early_stopped_str = 'yes' if early_stopped else 'no'
    
    print("\n========================================")
    print("GA-GWO HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("========================================")
    print(f"Dataset            : dataset_listo_para_rna.csv {df.shape}")
    print(f"Train samples      : {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Class distribution (train): {train_dist_str}")
    print(f"Class distribution (test) : {test_dist_str}")
    print("----------------------------------------")
    print(f"Generations run    : {len(history_df)} (early stop: {early_stopped_str})")
    print(f"Best generation    : {best_gen}")
    print(f"Best fitness (cv)  : {best_fitness_cv:.4f}")
    print("----------------------------------------")
    print("Best hyperparameters found:")
    print(f"  hidden_layer_sizes : {best_params['hidden_layer_sizes']}")
    print(f"  activation         : {best_params['activation']}")
    print(f"  learning_rate_init : {best_params['learning_rate_init']}")
    print(f"  alpha              : {best_params['alpha']}")
    print(f"  max_iter           : {best_params['max_iter']}")
    print("----------------------------------------")
    print("Final MLP on test set:")
    print(f"  Accuracy           : {test_acc:.4f}")
    print(f"  F1-weighted        : {test_f1:.4f}")
    print(f"  Per-class F1       : {per_class_f1_str}")
    print("----------------------------------------")
    print("Output files:")
    print("  Stats CSV          : ga_gwo_history.csv")
    print("========================================")
    
    print("\nClassification Report:\n")
    print(class_report_str)

if __name__ == '__main__':
    main()
