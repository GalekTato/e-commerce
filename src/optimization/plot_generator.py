import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path

# Configuración Estética
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.dpi': 150
})

COLORS = {
    'primary': '#1f77b4', 'secondary': '#ff7f0e', 'tertiary': '#2ca02c',
    'danger': '#d62728', 'alpha': '#d62728', 'beta': '#3182bd', 'delta': '#31a354'
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_HISTORY_DIR = PROJECT_ROOT / 'results' / 'history'
RESULTS_REPORTS_DIR = PROJECT_ROOT / 'results' / 'reports'
PLOTS_GA_DIR = PROJECT_ROOT / 'plots' / 'ga'
PLOTS_GWO_DIR = PROJECT_ROOT / 'plots' / 'gwo'
PLOTS_MODEL_DIR = PROJECT_ROOT / 'plots' / 'model'

def load_data() -> tuple:
    hist_path = RESULTS_HISTORY_DIR
    rep_path = RESULTS_REPORTS_DIR

    ga_df = pd.read_csv(hist_path / 'ga_gwo_history.csv')
    ga_pop_df = pd.read_csv(hist_path / 'ga_population_history.csv')
    gwo_df = pd.read_csv(hist_path / 'gwo_wolves_history.csv')
    eval_df = pd.read_csv(hist_path / 'all_evaluated_individuals.csv')

    with open(rep_path / 'final_metrics.json', 'r') as f:
        metrics = json.load(f)

    y_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv').squeeze()
    y_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv').squeeze()
        
    return ga_df, ga_pop_df, gwo_df, eval_df, metrics, y_train, y_test

# --- GA PLOTS ---
def plot_ga_evolution(ga_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    gens = ga_df['generation']
    plt.plot(gens, ga_df['best_fitness'], label='Mejor', color=COLORS['tertiary'], marker='o')
    plt.plot(gens, ga_df['mean_fitness'], label='Media', color=COLORS['primary'], linestyle='--')
    plt.plot(gens, ga_df['worst_fitness'], label='Peor', color=COLORS['danger'], linestyle=':')
    plt.fill_between(gens, ga_df['mean_fitness'] - ga_df['std_fitness'], ga_df['mean_fitness'] + ga_df['std_fitness'], alpha=0.15)
    plt.title('Evolución de Aptitud GA')
    plt.xlabel('Generación')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ga_fitness_evolution.png'))
    plt.close()

def plot_ga_population_boxplot(ga_pop_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='generation', y='fitness', data=ga_pop_df, color='#9ecae1')
    plt.title('Distribución de Aptitud por Generación')
    plt.xlabel('Generación')
    plt.ylabel('F1-Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ga_population_boxplot.png'))
    plt.close()

def plot_ga_population_size(ga_pop_df: pd.DataFrame, save_path: str):
    counts = ga_pop_df.groupby('generation').size().reset_index(name='population_size')
    plt.figure(figsize=(8, 4))
    plt.plot(counts['generation'], counts['population_size'], color=COLORS['secondary'], marker='o')
    plt.title('Tamaño de la Población GA por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Número de individuos evaluados')
    plt.ylim(0, max(counts['population_size'].max() + 1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ga_population_size.png'))
    plt.close()

def plot_ga_diversity(ga_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(ga_df['generation'], ga_df['std_fitness'], color='purple', marker='s')
    plt.title('Diversidad Genética (Desviación Estándar)')
    plt.xlabel('Generación')
    plt.ylabel('Desviación Estándar del Fitness')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ga_diversity.png'))
    plt.close()

def plot_ga_hyperparameter_heatmap(eval_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 6))
    pivot = eval_df.pivot_table(index='hidden_layer_sizes', columns='activation', values='fitness', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
    plt.title('Heatmap Hiperparámetros vs F1-Score Promedio')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ga_hyperparameter_heatmap.png'))
    plt.close()

# --- GWO PLOTS ---
def plot_gwo_convergence(gwo_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    grouped = gwo_df.groupby('gwo_iteration').mean().reset_index()
    iters = grouped['gwo_iteration']
    plt.plot(iters, grouped['alpha_fitness'], label='Alfa', color=COLORS['alpha'], marker='s')
    plt.plot(iters, grouped['beta_fitness'], label='Beta', color=COLORS['beta'], marker='^')
    plt.plot(iters, grouped['delta_fitness'], label='Delta', color=COLORS['delta'], marker='v')
    plt.title('Convergencia GWO (Promedio de Fases)')
    plt.xlabel('Iteración Local GWO')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gwo_convergence.png'))
    plt.close()

def plot_gwo_position_spread(gwo_df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 5))
    grouped = gwo_df.groupby('gwo_iteration')['spread'].mean().reset_index()
    plt.plot(grouped['gwo_iteration'], grouped['spread'], color='brown', marker='D')
    plt.title('Dispersión Posicional Promedio de la Manada GWO')
    plt.xlabel('Iteración Local GWO')
    plt.ylabel('Distancia Media entre Lobos')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gwo_position_spread.png'))
    plt.close()

def plot_gwo_stats_table(gwo_df: pd.DataFrame, save_path: str):
    last_iters = gwo_df[gwo_df['gwo_iteration'] == gwo_df['gwo_iteration'].max()]
    stats = pd.DataFrame({
        'Media': [last_iters['alpha_fitness'].mean(), last_iters['beta_fitness'].mean(), last_iters['delta_fitness'].mean()],
        'Std': [last_iters['alpha_fitness'].std(), last_iters['beta_fitness'].std(), last_iters['delta_fitness'].std()],
        'Min': [last_iters['alpha_fitness'].min(), last_iters['beta_fitness'].min(), last_iters['delta_fitness'].min()],
        'Max': [last_iters['alpha_fitness'].max(), last_iters['beta_fitness'].max(), last_iters['delta_fitness'].max()]
    }, index=['Alfa', 'Beta', 'Delta']).round(4)
    
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=stats.values, colLabels=stats.columns, rowLabels=stats.index, cellLoc='center', loc='center')
    table.scale(1, 2)
    plt.title('Estadísticas Finales GWO')
    plt.savefig(os.path.join(save_path, 'gwo_stats_table.png'))
    plt.close()

# --- MODEL PLOTS ---
def plot_confusion_matrix(metrics: dict, save_path: str):
    cm = np.array(metrics['confusion_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_classification_report(metrics: dict, save_path: str):
    report = metrics['classification_report_dict']
    df_report = pd.DataFrame(report).iloc[:-1, :3].T
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Reporte de Clasificación')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'classification_report.png'))
    plt.close()

def plot_cross_validation(metrics: dict, save_path: str):
    cv_scores = metrics.get('cv_scores', [])
    if not cv_scores: return
    plt.figure(figsize=(7, 4))
    x_pos = np.arange(len(cv_scores)) + 1
    std_dev = np.std(cv_scores)
    plt.bar(x_pos, cv_scores, yerr=std_dev, color='skyblue', edgecolor='black', capsize=5)
    plt.axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Media: {np.mean(cv_scores):.3f}')
    plt.ylim(0.85, max(cv_scores) + std_dev * 2)
    plt.title('Validación Cruzada Estratificada (K=5)')
    plt.xlabel('Fold')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cross_validation.png'), dpi=150)
    plt.close()

def plot_train_test_class_distribution(y_train, y_test, save_path: str):
    plt.figure(figsize=(8, 5))
    tr_counts = y_train.value_counts(normalize=True).sort_index()
    te_counts = y_test.value_counts(normalize=True).sort_index()
    df = pd.DataFrame({'Train': tr_counts, 'Test': te_counts})
    df.index = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    df.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.title('Distribución de Clases (Train vs Test)')
    plt.ylabel('Proporción')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'train_test_class_distribution.png'))
    plt.close()

def plot_mlp_architecture(metrics: dict, save_path: str):
    best = metrics['best_params']
    layers = list(best['hidden_layer_sizes']) if isinstance(best['hidden_layer_sizes'], (list, tuple)) else [best['hidden_layer_sizes']]
    layer_sizes = [8] + layers + [3]  # Visually 8 input nodes
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    max_nodes = max(layer_sizes)
    fig_height = max(6, max_nodes * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    
    left, right = 0.1, 0.9
    layer_x = np.linspace(left, right, len(layer_sizes))
    
    node_coords = []
    
    for i, n_nodes in enumerate(layer_sizes):
        layer_top = 0.5 + (n_nodes * 0.4) / max_nodes
        layer_bottom = 0.5 - (n_nodes * 0.4) / max_nodes
        if n_nodes == 1:
            y_space = [0.5]
        else:
            y_space = np.linspace(layer_top, layer_bottom, n_nodes)
        
        node_coords.append(list(zip([layer_x[i]] * n_nodes, y_space)))
        
        color = colors[0] if i == 0 else (colors[2] if i == len(layer_sizes)-1 else colors[1])
        
        for idx, (x, y) in enumerate(node_coords[i]):
            if i == 0 and idx == 4:
                ax.text(x, y, "$\\cdot$\n$\\cdot$\n$\\cdot$", ha='center', va='center', fontsize=20, color=color, zorder=5)
            else:
                circle = plt.Circle((x, y), 0.015, color=color, zorder=4, ec='white', lw=1.5)
                ax.add_artist(circle)
            
        if i == 0:
            label = "Capa Entrada\n(41 descriptores en total)"
        elif i == len(layer_sizes)-1:
            label = "Capa Salida\n(3 Nodos)"
        else:
            label = f"Capa Oculta {i}\n({n_nodes} Nodos)"
            
        ax.text(layer_x[i], layer_top + 0.08, label, ha='center', va='bottom', fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    for i in range(len(layer_sizes) - 1):
        for idx1, (x1, y1) in enumerate(node_coords[i]):
            if i == 0 and idx1 == 4: continue
            for idx2, (x2, y2) in enumerate(node_coords[i+1]):
                ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.15, lw=0.6, zorder=1)
                
    plt.title('Arquitectura Completa del Perceptrón Multicapa Óptimo', fontsize=16, weight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mlp_architecture.png'), dpi=150, bbox_inches='tight', transparent=False)
    plt.close()

def plot_perceptron_diagram(save_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.text(0.2, 0.8, "$x_1$", ha='center', va='center', fontsize=12)
    ax.text(0.2, 0.5, "$x_2$", ha='center', va='center', fontsize=12)
    ax.text(0.2, 0.2, "$x_n$", ha='center', va='center', fontsize=12)
    ax.plot([0.3, 0.5], [0.8, 0.5], 'k->')
    ax.plot([0.3, 0.5], [0.5, 0.5], 'k->')
    ax.plot([0.3, 0.5], [0.2, 0.5], 'k->')
    ax.text(0.35, 0.7, "$w_1$", fontsize=10)
    ax.text(0.35, 0.55, "$w_2$", fontsize=10)
    ax.text(0.35, 0.3, "$w_n$", fontsize=10)
    
    circle = plt.Circle((0.55, 0.5), 0.1, color='lightblue', ec='black')
    ax.add_patch(circle)
    ax.text(0.55, 0.5, r"$\sum + b$", ha='center', va='center', fontsize=14)
    
    ax.plot([0.65, 0.8], [0.5, 0.5], 'k->')
    ax.text(0.72, 0.55, "$f(z)$", fontsize=12)
    ax.text(0.85, 0.5, "Salida $y$", fontsize=12)
    
    plt.title('Diagrama del Perceptrón Base')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'perceptron_diagram.png'))
    plt.close()

def plot_hybrid_evolution_timeline(ga_df: pd.DataFrame, gwo_df: pd.DataFrame, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ga_x = np.arange(1, len(ga_df) + 1)
    ga_y = ga_df['best_fitness'].values
    ax1.plot(ga_x, ga_y, label='Fase GA', color='green', marker='o', linestyle='-')
    
    ga_mejora = ga_y[-1] - ga_y[0]
    gwo_mejora = 0.0
    
    ax1.set_title("Fase GA (exploración global)")
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Mejor F1-Score')
    ax1.grid(True, alpha=0.3)
    
    if not gwo_df.empty:
        last_gen = gwo_df['generation'].max()
        gwo_run = gwo_df[gwo_df['generation'] == last_gen]
        gwo_y = gwo_run['alpha_fitness'].values
        
        gwo_x = np.arange(1, len(gwo_y) + 1)
        ax2.plot(gwo_x, gwo_y, label='Fase GWO', color='red', marker='s', linestyle='--')
        
        ax2.set_title("Fase GWO (explotación local)")
        ax2.set_xlabel('Iteración GWO')
        ax2.set_ylabel('Mejor F1-Score')
        ax2.grid(True, alpha=0.3)
        
        # Zoomed Y-axis
        y_min, y_max = 0.8840, 0.8960
        if min(gwo_y) < y_min or max(gwo_y) > y_max:
            margin = (max(gwo_y) - min(gwo_y)) * 0.1
            if margin == 0: margin = 0.005
            y_min = min(gwo_y) - margin
            y_max = max(gwo_y) + margin
        ax2.set_ylim(y_min, y_max)
        
        gwo_mejora = gwo_y[-1] - gwo_y[0]
        
        textstr = f"Mejora GA: +{ga_mejora:.4f} | Mejora GWO: +{gwo_mejora:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    else:
        ax2.set_title("Fase GWO (No disponible)")
        ax2.axis('off')

    plt.suptitle('Evolución Híbrida GA+GWO del Mejor Fitness', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hybrid_evolution_timeline.png'), dpi=150)
    plt.close()

def main():
    # Crear carpetas si no existen
    PLOTS_GA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_GWO_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando datos para generar 15 gráficas...")
    ga_df, ga_pop_df, gwo_df, eval_df, metrics, y_train, y_test = load_data()
    
    # GA
    plot_ga_evolution(ga_df, str(PLOTS_GA_DIR))
    plot_ga_population_boxplot(ga_pop_df, str(PLOTS_GA_DIR))
    plot_ga_population_size(ga_pop_df, str(PLOTS_GA_DIR))
    plot_ga_diversity(ga_df, str(PLOTS_GA_DIR))
    plot_ga_hyperparameter_heatmap(eval_df, str(PLOTS_GA_DIR))
    
    # GWO
    if not gwo_df.empty:
        plot_gwo_convergence(gwo_df, str(PLOTS_GWO_DIR))
        plot_gwo_position_spread(gwo_df, str(PLOTS_GWO_DIR))
        plot_gwo_stats_table(gwo_df, str(PLOTS_GWO_DIR))
        
    # MODEL
    plot_confusion_matrix(metrics, str(PLOTS_MODEL_DIR))
    plot_classification_report(metrics, str(PLOTS_MODEL_DIR))
    plot_cross_validation(metrics, str(PLOTS_MODEL_DIR))
    plot_train_test_class_distribution(y_train, y_test, str(PLOTS_MODEL_DIR))
    plot_mlp_architecture(metrics, str(PLOTS_MODEL_DIR))
    plot_perceptron_diagram(str(PLOTS_MODEL_DIR))
    plot_hybrid_evolution_timeline(ga_df, gwo_df, str(PLOTS_MODEL_DIR))
    
    print("==================================================")
    print(f"Todas las gráficas han sido generadas en {PROJECT_ROOT / 'plots'}")
    print("==================================================")

if __name__ == '__main__':
    main()