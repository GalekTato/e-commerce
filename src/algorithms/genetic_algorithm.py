import random
import string
import statistics as stats
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
OBJETIVO = "GENETICOSEVOLUCION"
POBLACION_SIZE = 150
TASA_MUTACION = 0.01  # 1% de probabilidad de mutación

def generar_individuo(longitud):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(longitud))

def calcular_fitness(individuo):
    return sum(1 for i in range(len(OBJETIVO)) if individuo[i] == OBJETIVO[i])

def cruzar(padre1, padre2):
    punto = random.randint(0, len(OBJETIVO) - 1)
    return padre1[:punto] + padre2[punto:]

def mutar(individuo):
    lista = list(individuo)
    for i in range(len(lista)):
        if random.random() < TASA_MUTACION:
            lista[i] = random.choice(string.ascii_uppercase)
    return ''.join(lista)

# --- ALMACENAMIENTO DE ESTADÍSTICAS ---
estad_min = []
estad_max = []
estad_prom = []
estad_std = []

# --- ALGORITMO GENÉTICO ---
poblacion = [generar_individuo(len(OBJETIVO)) for _ in range(POBLACION_SIZE)]
generacion = 0
mejor_individuo = ""

while mejor_individuo != OBJETIVO:
    generacion += 1

    # Evaluar fitness total
    fitness_vals = [calcular_fitness(ind) for ind in poblacion]

    # Guardar estadísticas
    estad_min.append(min(fitness_vals))
    estad_max.append(max(fitness_vals))
    estad_prom.append(stats.mean(fitness_vals))
    estad_std.append(stats.pstdev(fitness_vals))

    # Ordenar población
    poblacion = sorted(poblacion, key=lambda x: calcular_fitness(x), reverse=True)
    mejor_individuo = poblacion[0]

    print(f"Gen {generacion}: {mejor_individuo} (Fitness: {calcular_fitness(mejor_individuo)})")

    if mejor_individuo == OBJETIVO:
        break

    # Selección
    elite = poblacion[:POBLACION_SIZE // 4]
    nueva_poblacion = []

    # Nueva población por cruce + mutación
    while len(nueva_poblacion) < POBLACION_SIZE:
        p1 = random.choice(elite)
        p2 = random.choice(elite)
        hijo = cruzar(p1, p2)
        hijo = mutar(hijo)
        nueva_poblacion.append(hijo)

    poblacion = nueva_poblacion

print(f"\n¡Objetivo alcanzado en la generación {generacion}!")

# --- GRÁFICA DE ESTADÍSTICAS ---
plt.figure(figsize=(12, 6))
plt.plot(estad_min, label="Mínimo", linestyle="--")
plt.plot(estad_max, label="Máximo", linewidth=2)
plt.plot(estad_prom, label="Promedio", linewidth=2)
plt.plot(estad_std, label="Desviación estándar", linestyle=":")

plt.title("Estadísticas del Fitness por Generación")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.show()