import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import threading



distancias = np.array([
    [0,2,2,5,7],
    [2,0,4,8,2],
    [2,4,0,1,3],
    [5,8,1,0,2],
    [7,2,3,2,0]
])


n_ciudades = len(distancias)
n_hormigas = 100
n_iteraciones = 50
alpha = 1
beta = 2
rho = 0.5
Q = 100

feromonas = np.ones((n_ciudades, n_ciudades))
lock = threading.Lock()  # Para proteger lectura/escritura de feromonas


def construir_ruta(feromonas):
    ruta = [random.randint(0, n_ciudades - 1)]

    while len(ruta) < n_ciudades:
        i = ruta[-1]
        no_visitados = list(set(range(n_ciudades)) - set(ruta))
        probs = probabilidad(i, no_visitados, feromonas)
        siguiente = random.choices(no_visitados, weights=probs)[0]
        ruta.append(siguiente)

    return ruta


def probabilidad(i, no_visitados, feromonas):
    probs = []
    for j in no_visitados:
        tau = feromonas[i][j] ** alpha
        eta = (1 / distancias[i][j]) ** beta
        probs.append(tau * eta)
    probs = np.array(probs)
    return probs / probs.sum()


def longitud_ruta(ruta):
    total = 0
    for i in range(len(ruta) - 1):
        total += distancias[ruta[i]][ruta[i + 1]]
    total += distancias[ruta[-1]][ruta[0]]
    return total


def hormiga_tarea(_):
    """Cada hormiga toma una snapshot de feromonas y construye su ruta de forma independiente."""
    with lock:
        feromonas_snapshot = feromonas.copy()  # Lectura segura
    ruta = construir_ruta(feromonas_snapshot)
    return ruta


def actualizar_feromonas(feromonas, rutas):
    feromonas *= (1 - rho)

    for ruta in rutas:
        L = longitud_ruta(ruta)
        for i in range(len(ruta) - 1):
            x, y = ruta[i], ruta[i + 1]
            feromonas[x][y] += Q / L
            feromonas[y][x] += Q / L

        feromonas[ruta[-1]][ruta[0]] += Q / L
        feromonas[ruta[0]][ruta[-1]] += Q / L


mejor_ruta = None
mejor_longitud = float('inf')

with ThreadPoolExecutor(max_workers=n_hormigas) as executor:
    for _ in range(n_iteraciones):
        # Todas las hormigas corren en paralelo
        rutas = list(executor.map(hormiga_tarea, range(n_hormigas)))

        for ruta in rutas:
            L = longitud_ruta(ruta)
            if L < mejor_longitud:
                mejor_longitud = L
                mejor_ruta = ruta

        with lock:
            actualizar_feromonas(feromonas, rutas)

print("Mejor ruta:", mejor_ruta)
print("Longitud de ruta:", mejor_longitud)