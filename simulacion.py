# Simulación de transferencias en red (Barabási–Albert)
# Requisitos: networkx, numpy, matplotlib, scipy
# pip install networkx numpy matplotlib scipy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
from collections import defaultdict
import random
import itertools
from joblib import Parallel, delayed

# --------------------------
# Parámetros (ajustables)
# --------------------------
N = 40           # número de nodos
m = 5              # enlaces por nuevo nodo (BA)
steps = 300000     # pasos temporales de transferencia
f = 0.07           # fracción de riqueza transferida del emisor en cada transacción
alpha = 0.6        # peso entre grado y riqueza para elegir receptor (0 = solo riqueza, 1 = solo grado)
seed = 41

np.random.seed(seed)
random.seed(seed)
# --------------------------
# Funciones auxiliares
# --------------------------
def gini(array):
    """Cálculo del índice de Gini (0..1)."""
    a = np.array(array, dtype=float)
    if a.size == 0:
        return 0.0
    if a.mean() == 0:
        return 0.0
    diff = np.abs(np.subtract.outer(a, a)).mean()
    return diff / (2.0 * a.mean())

# --------------------------
# Construcción de la red
# --------------------------
G = nx.barabasi_albert_graph(N, m, seed=seed)

# riqueza inicial: todos iguales (1.0) para observar solo dinámica
def generar_riqueza(node):
    return node, 1

resultados = Parallel(n_jobs=-1)(delayed(generar_riqueza)(node) for node in G.nodes())
wealth = {node: w for node, w in resultados}
#class_group = {node: np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]) for node in G.nodes()}
#wealth = {node: 1.0 for node in G.nodes()}
valores = np.array(list(wealth.values()))
p33, p66 = np.percentile(valores, [33, 66])

def asignar_clase(node, w):
    if w < p33:
        return node, 0
    elif w < p66:
        return node, 1
    else:
        return node, 2

resultados = Parallel(n_jobs=-1)(delayed(asignar_clase)(node, wealth[node]) for node in G.nodes())
class_group = {node: cls for node, cls in resultados}
# Layout fijo (para comparar antes/después)
pos = nx.spring_layout(G, seed=seed)

# Visualización inicial
plt.figure(figsize=(7, 6))
nx.draw_networkx_edges(G, pos, alpha=0.2)
nodes = nx.draw_networkx_nodes(
    G, pos, node_size=60, node_color=list(wealth.values()), cmap='plasma'
)
plt.title("Red inicial (riqueza uniforme)")
plt.colorbar(nodes, label="Riqueza")
plt.axis("off")
plt.tight_layout()
plt.show()

# registro de Gini en el tiempo (muestrear cada certain steps)
sample_every = max(1, steps // 200)  # ajustar muestras (como máximo ~200 puntos)
gini_times = []
time_points = []
# contador de tiempo que llevan en la clase actual (para histeresis)
class_time = {node: 0 for node in G.nodes()}

# parámetros de movilidad
class_update_every = max(1, steps // 200)  # cada cuántos pasos recomputar clases (ej: ~200 muestras)
min_hold = max(1, steps // 500)            # pasos mínimos en la "nueva clase" antes de hacer el cambio efectivo
# definición de percentiles para asignar clases según riqueza actual:
# por ejemplo: top 10% => clase alta; siguiente 30% => clase media; resto => baja
pct_high = 0.10
pct_mid = 0.30
# --------------------------
# Dinámica de transferencias
# --------------------------
# Reglas:
#  - Elegir un emisor aleatorio
#  - Elegir receptor entre sus vecinos con probabilidad proporcional a alpha*degree + (1-alpha)*wealth
#  - Transferir fraction f de la riqueza del emisor (si emisor no tiene vecinos o wealth=0 se salta)
# --------------------------
nodes_list = list(G.nodes())
total_created_wealth = 0.0
total_created_over_time = [] 
for t in range(steps):
    sender = random.choice(nodes_list)
    sender_wealth = wealth[sender]
    neighbors = list(G.neighbors(sender))

    if sender_wealth > 0 and neighbors:
        # -------------------- scores de vecinos --------------------
        scores = np.array([
            (1.5 if class_group[sender] == class_group[nb] else 1.0) * 
            (alpha * G.degree(nb) + (1 - alpha) * wealth[nb])
            for nb in neighbors
        ], dtype=float)

        # seguridad contra NaN o suma cero
        if scores.sum() <= 0 or np.isnan(scores).any():
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = scores / scores.sum()

        receiver = np.random.choice(neighbors, p=probs)

        # -------------------- cooperación y transferencia --------------------
        same_class = class_group[sender] == class_group[receiver]
        amount = f * sender_wealth
        coop_prob = 0.8 if same_class else 0.5  # intra-clase coop más probable
        cooperate = random.random() < coop_prob
        if cooperate:
            # cooperación exitosa
            created = amount * (0.1 if same_class else 0.5)
            wealth[sender] += created / 2
            wealth[receiver] += amount + created / 2
            total_created_wealth += created 
        else:
            # no cooperación: dilema del prisionero
            # el receptor "gana", el emisor "pierde"
            loss = amount * (0.5 if same_class else 0.1)
            wealth[sender] -= (amount+loss)
            wealth[receiver] += amount
            total_created_wealth -= loss
        if t % sample_every == 0:
            total_created_over_time.append(total_created_wealth)
        # -------------------- ruptura de lazos --------------------
        for nb in neighbors:
            if nb != receiver:
                # solo romper lazos con los que no cooperaron
                if not cooperate and random.random() < random.uniform(0.1, 0.5):
                    if G.has_edge(sender, nb):
                        G.remove_edge(sender, nb)

        # -------------------- creación de nuevas conexiones --------------------
        p_new_connection = 0.01
        num_attempts = int(len(nodes_list) * 0.05)
        for _ in range(num_attempts):
            u, v = random.sample(nodes_list, 2)
            if not G.has_edge(u, v) and random.random() < p_new_connection:
                G.add_edge(u, v, weight=0.5)
                G[u][v]["cross_class"] = class_group[u] != class_group[v]

    # -------------------- registro de Gini --------------------
    if t % sample_every == 0:
        arr = np.array(list(wealth.values()))
        gini_times.append(gini(arr))
        time_points.append(t)

    # -------------------- actualización de clases --------------------
    if t % class_update_every == 0 and t > 0:
        vals = np.array(list(wealth.values()))
        thr_high = np.quantile(vals, 1.0 - pct_high)
        thr_mid  = np.quantile(vals, 1.0 - (pct_high + pct_mid))

        # proponer nuevas clases
        proposed = {}
        for node in G.nodes():
            w = wealth[node]
            if w >= thr_high:
                proposed[node] = 2
            elif w >= thr_mid:
                proposed[node] = 1
            else:
                proposed[node] = 0

        # aplicar histeresis
        for node in G.nodes():
            if proposed[node] == class_group[node]:
                class_time[node] += class_update_every
            else:
                class_time[node] = 0

        for node in G.nodes():
            new_class = proposed[node]
            if new_class != class_group[node]:
                if new_class == 2 and wealth[node] >= thr_high * 1.02:
                    class_group[node] = new_class
                elif new_class == 1 and wealth[node] >= thr_mid * 1.01:
                    class_group[node] = new_class
                elif new_class == 0 and wealth[node] <= thr_mid * 0.99:
                    class_group[node] = new_class
            class_time[node] += class_update_every
# --------------------------
# Métricas finales
# --------------------------
wealth_values = np.array([wealth[node] for node in G.nodes()])
degrees = np.array([G.degree(node) for node in G.nodes()])
neighbor_sum = np.array([sum(wealth[n] for n in G.neighbors(node)) for node in G.nodes()])

print(f"Pasos: {steps}, N={N}, m={m}, f={f}, alpha={alpha}")
print(f"Gini final: {gini(wealth_values):.4f}")
print(f"Top 1% share: {np.sort(wealth_values)[-max(1,int(0.01*N)) :].sum() / wealth_values.sum():.4f}")

corr_deg, _ = pearsonr(degrees, wealth_values)
corr_nei, _ = pearsonr(neighbor_sum, wealth_values)
print(f"Correlación grado-riqueza (Pearson): {corr_deg:.4f}")
print(f"Correlación suma-vecinos-riqueza (Pearson): {corr_nei:.4f}")

# --------------------------
# Gráficos
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(time_points, gini_times, marker='o', markersize=3)
plt.xlabel("Paso")
plt.ylabel("Gini")
plt.title("Evolución del Gini durante la dinámica de transferencias")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histograma de riqueza final (log bins para ver cola)
plt.figure(figsize=(8,4))
vals = wealth_values
plt.hist(vals, bins=np.logspace(np.log10(max(vals.min(),1e-6)), np.log10(vals.max()+1e-6), 50))
plt.xscale('log')
plt.xlabel("Riqueza (escala log)")
plt.ylabel("Frecuencia")
plt.title("Histograma de riqueza final (escala log)")
plt.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.show()

# Scatter: grado vs riqueza (con tendencia)
plt.figure(figsize=(6,5))
plt.scatter(degrees, wealth_values, alpha=0.5, s=12)
# línea de tendencia (bin means)
bins = np.unique(degrees)
means = [wealth_values[degrees==b].mean() for b in bins]
plt.plot(bins, means, 'r-', lw=2, label='mean wealth por grado')
plt.xlabel("Grado (conexiones)")
plt.ylabel("Riqueza")
plt.title("Grado vs riqueza (final)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter: suma riqueza vecinos vs riqueza propia
plt.figure(figsize=(6,5))
plt.scatter(neighbor_sum, wealth_values, alpha=0.6, color='orange', s=12)
plt.xlabel("Suma de riqueza de los vecinos")
plt.ylabel("Riqueza propia")
plt.title("Riqueza propia vs riqueza de vecinos (final)")
plt.grid(True)
plt.tight_layout()
plt.show()
# Visualización final
# Calcular percentiles para clasificar nodos
vals = np.array(list(wealth.values()))
thr_top1 = np.quantile(vals, 0.99)
thr_top10 = np.quantile(vals, 0.90)

# Asignar colores según riqueza
node_colors = []
for w in vals:
    if w >= thr_top1:
        node_colors.append('green')     # Top 1%
    elif w >= thr_top10:
        node_colors.append('yellow')    # 10-1%
    else:
        node_colors.append('red')       # resto

pos_final = nx.spring_layout(G, pos=pos, iterations=100, seed=seed)
plt.figure(figsize=(8, 7))
nx.draw_networkx_edges(G, pos_final, alpha=0.2)
nx.draw_networkx_nodes(G, pos_final, node_size=60, node_color=node_colors)
plt.title("Red final (colores = riqueza por percentil)")
plt.axis("off")
plt.tight_layout()
plt.show()
x = range(len(total_created_over_time))
y = np.array(total_created_over_time)

plt.figure(figsize=(10,5))
plt.plot(x, y, color='black', lw=2, label='Riqueza neta')
plt.fill_between(x, y, 0, where=(y >= 0), facecolor='green', alpha=0.4, interpolate=True)
plt.fill_between(x, y, 0, where=(y < 0), facecolor='red', alpha=0.4, interpolate=True)

plt.axhline(0, color='gray', lw=1, linestyle='--')  # línea cero
plt.xlabel("Muestra temporal", fontsize=12)
plt.ylabel("Riqueza acumulada generada", fontsize=12)
plt.title("Evolución de la riqueza generada durante la simulación", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()