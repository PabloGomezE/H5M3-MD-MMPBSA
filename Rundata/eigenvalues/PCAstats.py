import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path


FILES = {
    "D1.1": {"pc1": "proj_D1_PC1.xvg", "pc2": "proj_D1_PC2.xvg"},
    "Astrakhan": {"pc1": "proj_Astra_PC1.xvg", "pc2": "proj_Astra_PC2.xvg"},
    "H5M3": {"pc1": "proj_VXT_PC1_subsampled.xvg", "pc2": "proj_VXT_PC2.xvg"},
}

COLORS = {
    "D1.1": "#2ca02c",
    "Astrakhan": "#1f77b4",
    "H5M3": "#d62728",
}


def load_pc_component(filepath: str):
    """Carga componente PC de archivo .xvg"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('#', '@', '&')) or line.strip() == '':
                continue
            vals = line.split()
            if len(vals) >= 2:
                data.append([float(vals[0]), float(vals[1])])
    
    data = np.array(data)
    time = data[:, 0] / 1000
    pc = data[:, 1]
    return time, pc


def load_pca_data(system_files):
    """Carga PC1 y PC2, sincroniza longitudes"""
    time1, pc1 = load_pc_component(system_files["pc1"])
    time2, pc2 = load_pc_component(system_files["pc2"])
    min_len = min(len(time1), len(time2))
    return pc1[:min_len], pc2[:min_len], time1[:min_len]


def calculate_overlap_percentage(coords1, coords2, threshold=0.5):
    """
    Calcula porcentaje de overlap entre dos conjuntos de puntos.
    Usa distancia mínima a los puntos del otro conjunto.
    
    Args:
        coords1: Array (N, 2) - coordenadas del sistema 1
        coords2: Array (M, 2) - coordenadas del sistema 2
        threshold: distancia máxima para considerar un punto como "overlapping" (nm)
    
    Returns:
        overlap_pct: porcentaje de overlap
    """
    # Calcular distancia mínima de cada punto en coords1 a coords2
    distances = cdist(coords1, coords2, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    # Puntos que están dentro del threshold
    overlap_points = np.sum(min_distances <= threshold)
    overlap_pct = (overlap_points / len(coords1)) * 100
    
    return overlap_pct


def calculate_centroid_distance(coords1, coords2):
    """Calcula distancia euclidiana entre centroides"""
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance, centroid1, centroid2



print("Cargando datos PCA...")
pca_data = {}
for system, files in FILES.items():
    if Path(files["pc1"]).exists() and Path(files["pc2"]).exists():
        try:
            pc1, pc2, time = load_pca_data(files)
            coords = np.column_stack((pc1, pc2))
            pca_data[system] = {"pc1": pc1, "pc2": pc2, "coords": coords, "time": time}
            print(f"✓ {system}: {len(pc1)} frames")
        except Exception as e:
            print(f"✗ Error cargando {system}: {e}")



print("\n" + "="*70)
print("OVERLAP PERCENTAGE ANALYSIS (threshold = 0.5 nm)")
print("="*70)

threshold = 0.5
overlap_matrix = {}

systems = list(pca_data.keys())
for i, sys1 in enumerate(systems):
    for j, sys2 in enumerate(systems):
        if i != j:
            coords1 = pca_data[sys1]["coords"]
            coords2 = pca_data[sys2]["coords"]
            
            overlap_pct = calculate_overlap_percentage(coords1, coords2, threshold=threshold)
            key = f"{sys1} → {sys2}"
            overlap_matrix[key] = overlap_pct
            
            print(f"\n{key}:")
            print(f"  {overlap_pct:.2f}% de {sys1} overlaps con {sys2}")


print("\n" + "="*70)
print("CENTROID DISTANCE ANALYSIS")
print("="*70)

centroids = {}
for system in pca_data.keys():
    coords = pca_data[system]["coords"]
    centroid = np.mean(coords, axis=0)
    centroids[system] = centroid
    print(f"\n{system} centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}) nm")

print("\n" + "-"*70)
print("Pairwise Centroid Distances:")
print("-"*70)

centroid_distances = {}
for i, sys1 in enumerate(systems):
    for j, sys2 in enumerate(systems):
        if i < j:  # Solo pares únicos
            dist, c1, c2 = calculate_centroid_distance(
                pca_data[sys1]["coords"], 
                pca_data[sys2]["coords"]
            )
            key = f"{sys1} ↔ {sys2}"
            centroid_distances[key] = dist
            print(f"\n{key}:")
            print(f"  Distancia: {dist:.3f} nm")
            print(f"  {sys1} centroid: ({c1[0]:.3f}, {c1[1]:.3f})")
            print(f"  {sys2} centroid: ({c2[0]:.3f}, {c2[1]:.3f})")



print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

print("\nOverlap Matrix (%):")
print("-" * 50)
print(f"{'From':<15} {'To':<15} {'Overlap %':<15}")
print("-" * 50)
for key, value in sorted(overlap_matrix.items()):
    sys1, sys2 = key.split(' → ')
    print(f"{sys1:<15} {sys2:<15} {value:>6.2f}%")

print("\n\nCentroid Distances (nm):")
print("-" * 50)
print(f"{'System Pair':<25} {'Distance (nm)':<15}")
print("-" * 50)
for key, value in sorted(centroid_distances.items()):
    print(f"{key:<25} {value:>8.3f}")



print("\nGenerando heatmap de overlap...")

fig, ax = plt.subplots(figsize=(8, 6))

# Crear matriz de overlap simétrica
overlap_heatmap = np.zeros((3, 3))
sys_list = ["D1.1", "Astrakhan", "H5M3"]
sys_to_idx = {sys: i for i, sys in enumerate(sys_list)}

for key, value in overlap_matrix.items():
    sys1, sys2 = key.split(' → ')
    i = sys_to_idx[sys1]
    j = sys_to_idx[sys2]
    overlap_heatmap[i, j] = value

im = ax.imshow(overlap_heatmap, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto')

# Labels
ax.set_xticks(range(len(sys_list)))
ax.set_yticks(range(len(sys_list)))
ax.set_xticklabels(sys_list, fontsize=11, fontweight='bold')
ax.set_yticklabels(sys_list, fontsize=11, fontweight='bold')

# Título
ax.set_title('Overlap Percentage Matrix\n(threshold = 0.5 nm)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Target System', fontsize=11, fontweight='bold')
ax.set_ylabel('Source System', fontsize=11, fontweight='bold')

# Valores en las celdas
for i in range(len(sys_list)):
    for j in range(len(sys_list)):
        if i != j:
            text = ax.text(j, i, f'{overlap_heatmap[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Overlap (%)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("overlap_heatmap.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.savefig("overlap_heatmap.pdf", bbox_inches="tight", facecolor='white')
print("✓ Guardado: overlap_heatmap.png/.pdf")
plt.show()



print("Generando gráfico de centroid distances...")

fig, ax = plt.subplots(figsize=(10, 6))

systems_pairs = list(centroid_distances.keys())
distances = list(centroid_distances.values())
colors_list = ['#2ca02c', '#1f77b4', '#d62728'][:len(systems_pairs)]

bars = ax.barh(systems_pairs, distances, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

# Valores en las barras
for i, (bar, dist) in enumerate(zip(bars, distances)):
    ax.text(dist + 0.05, i, f'{dist:.3f} nm', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Distance (nm)', fontsize=12, fontweight='bold')
ax.set_title('Centroid Distances Between Systems (PCA Space)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, max(distances) * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle=':')

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("centroid_distances.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.savefig("centroid_distances.pdf", bbox_inches="tight", facecolor='white')
print("✓ Guardado: centroid_distances.png/.pdf")
plt.show()

print("\n✓ Análisis de overlap y centroid distances completado!")
