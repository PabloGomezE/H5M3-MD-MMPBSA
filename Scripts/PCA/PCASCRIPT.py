import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats
import seaborn as sns

# =========================================================
# 1. Configuración de estilo profesional
# =========================================================
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")

FILES = {
    "H5M3": {"pc1": "proj_VXT_2D.xvg", "pc2": "proj_VXT_PC2.xvg"},
    "D1.1": {"pc1": "proj_D1_PC1.xvg", "pc2": "proj_D1_PC2.xvg"},
    "Astrakhan": {"pc1": "proj_Astra_PC1.xvg", "pc2": "proj_Astra_PC2.xvg"},
}
COLORS = {
    "H5M3": "#909090",
    "D1.1": "#d62728", 
    "Astrakhan": "#2b2b2b",
}
ALPHAS = {"H5M3": 0.6, "D1.1": 0.6, "Astrakhan": 0.6}
SIZES = {"H5M3": 1.2, "D1.1": 1.2, "Astrakhan": 1.2}

# =========================================================
# 2. Funciones de carga
# =========================================================
def load_pc_component(filepath: str):
    """Carga componente principal individual."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('#', '@', '&')) or line.strip() == '':
                continue
            vals = line.split()
            if len(vals) >= 2:
                data.append([float(vals[0]), float(vals[1])])
    
    data = np.array(data)
    time = data[:, 0] / 1000  # ps a ns
    pc   = data[:, 1]         # PC component
    return time, pc

def load_pca_data(system_files):
    """Combina PC1 y PC2 de un sistema."""
    time1, pc1 = load_pc_component(system_files["pc1"])
    time2, pc2 = load_pc_component(system_files["pc2"])
    
    # Verificar que los tiempos coincidan
    min_len = min(len(time1), len(time2))
    return pc1[:min_len], pc2[:min_len], time1[:min_len]

# =========================================================
# 3. Cargar todos los datos
# =========================================================
pca_data = {}
for system, files in FILES.items():
    if Path(files["pc1"]).exists() and Path(files["pc2"]).exists():
        pc1, pc2, time = load_pca_data(files)
        pca_data[system] = {"pc1": pc1, "pc2": pc2, "time": time}
        print(f"Cargado {system}: {len(pc1)} frames")
        print(f"  PC1 rango: {pc1.min():.3f} a {pc1.max():.3f}")
        print(f"  PC2 rango: {pc2.min():.3f} a {pc2.max():.3f}")

# =========================================================
# 4. Figura PCA de alta calidad
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Plot con densidad de contornos para cada sistema
for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    
    # Scatter plot principal
    scatter = ax.scatter(pc1, pc2, 
                        c=COLORS[system], 
                        alpha=ALPHAS[system], 
                        s=SIZES[system], 
                        label=system,
                        rasterized=True,
                        edgecolors='none')

# Añadir contornos de densidad para mejor visualización
for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    
    # Calcular densidad con kernel density estimation
    try:
        from scipy.stats import gaussian_kde
        xy = np.vstack([pc1, pc2])
        density = gaussian_kde(xy)
        
        # Crear grid para contornos
        pc1_range = np.linspace(pc1.min(), pc1.max(), 50)
        pc2_range = np.linspace(pc2.min(), pc2.max(), 50)
        PC1, PC2 = np.meshgrid(pc1_range, pc2_range)
        positions = np.vstack([PC1.ravel(), PC2.ravel()])
        Z = density(positions).reshape(PC1.shape)
        
        # Añadir contornos sutiles
        ax.contour(PC1, PC2, Z, levels=3, colors=COLORS[system], 
                  alpha=0.3, linewidths=0.8)
    except:
        pass  # Si falla, continúa sin contornos

# =========================================================
# 5. Estética profesional
# =========================================================
ax.set_xlabel("PC1 (nm)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("PC2 (nm)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Conformational Space Sampling\n(Principal Component Analysis)",
             fontsize=16, fontweight='bold', pad=20)

# Configurar ejes
ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=0.8)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

# Spines elegantes
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grid sutil
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')

# Leyenda profesional
#legend = ax.legend(fontsize=12, frameon=True, framealpha=0.95, 
 #                 edgecolor='black', fancybox=True, shadow=True,
  #                markerscale=4, loc='upper right')
#legend.get_frame().set_linewidth(1.0)

# =========================================================
# 6. Análisis cuantitativo
# =========================================================
print(f"\n{'='*60}")
print("ANÁLISIS CUANTITATIVO DEL SOLAPAMIENTO CONFORMACIONAL")
print("="*60)

# Calcular áreas de exploración
areas = {}
for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    area = (pc1.max() - pc1.min()) * (pc2.max() - pc2.min())
    areas[system] = area
    
    print(f"\n{system}:")
    print(f"  PC1: {pc1.mean():.3f} ± {pc1.std():.3f}")
    print(f"  PC2: {pc2.mean():.3f} ± {pc2.std():.3f}")
    print(f"  Área explorada: {area:.4f} nm²")

# Análisis de solapamiento por pares
from scipy.spatial.distance import cdist

systems = list(pca_data.keys())
print(f"\n{'Comparación':<15} {'Distancia centros':>18} {'Solapamiento estimado':>20}")
print("-" * 60)

for i, sys1 in enumerate(systems):
    for j, sys2 in enumerate(systems):
        if i < j:
            # Calcular distancia entre centroides
            center1 = [pca_data[sys1]["pc1"].mean(), pca_data[sys1]["pc2"].mean()]
            center2 = [pca_data[sys2]["pc1"].mean(), pca_data[sys2]["pc2"].mean()]
            dist = np.linalg.norm(np.array(center1) - np.array(center2))
            
            # Estimar solapamiento (simplificado)
            std1 = np.sqrt(pca_data[sys1]["pc1"].std()**2 + pca_data[sys1]["pc2"].std()**2)
            std2 = np.sqrt(pca_data[sys2]["pc1"].std()**2 + pca_data[sys2]["pc2"].std()**2)
            overlap_ratio = 1 / (1 + dist / (std1 + std2))
            
            print(f"{sys1:<7} vs {sys2:<7} {dist:>15.3f} nm {overlap_ratio:>18.1%}")

plt.tight_layout(pad=1.5)
plt.savefig("pca_conformational_space_hq.png", dpi=400, bbox_inches="tight", 
           facecolor='white', edgecolor='none')
plt.savefig("pca_conformational_space_hq.pdf", bbox_inches="tight", 
           facecolor='white', edgecolor='none')

print(f"\nFiguras guardadas: pca_conformational_space_hq.png/.pdf")
plt.show()


#######################

from scipy.stats import gaussian_kde

for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    pc1_95 = np.percentile(pc1, [2.5, 97.5])
    pc2_95 = np.percentile(pc2, [2.5, 97.5])
    area_95 = (pc1_95[1] - pc1_95[0]) * (pc2_95[1] - pc2_95[0])
    print(f"{system}: Área efectiva (95%) = {area_95:.3f} nm²")


###################################


for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    print(f"\n{system} - Estadísticas detalladas:")
    print(f"  PC1: mean={pc1.mean():.3f}, std={pc1.std():.3f}")
    print(f"  PC2: mean={pc2.mean():.3f}, std={pc2.std():.3f}")
    print(f"  PC1 percentiles: {np.percentile(pc1, [5, 25, 50, 75, 95])}")
    print(f"  PC2 percentiles: {np.percentile(pc2, [5, 25, 50, 75, 95])}")
#################################

for system, data in pca_data.items():
    print(f"\n{system} - Primeros 5 valores:")
    print(f"  PC1: {data['pc1'][:5]}")
    print(f"  PC2: {data['pc2'][:5]}")
    print(f"  Total frames: {len(data['pc1'])}")


####################
# Verificar asignación visual
for system, data in pca_data.items():
    print(f"{system} -> Color: {COLORS[system]}")
    print(f"  Centro: PC1={data['pc1'].mean():.3f}, PC2={data['pc2'].mean():.3f}")



########################

# Plot simple sin estética para verificar
fig, ax = plt.subplots()
for system, data in pca_data.items():
    ax.scatter(data['pc1'], data['pc2'], label=f"{system}", alpha=0.5, s=0.5)
    # Marcar el centro con una X grande
    ax.scatter(data['pc1'].mean(), data['pc2'].mean(), 
               marker='X', s=100, label=f"{system}_center")
ax.legend()
ax.set_title("Verificación simple")
plt.show()
