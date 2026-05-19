import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.patches import Ellipse


FILES = {
    "D1.1": {"pc1": "proj_D1_PC1.xvg", "pc2": "proj_D1_PC2.xvg"},                    # Verde
    "Astrakhan": {"pc1": "proj_Astra_PC1.xvg", "pc2": "proj_Astra_PC2.xvg"},         # Azul
    "H5M3": {"pc1": "proj_VXT_PC1_subsampled.xvg", "pc2": "proj_VXT_PC2.xvg"},       # Rojo (NUEVO)
}

COLORS = {
    "D1.1": "#2ca02c",      # Verde
    "Astrakhan": "#1f77b4", # Azul  
    "H5M3": "#d62728",      # Rojo
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
    time = data[:, 0] / 1000  # Convertir a ns
    pc   = data[:, 1]
    return time, pc


def load_pca_data(system_files):
    """Carga PC1 y PC2, sincroniza longitudes"""
    time1, pc1 = load_pc_component(system_files["pc1"])
    time2, pc2 = load_pc_component(system_files["pc2"])
    min_len = min(len(time1), len(time2))
    return pc1[:min_len], pc2[:min_len], time1[:min_len]


def confidence_ellipse(x, y, ax, confidence=0.95, **kwargs):
    """Dibuja elipse de confianza 95% alrededor de los datos."""
    if len(x) < 2:
        return
    
    cov = np.cov(x, y)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    from scipy.stats import chi2
    chi2_val = chi2.ppf(confidence, df=2)
    
    width, height = 2 * np.sqrt(eigenvals * chi2_val)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    ellipse = Ellipse((x.mean(), y.mean()), width, height, 
                     angle=angle, **kwargs)
    ax.add_patch(ellipse)



print("Cargando datos PCA...")
pca_data = {}
for system, files in FILES.items():
    if Path(files["pc1"]).exists() and Path(files["pc2"]).exists():
        try:
            pc1, pc2, time = load_pca_data(files)
            pca_data[system] = {"pc1": pc1, "pc2": pc2, "time": time}
            print(f"✓ {system}: {len(pc1)} frames (PC1), {len(pc2)} frames (PC2)")
        except Exception as e:
            print(f"✗ Error cargando {system}: {e}")
    else:
        print(f"✗ Archivos no encontrados para {system}")



print("\nGenerando figura PCA 2D...")
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Scatter plot
for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    
    ax.scatter(pc1, pc2, 
               c=COLORS[system], 
               alpha=0.6, 
               s=2, 
               label=system,
               rasterized=True)

# Confidence 95%
for system, data in pca_data.items():
    pc1, pc2 = data["pc1"], data["pc2"]
    
    confidence_ellipse(pc1, pc2, ax, confidence=0.95,
                      facecolor='none', 
                      edgecolor=COLORS[system], 
                      linewidth=1.5,
                      linestyle='-',
                      alpha=0.8)


ax.set_xlabel("PC1 (nm)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("PC2 (nm)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Conformational Space Sampling\n(Principal Component Analysis)",
             fontsize=16, fontweight='bold', pad=20)

#ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
ax.tick_params(axis='both', which='minor', labelsize=10, length=3, width=0.8)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, color='gray')

plt.tight_layout(pad=1.5)
plt.savefig("pca_2d_with_ellipses.png", dpi=400, bbox_inches="tight", 
           facecolor='white', edgecolor='none')
plt.savefig("pca_2d_with_ellipses.pdf", bbox_inches="tight", 
           facecolor='white', edgecolor='none')
print("✓ Guardado: pca_2d_with_ellipses.png/.pdf")
plt.show()



print("\n" + "="*60)
print("ANÁLISIS TEMPORAL: H5M3 (PRIMERA vs SEGUNDA MITAD)")
print("="*60)

h5m3_pc1 = pca_data["H5M3"]["pc1"]
h5m3_pc2 = pca_data["H5M3"]["pc2"]
mid_point = len(h5m3_pc1) // 2

first_half_pc1 = h5m3_pc1[:mid_point]
second_half_pc1 = h5m3_pc1[mid_point:]
first_half_pc2 = h5m3_pc2[:mid_point]
second_half_pc2 = h5m3_pc2[mid_point:]

print(f"\nPC1 Analysis:")
print(f"  Primera mitad (40 ns): media={first_half_pc1.mean():.3f} nm, std={first_half_pc1.std():.3f} nm")
print(f"  Segunda mitad (40 ns): media={second_half_pc1.mean():.3f} nm, std={second_half_pc1.std():.3f} nm")
print(f"  Diferencia de medias: {abs(first_half_pc1.mean() - second_half_pc1.mean()):.3f} nm")

print(f"\nPC2 Analysis:")
print(f"  Primera mitad (40 ns): media={first_half_pc2.mean():.3f} nm, std={first_half_pc2.std():.3f} nm")
print(f"  Segunda mitad (40 ns): media={second_half_pc2.mean():.3f} nm, std={second_half_pc2.std():.3f} nm")
print(f"  Diferencia de medias: {abs(first_half_pc2.mean() - second_half_pc2.mean()):.3f} nm")

print(f"\nOverlap Analysis:")
pc1_range_first = (first_half_pc1.min(), first_half_pc1.max())
pc1_range_second = (second_half_pc1.min(), second_half_pc1.max())
print(f"  PC1 rango primera mitad: {pc1_range_first[0]:.3f} to {pc1_range_first[1]:.3f} nm")
print(f"  PC1 rango segunda mitad: {pc1_range_second[0]:.3f} to {pc1_range_second[1]:.3f} nm")

pc2_range_first = (first_half_pc2.min(), first_half_pc2.max())
pc2_range_second = (second_half_pc2.min(), second_half_pc2.max())
print(f"  PC2 rango primera mitad: {pc2_range_first[0]:.3f} to {pc2_range_first[1]:.3f} nm")
print(f"  PC2 rango segunda mitad: {pc2_range_second[0]:.3f} to {pc2_range_second[1]:.3f} nm")



print("\nGenerando figura evolución temporal...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

time = pca_data["H5M3"]["time"]
pc1 = pca_data["H5M3"]["pc1"]
pc2 = pca_data["H5M3"]["pc2"]


ax1.plot(time, pc1, color='#d62728', alpha=0.7, linewidth=0.8, label='PC1')
ax1.axhline(first_half_pc1.mean(), color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5, label='Primera mitad media')
ax1.axhline(second_half_pc1.mean(), color='darkred', linestyle='--', alpha=0.5, linewidth=1.5, label='Segunda mitad media')
ax1.axvline(40, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.set_ylabel("PC1 (nm)", fontsize=12, fontweight='bold')
ax1.set_title("H5M3: Evolución temporal de PC1", fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend(loc='best', fontsize=10)
ax1.set_xlim(time.min(), time.max())


ax2.plot(time, pc2, color='#ff7f0e', alpha=0.7, linewidth=0.8, label='PC2')
ax2.axhline(first_half_pc2.mean(), color='#ff7f0e', linestyle='--', alpha=0.5, linewidth=1.5, label='Primera mitad media')
ax2.axhline(second_half_pc2.mean(), color='darkorange', linestyle='--', alpha=0.5, linewidth=1.5, label='Segunda mitad media')
ax2.axvline(40, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax2.set_xlabel("Tiempo (ns)", fontsize=12, fontweight='bold')
ax2.set_ylabel("PC2 (nm)", fontsize=12, fontweight='bold')
ax2.set_title("H5M3: Evolución temporal de PC2", fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend(loc='best', fontsize=10)
ax2.set_xlim(time.min(), time.max())

plt.tight_layout()
plt.savefig("pca_timeseries_h5m3.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.savefig("pca_timeseries_h5m3.pdf", bbox_inches="tight", facecolor='white')
print("✓ Guardado: pca_timeseries_h5m3.png/.pdf")
plt.show()

print("\n✓ Análisis completado exitosamente!")
